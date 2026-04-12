#!/usr/bin/env python3
"""
Taichi SWE stepper (CPU/CUDA) - fused/optimized variant.

This version keeps the same RK4 time integration and the same physical flux
expressions as the EASIER SWE tutorial, but **fuses** the edge pipeline into a
single kernel to reduce memory traffic:

  face_reconstruct -> (u,v) -> flux -> scatter  ==> one pass over edges

The non-fused baseline remains in `taichi/swe_main.py`.
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import taichi as ti 

def _maybe_import_h5py():
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "h5py is required for HDF5 inputs. Install it or run with --sample."
        ) from e
    return h5py


@dataclass(frozen=True)
class SweDiskArrays:
    src: np.ndarray  # (ne,) int64
    dst: np.ndarray  # (ne,) int64
    bcells: np.ndarray  # (nbc,) int64
    alpha: np.ndarray  # (ne,) float64
    sx: np.ndarray  # (ne,) float64
    sy: np.ndarray  # (ne,) float64
    area: np.ndarray  # (nc,) float64
    h: np.ndarray  # (nc,) float64
    x: np.ndarray  # (nc,) float64
    y: np.ndarray  # (nc,) float64
    bsx: np.ndarray  # (nbc,) float64
    bsy: np.ndarray  # (nbc,) float64

    @property
    def nc(self) -> int:
        return int(self.h.size)

    @property
    def ne(self) -> int:
        return int(self.src.size)

    @property
    def nbc(self) -> int:
        return int(self.bcells.size)


def _require_1d(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    return arr


def validate_arrays(arrays: SweDiskArrays) -> None:
    if arrays.src.shape != arrays.dst.shape:
        raise ValueError("src and dst must have identical shapes")
    if arrays.alpha.shape != arrays.src.shape:
        raise ValueError("alpha must have shape (ne,)")
    if arrays.sx.shape != arrays.src.shape or arrays.sy.shape != arrays.src.shape:
        raise ValueError("sx/sy must have shape (ne,)")
    if arrays.area.shape != arrays.h.shape:
        raise ValueError("area and h must have shape (nc,)")
    if arrays.x.shape != arrays.h.shape or arrays.y.shape != arrays.h.shape:
        raise ValueError("x/y must have shape (nc,)")
    if arrays.bcells.size > 0:
        if arrays.bsx.shape != arrays.bcells.shape or arrays.bsy.shape != arrays.bcells.shape:
            raise ValueError("bsx/bsy must have shape (nbc,) matching bcells when bcells is non-empty")
    else:
        if arrays.bsx.size != 0 or arrays.bsy.size != 0:
            raise ValueError("bsx/bsy must be empty when bcells is empty")
    if np.any(arrays.area <= 0.0):
        raise ValueError("area must be positive")


def load_from_hdf5(mesh_path: Path, state_path: Path) -> SweDiskArrays:
    h5py = _maybe_import_h5py()
    mesh_path = mesh_path.expanduser().resolve()
    state_path = state_path.expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    with h5py.File(mesh_path, "r") as mesh, h5py.File(state_path, "r") as state:
        src = _require_1d("src", np.asarray(mesh["src"][()], dtype=np.int64))
        dst = _require_1d("dst", np.asarray(mesh["dst"][()], dtype=np.int64))
        bcells = _require_1d("bcells", np.asarray(mesh["bcells"][()], dtype=np.int64))

        alpha = _require_1d("alpha", np.asarray(state["alpha"][()], dtype=np.float64))
        area = _require_1d("area", np.asarray(state["area"][()], dtype=np.float64))
        sx = _require_1d("sx", np.asarray(state["sx"][()], dtype=np.float64))
        sy = _require_1d("sy", np.asarray(state["sy"][()], dtype=np.float64))
        h = _require_1d("h", np.asarray(state["h"][()], dtype=np.float64))
        x = _require_1d("x", np.asarray(state["x"][()], dtype=np.float64))
        y = _require_1d("y", np.asarray(state["y"][()], dtype=np.float64))

        if "bsx" in state:
            bsx = _require_1d("bsx", np.asarray(state["bsx"][()], dtype=np.float64))
        else:
            bsx = np.empty((0,), dtype=np.float64)
        if "bsy" in state:
            bsy = _require_1d("bsy", np.asarray(state["bsy"][()], dtype=np.float64))
        else:
            bsy = np.empty((0,), dtype=np.float64)

    arrays = SweDiskArrays(
        src=src,
        dst=dst,
        bcells=bcells,
        alpha=alpha,
        sx=sx,
        sy=sy,
        area=area,
        h=h,
        x=x,
        y=y,
        bsx=bsx,
        bsy=bsy,
    )
    validate_arrays(arrays)
    return arrays


def make_sample_arrays(nc: int = 8192, ne: int = 65536, nbc: int = 2048, seed: int = 0) -> SweDiskArrays:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, nc, size=(ne,), dtype=np.int64)
    dst = rng.integers(0, nc, size=(ne,), dtype=np.int64)
    bcells = rng.integers(0, nc, size=(nbc,), dtype=np.int64)

    alpha = rng.random((ne,), dtype=np.float64)
    sx = (rng.normal(0.0, 1.0, size=(ne,)).astype(np.float64)) * 0.01
    sy = (rng.normal(0.0, 1.0, size=(ne,)).astype(np.float64)) * 0.01

    area = (rng.random((nc,), dtype=np.float64) + 0.1).astype(np.float64)
    h = (rng.random((nc,), dtype=np.float64) + 1.0).astype(np.float64)
    x = rng.random((nc,), dtype=np.float64).astype(np.float64)
    y = rng.random((nc,), dtype=np.float64).astype(np.float64)

    bsx = (rng.normal(0.0, 1.0, size=(nbc,)).astype(np.float64)) * 0.01
    bsy = (rng.normal(0.0, 1.0, size=(nbc,)).astype(np.float64)) * 0.01

    arrays = SweDiskArrays(
        src=src,
        dst=dst,
        bcells=bcells,
        alpha=alpha,
        sx=sx,
        sy=sy,
        area=area,
        h=h,
        x=x,
        y=y,
        bsx=bsx,
        bsy=bsy,
    )
    validate_arrays(arrays)
    return arrays


def write_timing_csv(output_dir: str | None, arch: str, iterations: int, seconds: float) -> Path:
    ms_per_iter = seconds / iterations * 1000.0
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"timing_{arch}_taichi_fused.csv"
    else:
        csv_path = Path(f"timing_{arch}_taichi_fused.csv").resolve()

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["num_times", "seconds", "ms_per_iteration"])
        w.writerow([iterations, f"{seconds:.6f}", f"{ms_per_iter:.4f}"])
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=-1)
    parser.add_argument("--profile-iters", type=int, default=-1)
    parser.add_argument("--output", type=str, help="output directory for timing CSV and snapshots")
    parser.add_argument("--output-interval", type=int, default=10)

    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sample-nc", type=int, default=8192)
    parser.add_argument("--sample-ne", type=int, default=65536)
    parser.add_argument("--sample-nbc", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=0)

    parser.add_argument("mesh", nargs="?", type=str)
    parser.add_argument("state", nargs="?", type=str)
    args = parser.parse_args()

    if args.dt <= 0:
        raise SystemExit("--dt must be positive")
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    if args.interop_threads <= 0:
        raise SystemExit("--interop-threads must be positive")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.output_interval <= 0:
        raise SystemExit("--output-interval must be positive")
    if args.profile_warmup < -1:
        raise SystemExit("--profile-warmup must be >= -1")

    # Pin CPU threading (mirrors EASIER).
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_INTEROP_THREADS"] = str(args.interop_threads)

    if args.sample:
        arrays = make_sample_arrays(
            nc=args.sample_nc,
            ne=args.sample_ne,
            nbc=args.sample_nbc,
            seed=args.sample_seed,
        )
    else:
        if not args.mesh or not args.state:
            raise SystemExit("Provide mesh+state HDF5 paths, or use --sample")
        arrays = load_from_hdf5(Path(args.mesh), Path(args.state))

    arch = ti.cpu if args.arch == "cpu" else ti.cuda
    if args.arch == "cpu":
        ti.init(arch=arch, default_fp=ti.f64, kernel_profiler=True, #kernel_profiler=False, 
        cpu_max_num_threads=args.threads)
    else:
        ti.init(arch=arch, default_fp=ti.f64, kernel_profiler=False)

    nc, ne, nbc = arrays.nc, arrays.ne, arrays.nbc
    print(f"[taichi-swe-fused] nc={nc}, ne={ne}, nbc={nbc}")

    # Fields
    src = ti.field(dtype=ti.i32, shape=(ne,))
    dst = ti.field(dtype=ti.i32, shape=(ne,))
    bcells = ti.field(dtype=ti.i32, shape=(nbc,)) if nbc > 0 else None

    alpha = ti.field(dtype=ti.f64, shape=(ne,))
    sx = ti.field(dtype=ti.f64, shape=(ne,))
    sy = ti.field(dtype=ti.f64, shape=(ne,))
    bsx = ti.field(dtype=ti.f64, shape=(nbc,)) if nbc > 0 else None
    bsy = ti.field(dtype=ti.f64, shape=(nbc,)) if nbc > 0 else None

    area = ti.field(dtype=ti.f64, shape=(nc,))
    x = ti.field(dtype=ti.f64, shape=(nc,))
    y = ti.field(dtype=ti.f64, shape=(nc,))

    h = ti.field(dtype=ti.f64, shape=(nc,))
    uh = ti.field(dtype=ti.f64, shape=(nc,))
    vh = ti.field(dtype=ti.f64, shape=(nc,))

    dh1 = ti.field(dtype=ti.f64, shape=(nc,))
    du1 = ti.field(dtype=ti.f64, shape=(nc,))
    dv1 = ti.field(dtype=ti.f64, shape=(nc,))
    dh2 = ti.field(dtype=ti.f64, shape=(nc,))
    du2 = ti.field(dtype=ti.f64, shape=(nc,))
    dv2 = ti.field(dtype=ti.f64, shape=(nc,))
    dh3 = ti.field(dtype=ti.f64, shape=(nc,))
    du3 = ti.field(dtype=ti.f64, shape=(nc,))
    dv3 = ti.field(dtype=ti.f64, shape=(nc,))
    dh4 = ti.field(dtype=ti.f64, shape=(nc,))
    du4 = ti.field(dtype=ti.f64, shape=(nc,))
    dv4 = ti.field(dtype=ti.f64, shape=(nc,))

    tmp_h = ti.field(dtype=ti.f64, shape=(nc,))
    tmp_uh = ti.field(dtype=ti.f64, shape=(nc,))
    tmp_vh = ti.field(dtype=ti.f64, shape=(nc,))

    # Upload
    src.from_numpy(arrays.src.astype(np.int32, copy=False))
    dst.from_numpy(arrays.dst.astype(np.int32, copy=False))
    alpha.from_numpy(arrays.alpha)
    sx.from_numpy(arrays.sx)
    sy.from_numpy(arrays.sy)
    area.from_numpy(arrays.area)
    x.from_numpy(arrays.x)
    y.from_numpy(arrays.y)

    if nbc > 0:
        assert bcells is not None and bsx is not None and bsy is not None
        bcells.from_numpy(arrays.bcells.astype(np.int32, copy=False))
        bsx.from_numpy(arrays.bsx)
        bsy.from_numpy(arrays.bsy)

    h.from_numpy(arrays.h)
    uh.fill(0.0)
    vh.fill(0.0)

    dt = float(args.dt)

    @ti.kernel
    def zero3(a: ti.template(), b: ti.template(), c: ti.template()):
        for i in range(nc):
            a[i] = 0.0
            b[i] = 0.0
            c[i] = 0.0

    @ti.kernel
    def combine_scalar(base: ti.template(), delta: ti.template(), scale: ti.f64, out: ti.template()):
        for i in range(nc):
            out[i] = base[i] + scale * delta[i]

    @ti.kernel
    def scale_by_area3(dh: ti.template(), du: ti.template(), dv: ti.template()):
        for i in range(nc):
            inv_area = 1.0 / area[i]
            dh[i] = dh[i] * inv_area
            du[i] = du[i] * inv_area
            dv[i] = dv[i] * inv_area

    @ti.kernel
    def add_rk4_update(factor: ti.f64):
        for i in range(nc):
            h[i] += factor * (dh1[i] + dh2[i] + dh3[i] + dh4[i])
            uh[i] += factor * (du1[i] + du2[i] + du3[i] + du4[i])
            vh[i] += factor * (dv1[i] + dv2[i] + dv3[i] + dv4[i])

    @ti.kernel
    def delta_edges_fused(
        h_in: ti.template(),
        uh_in: ti.template(),
        vh_in: ti.template(),
        dh: ti.template(),
        du: ti.template(),
        dv: ti.template(),
    ):
        # Fused edge pipeline: face reconstruction + velocity + flux + scatter.
        for e in range(ne):
            left = src[e]
            right = dst[e]
            a = alpha[e]

            # Face reconstruction (EASIER face_reconstruct)
            h_face = (1.0 - a) * h_in[left] + a * h_in[right]
            uh_face = (1.0 - a) * uh_in[left] + a * uh_in[right]
            vh_face = (1.0 - a) * vh_in[left] + a * vh_in[right]

            # Face velocity (EASIER: u_f = uh_f/h_f, v_f = vh_f/h_f)
            u = uh_face / h_face
            v = vh_face / h_face

            # Flux terms (EASIER)
            h_square = 0.5 * h_face * h_face
            sx_val = sx[e]
            sy_val = sy[e]
            uh_sx = uh_face * sx_val
            vh_sy = vh_face * sy_val

            cell = right  # scatter to dst
            contrib_h = uh_sx + vh_sy
            contrib_uh = (u * uh_face + h_square) * sx_val + u * vh_sy
            contrib_vh = v * uh_sx + (v * vh_face + h_square) * sy_val

            ti.atomic_add(dh[cell], -contrib_h)
            ti.atomic_add(du[cell], -contrib_uh)
            ti.atomic_add(dv[cell], -contrib_vh)

    @ti.kernel
    def delta_boundary_fused(h_in: ti.template(), du: ti.template(), dv: ti.template()):
        for i in range(nbc):
            cell = bcells[i]
            h_cell = h_in[cell]
            h_square = 0.5 * h_cell * h_cell
            ti.atomic_add(du[cell], -h_square * bsx[i])
            ti.atomic_add(dv[cell], -h_square * bsy[i])

    def delta(h_in, uh_in, vh_in, delta_h, delta_uh, delta_vh):
        # Same math as EASIER's delta(), but fused for fewer memory passes.
        zero3(delta_h, delta_uh, delta_vh)
        delta_edges_fused(h_in, uh_in, vh_in, delta_h, delta_uh, delta_vh)
        if nbc > 0:
            delta_boundary_fused(h_in, delta_uh, delta_vh)
        scale_by_area3(delta_h, delta_uh, delta_vh)

    def forward():
        delta(h, uh, vh, dh1, du1, dv1)

        combine_scalar(h, dh1, 0.5 * dt, tmp_h)
        combine_scalar(uh, du1, 0.5 * dt, tmp_uh)
        combine_scalar(vh, dv1, 0.5 * dt, tmp_vh)
        delta(tmp_h, tmp_uh, tmp_vh, dh2, du2, dv2)

        combine_scalar(h, dh2, 0.5 * dt, tmp_h)
        combine_scalar(uh, du2, 0.5 * dt, tmp_uh)
        combine_scalar(vh, dv2, 0.5 * dt, tmp_vh)
        delta(tmp_h, tmp_uh, tmp_vh, dh3, du3, dv3)

        combine_scalar(h, dh3, dt, tmp_h)
        combine_scalar(uh, du3, dt, tmp_uh)
        combine_scalar(vh, dv3, dt, tmp_vh)
        delta(tmp_h, tmp_uh, tmp_vh, dh4, du4, dv4)

        add_rk4_update(dt / 6.0)

    def sync():
        if args.arch == "cuda":
            ti.sync()

    def snapshot_path(snapshot_id: int) -> Path:
        out_dir = Path(args.output or ".").expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"data{snapshot_id:03d}.npz"

    def write_snapshot(snapshot_id: int) -> None:
        path = snapshot_path(snapshot_id)
        np.savez(path, x=x.to_numpy(), y=y.to_numpy(), z=h.to_numpy())

    if not args.profile:
        for step_id in range(args.steps):
            forward()
            if args.output and (step_id % args.output_interval == 0):
                sync()
                write_snapshot(step_id // args.output_interval)
        sync()
        print(f"[done] ran {args.steps} steps (dt={dt}) on arch={args.arch}")
        return

    profile_warmup = int(args.profile_warmup)
    if profile_warmup < 0:
        profile_warmup = 5 if args.arch == "cpu" else 50

    profile_iters = int(args.profile_iters)
    if profile_iters <= 0:
        profile_iters = 20 if args.arch == "cpu" else 100

    if profile_iters <= 0:
        raise SystemExit("--profile-iters must be positive")

    print(f"Warming up {profile_warmup} times")
    for _ in range(profile_warmup):
        forward()
    sync()

    print(f"Running {profile_iters} times")
    t0 = time.perf_counter()
    for _ in range(profile_iters):
        forward()
    sync()
    t1 = time.perf_counter()
    # export profiling info
    # ti.profiler.print_scoped_profiler_info()
    # ti.profiler.print_kernel_profiler_info('trace')         # prints timing summary to console

    seconds = t1 - t0
    ms_per_iter = seconds / profile_iters * 1000.0
    print(
        f"Time to run forward() {profile_iters} times: {seconds:.6f} seconds, "
        f"{ms_per_iter:.4f} ms per iteration. (arch={args.arch})"
    )
    csv_path = write_timing_csv(args.output, args.arch, profile_iters, seconds)
    print(f"Writing profile results to: {csv_path}")


if __name__ == "__main__":
    main()

