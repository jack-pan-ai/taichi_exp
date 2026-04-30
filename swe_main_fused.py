#!/usr/bin/env python3
"""
Taichi SWE stepper (CPU/CUDA) with torch.fx/EASIER graph-faithful fusion.

This file intentionally preserves the exact 20 fused module boundaries from the
EASIER/torch.fx graph and mirrors the ordering used by
`kokkos/example/shallow_water_equation/shallow_water_fused.cpp`.
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


def make_sample_arrays(
    nc: int = 8192,
    ne: int = 65536,
    nbc: int = 2048,
    seed: int = 0,
) -> SweDiskArrays:
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
        ti.init(arch=arch, default_fp=ti.f64, kernel_profiler=True, cpu_max_num_threads=args.threads)
    else:
        ti.init(arch=arch, default_fp=ti.f64, kernel_profiler=False)

    nc, ne, nbc = arrays.nc, arrays.ne, arrays.nbc
    nbc_shape = max(nbc, 1)
    print(f"[taichi-swe-fused-graph] nc={nc}, ne={ne}, nbc={nbc}")

    # Index arrays:
    # - src/dst: face gather indices and face scatter destination (dst)
    # - bcells: boundary mapping / boundary scatter destination
    src = ti.field(dtype=ti.i32, shape=(ne,))
    dst = ti.field(dtype=ti.i32, shape=(ne,))
    bcells = ti.field(dtype=ti.i32, shape=(nbc_shape,))

    alpha = ti.field(dtype=ti.f64, shape=(ne,))
    sx = ti.field(dtype=ti.f64, shape=(ne,))
    sy = ti.field(dtype=ti.f64, shape=(ne,))
    bsx = ti.field(dtype=ti.f64, shape=(nbc_shape,))
    bsy = ti.field(dtype=ti.f64, shape=(nbc_shape,))

    area = ti.field(dtype=ti.f64, shape=(nc,))
    x = ti.field(dtype=ti.f64, shape=(nc,))
    y = ti.field(dtype=ti.f64, shape=(nc,))

    h = ti.field(dtype=ti.f64, shape=(nc,))
    uh = ti.field(dtype=ti.f64, shape=(nc,))
    vh = ti.field(dtype=ti.f64, shape=(nc,))

    truediv_2 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_3 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_4 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_7 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_8 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_9 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_12 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_13 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_14 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_17 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_18 = ti.field(dtype=ti.f64, shape=(nc,))
    truediv_19 = ti.field(dtype=ti.f64, shape=(nc,))

    add_10 = ti.field(dtype=ti.f64, shape=(nc,))
    add_11 = ti.field(dtype=ti.f64, shape=(nc,))
    add_12 = ti.field(dtype=ti.f64, shape=(nc,))
    add_23 = ti.field(dtype=ti.f64, shape=(nc,))
    add_24 = ti.field(dtype=ti.f64, shape=(nc,))
    add_25 = ti.field(dtype=ti.f64, shape=(nc,))
    add_36 = ti.field(dtype=ti.f64, shape=(nc,))
    add_37 = ti.field(dtype=ti.f64, shape=(nc,))
    add_38 = ti.field(dtype=ti.f64, shape=(nc,))

    scatter = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_1 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_2 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_3 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_4 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_5 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_6 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_7 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_8 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_9 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_10 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_11 = ti.field(dtype=ti.f64, shape=(nc,))

    scatter_b = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_b_1 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_b_2 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_b_3 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_b_4 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_b_5 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_b_6 = ti.field(dtype=ti.f64, shape=(nc,))
    scatter_b_7 = ti.field(dtype=ti.f64, shape=(nc,))

    src.from_numpy(arrays.src.astype(np.int32, copy=False))
    dst.from_numpy(arrays.dst.astype(np.int32, copy=False))
    alpha.from_numpy(arrays.alpha)
    sx.from_numpy(arrays.sx)
    sy.from_numpy(arrays.sy)
    area.from_numpy(arrays.area)
    x.from_numpy(arrays.x)
    y.from_numpy(arrays.y)

    if nbc > 0:
        bcells.from_numpy(arrays.bcells.astype(np.int32, copy=False))
        bsx.from_numpy(arrays.bsx)
        bsy.from_numpy(arrays.bsy)
    else:
        bcells.fill(0)
        bsx.fill(0.0)
        bsy.fill(0.0)

    h.from_numpy(arrays.h)
    uh.fill(0.0)
    vh.fill(0.0)

    half_dt = 0.00025
    full_dt = 0.0005
    rk_weight = 8.333333333333333e-05

    # torch.fx: easier0_select_reduce30
    @ti.kernel
    def easier0_select_reduce30():
        for i in range(nbc):
            cell = bcells[i]
            p = 0.5 * h[cell] * h[cell]
            ti.atomic_add(scatter_b[cell], p * bsx[i])
            ti.atomic_add(scatter_b_1[cell], p * bsy[i])

    # torch.fx: easier1_select_reduce16
    @ti.kernel
    def easier1_select_reduce16():
        for e in range(ne):
            left = src[e]
            right = dst[e]
            a = alpha[e]

            h_face = (1.0 - a) * h[left] + a * h[right]
            uh_face = (1.0 - a) * uh[left] + a * uh[right]
            vh_face = (1.0 - a) * vh[left] + a * vh[right]

            u = uh_face / h_face
            v = vh_face / h_face

            mass_flux = uh_face * sx[e] + vh_face * sy[e]
            xmom_flux = (u * uh_face + 0.5 * h_face * h_face) * sx[e] + (u * vh_face) * sy[e]
            ymom_flux = (v * uh_face) * sx[e] + (v * vh_face + 0.5 * h_face * h_face) * sy[e]

            ti.atomic_add(scatter[right], mass_flux)
            ti.atomic_add(scatter_1[right], xmom_flux)
            ti.atomic_add(scatter_2[right], ymom_flux)

    # torch.fx: easier2_map35
    @ti.kernel
    def easier2_map35():
        for i in range(nc):
            truediv_2[i] = -scatter[i] / area[i]
            add_10[i] = h[i] + half_dt * truediv_2[i]

    # torch.fx: easier3_map47
    @ti.kernel
    def easier3_map47():
        for i in range(nc):
            truediv_3[i] = -(scatter_1[i] + scatter_b[i]) / area[i]
            add_11[i] = uh[i] + half_dt * truediv_3[i]

    # torch.fx: easier4_map59
    @ti.kernel
    def easier4_map59():
        for i in range(nc):
            truediv_4[i] = -(scatter_2[i] + scatter_b_1[i]) / area[i]
            add_12[i] = vh[i] + half_dt * truediv_4[i]

    # torch.fx: easier5_select_reduce92
    @ti.kernel
    def easier5_select_reduce92():
        for i in range(nbc):
            cell = bcells[i]
            p = 0.5 * add_10[cell] * add_10[cell]
            ti.atomic_add(scatter_b_2[cell], p * bsx[i])
            ti.atomic_add(scatter_b_3[cell], p * bsy[i])

    # torch.fx: easier6_select_reduce80
    @ti.kernel
    def easier6_select_reduce80():
        for e in range(ne):
            left = src[e]
            right = dst[e]
            a = alpha[e]

            h_face = (1.0 - a) * add_10[left] + a * add_10[right]
            uh_face = (1.0 - a) * add_11[left] + a * add_11[right]
            vh_face = (1.0 - a) * add_12[left] + a * add_12[right]

            u = uh_face / h_face
            v = vh_face / h_face

            mass_flux = uh_face * sx[e] + vh_face * sy[e]
            xmom_flux = (u * uh_face + 0.5 * h_face * h_face) * sx[e] + (u * vh_face) * sy[e]
            ymom_flux = (v * uh_face) * sx[e] + (v * vh_face + 0.5 * h_face * h_face) * sy[e]

            ti.atomic_add(scatter_4[right], xmom_flux)
            ti.atomic_add(scatter_3[right], mass_flux)
            ti.atomic_add(scatter_5[right], ymom_flux)

    # torch.fx: easier7_map97
    @ti.kernel
    def easier7_map97():
        for i in range(nc):
            truediv_7[i] = -scatter_3[i] / area[i]
            add_23[i] = h[i] + half_dt * truediv_7[i]

    # torch.fx: easier8_map118
    @ti.kernel
    def easier8_map118():
        for i in range(nc):
            truediv_9[i] = -(scatter_5[i] + scatter_b_3[i]) / area[i]
            add_25[i] = vh[i] + half_dt * truediv_9[i]

    # torch.fx: easier9_map107
    @ti.kernel
    def easier9_map107():
        for i in range(nc):
            truediv_8[i] = -(scatter_4[i] + scatter_b_2[i]) / area[i]
            add_24[i] = uh[i] + half_dt * truediv_8[i]

    # torch.fx: easier10_select_reduce151
    @ti.kernel
    def easier10_select_reduce151():
        for i in range(nbc):
            cell = bcells[i]
            p = 0.5 * add_23[cell] * add_23[cell]
            ti.atomic_add(scatter_b_4[cell], p * bsx[i])
            ti.atomic_add(scatter_b_5[cell], p * bsy[i])

    # torch.fx: easier11_select_reduce139
    @ti.kernel
    def easier11_select_reduce139():
        for e in range(ne):
            left = src[e]
            right = dst[e]
            a = alpha[e]

            h_face = (1.0 - a) * add_23[left] + a * add_23[right]
            uh_face = (1.0 - a) * add_24[left] + a * add_24[right]
            vh_face = (1.0 - a) * add_25[left] + a * add_25[right]

            u = uh_face / h_face
            v = vh_face / h_face

            mass_flux = uh_face * sx[e] + vh_face * sy[e]
            xmom_flux = (u * uh_face + 0.5 * h_face * h_face) * sx[e] + (u * vh_face) * sy[e]
            ymom_flux = (v * uh_face) * sx[e] + (v * vh_face + 0.5 * h_face * h_face) * sy[e]

            ti.atomic_add(scatter_7[right], xmom_flux)
            ti.atomic_add(scatter_6[right], mass_flux)
            ti.atomic_add(scatter_8[right], ymom_flux)

    # torch.fx: easier12_map177
    @ti.kernel
    def easier12_map177():
        for i in range(nc):
            truediv_14[i] = -(scatter_8[i] + scatter_b_5[i]) / area[i]
            add_38[i] = vh[i] + full_dt * truediv_14[i]

    # torch.fx: easier13_map166
    @ti.kernel
    def easier13_map166():
        for i in range(nc):
            truediv_13[i] = -(scatter_7[i] + scatter_b_4[i]) / area[i]
            add_37[i] = uh[i] + full_dt * truediv_13[i]

    # torch.fx: easier14_map156
    @ti.kernel
    def easier14_map156():
        for i in range(nc):
            truediv_12[i] = -scatter_6[i] / area[i]
            add_36[i] = h[i] + full_dt * truediv_12[i]

    # torch.fx: easier15_select_reduce198
    @ti.kernel
    def easier15_select_reduce198():
        for e in range(ne):
            left = src[e]
            right = dst[e]
            a = alpha[e]

            h_face = (1.0 - a) * add_36[left] + a * add_36[right]
            uh_face = (1.0 - a) * add_37[left] + a * add_37[right]
            vh_face = (1.0 - a) * add_38[left] + a * add_38[right]

            u = uh_face / h_face
            v = vh_face / h_face

            mass_flux = uh_face * sx[e] + vh_face * sy[e]
            xmom_flux = (u * uh_face + 0.5 * h_face * h_face) * sx[e] + (u * vh_face) * sy[e]
            ymom_flux = (v * uh_face) * sx[e] + (v * vh_face + 0.5 * h_face * h_face) * sy[e]

            ti.atomic_add(scatter_10[right], xmom_flux)
            ti.atomic_add(scatter_9[right], mass_flux)
            ti.atomic_add(scatter_11[right], ymom_flux)

    # torch.fx: easier16_select_reduce210
    @ti.kernel
    def easier16_select_reduce210():
        for i in range(nbc):
            cell = bcells[i]
            p = 0.5 * add_36[cell] * add_36[cell]
            ti.atomic_add(scatter_b_6[cell], p * bsx[i])
            ti.atomic_add(scatter_b_7[cell], p * bsy[i])

    # torch.fx: easier17_map239
    @ti.kernel
    def easier17_map239():
        for i in range(nc):
            truediv_17[i] = -scatter_9[i] / area[i]
            h[i] += rk_weight * (truediv_2[i] + truediv_7[i] + truediv_12[i] + truediv_17[i])

    # torch.fx: easier18_map249
    @ti.kernel
    def easier18_map249():
        for i in range(nc):
            truediv_19[i] = -(scatter_11[i] + scatter_b_7[i]) / area[i]
            vh[i] += rk_weight * (truediv_4[i] + truediv_9[i] + truediv_14[i] + truediv_19[i])

    # torch.fx: easier19_map244
    @ti.kernel
    def easier19_map244():
        for i in range(nc):
            truediv_18[i] = -(scatter_10[i] + scatter_b_6[i]) / area[i]
            uh[i] += rk_weight * (truediv_3[i] + truediv_8[i] + truediv_13[i] + truediv_18[i])

    def forward() -> None:
        scatter_b.fill(0.0)
        scatter_b_1.fill(0.0)
        easier0_select_reduce30()

        scatter.fill(0.0)
        scatter_1.fill(0.0)
        scatter_2.fill(0.0)
        easier1_select_reduce16()

        easier2_map35()
        easier3_map47()
        easier4_map59()

        scatter_b_2.fill(0.0)
        scatter_b_3.fill(0.0)
        easier5_select_reduce92()

        scatter_4.fill(0.0)
        scatter_3.fill(0.0)
        scatter_5.fill(0.0)
        easier6_select_reduce80()

        easier7_map97()
        easier8_map118()
        easier9_map107()

        scatter_b_4.fill(0.0)
        scatter_b_5.fill(0.0)
        easier10_select_reduce151()

        scatter_7.fill(0.0)
        scatter_6.fill(0.0)
        scatter_8.fill(0.0)
        easier11_select_reduce139()

        easier12_map177()
        easier13_map166()
        easier14_map156()

        scatter_10.fill(0.0)
        scatter_9.fill(0.0)
        scatter_11.fill(0.0)
        easier15_select_reduce198()

        scatter_b_6.fill(0.0)
        scatter_b_7.fill(0.0)
        easier16_select_reduce210()

        easier17_map239()
        easier18_map249()
        easier19_map244()

    def sync() -> None:
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
        print(f"[done] ran {args.steps} steps (dt=0.0005) on arch={args.arch}")
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
