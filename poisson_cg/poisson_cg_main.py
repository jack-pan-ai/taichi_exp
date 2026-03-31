#!/usr/bin/env python3
"""
Taichi Poisson + CG (CPU/CUDA) — simple SpMV schedule.

SpMV matches EASIER Linsys and Kokkos `poisson_cg`:
  y[i] = Ac[i]*p[i] + sum_{e: src[e]=i} Af[e]*p[dst[e]]

**Simple** mode uses a separate edge buffer + gather kernel + scatter kernel + diagonal,
mirroring Kokkos `poisson_cg.cpp`. See `poisson_cg_main_fused.py` for the fused edge kernel.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _maybe_import_h5py():
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("h5py is required for HDF5 inputs.") from e
    return h5py


@dataclass(frozen=True)
class PoissonDiskArrays:
    src: np.ndarray
    dst: np.ndarray
    Ac: np.ndarray
    Af: np.ndarray
    b: np.ndarray

    @property
    def nc(self) -> int:
        return int(self.b.size)

    @property
    def ne(self) -> int:
        return int(self.src.size)


def load_from_hdf5(mesh_path: Path, poisson_path: Path) -> PoissonDiskArrays:
    h5py = _maybe_import_h5py()
    mesh_path = mesh_path.expanduser().resolve()
    poisson_path = poisson_path.expanduser().resolve()
    with h5py.File(mesh_path, "r") as mesh, h5py.File(poisson_path, "r") as p:
        src = np.asarray(mesh["src"][()], dtype=np.int64).reshape(-1)
        dst = np.asarray(mesh["dst"][()], dtype=np.int64).reshape(-1)
        Ac = np.asarray(p["Ac"][()], dtype=np.float64).reshape(-1)
        Af = np.asarray(p["Af"][()], dtype=np.float64).reshape(-1)
        b = np.asarray(p["b"][()], dtype=np.float64).reshape(-1)
    if src.shape != dst.shape or Af.shape != src.shape:
        raise ValueError("Inconsistent mesh/poisson shapes")
    if Ac.shape != b.shape:
        raise ValueError("Ac and b must have length nc")
    return PoissonDiskArrays(src=src, dst=dst, Ac=Ac, Af=Af, b=b)


def write_timing_csv(
    output_dir: str | None, variant: str, arch: str, iterations: int, seconds: float
) -> Path:
    # Match taichi/swe_main.py naming and CSV schema.
    ms_per_iter = seconds / iterations * 1000.0
    name = f"timing_{arch}_taichi.csv"
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / name
    else:
        csv_path = Path(name).resolve()
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
    parser.add_argument(
        "--threads",
        type=int,
        default=20,
        help="CPU thread count for Taichi CPU backend",
    )
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=-1)
    parser.add_argument("--profile-iters", type=int, default=-1)
    parser.add_argument("--output", type=str, help="directory for timing CSV")
    parser.add_argument("mesh", type=str, help="mesh HDF5 (src, dst)")
    parser.add_argument("poisson", type=str, help="Poisson HDF5 (Ac, Af, b)")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

    arrays = load_from_hdf5(Path(args.mesh), Path(args.poisson))
    nc, ne = arrays.nc, arrays.ne
    print(f"[taichi-poisson-cg] nc={nc}, ne={ne} (simple SpMV)")

    import taichi as ti  # noqa: WPS433

    arch = ti.cpu if args.arch == "cpu" else ti.cuda
    if args.arch == "cpu":
        ti.init(arch=arch, default_fp=ti.f64, cpu_max_num_threads=args.threads)
    else:
        ti.init(arch=arch, default_fp=ti.f64)

    src = ti.field(dtype=ti.i32, shape=(ne,))
    dst = ti.field(dtype=ti.i32, shape=(ne,))
    Ac = ti.field(dtype=ti.f64, shape=(nc,))
    Af = ti.field(dtype=ti.f64, shape=(ne,))
    b = ti.field(dtype=ti.f64, shape=(nc,))

    x = ti.field(dtype=ti.f64, shape=(nc,))
    r = ti.field(dtype=ti.f64, shape=(nc,))
    z = ti.field(dtype=ti.f64, shape=(nc,))
    p = ti.field(dtype=ti.f64, shape=(nc,))
    Ap = ti.field(dtype=ti.f64, shape=(nc,))
    edge_buf = ti.field(dtype=ti.f64, shape=(ne,))
    alpha_buf = ti.field(dtype=ti.f64, shape=())
    beta_buf = ti.field(dtype=ti.f64, shape=())
    reduce_out = ti.field(dtype=ti.f64, shape=())

    src.from_numpy(arrays.src.astype(np.int32, copy=False))
    dst.from_numpy(arrays.dst.astype(np.int32, copy=False))
    Ac.from_numpy(arrays.Ac)
    Af.from_numpy(arrays.Af)
    b.from_numpy(arrays.b)
    x.fill(0.0)

    @ti.kernel
    def spmv_gather():
        for e in range(ne):
            edge_buf[e] = Af[e] * p[dst[e]]

    @ti.kernel
    def zero_ap():
        for i in range(nc):
            Ap[i] = 0.0

    @ti.kernel
    def spmv_scatter():
        for e in range(ne):
            ti.atomic_add(Ap[src[e]], edge_buf[e])

    @ti.kernel
    def spmv_diag():
        for i in range(nc):
            Ap[i] += Ac[i] * p[i]

    def apply_A():
        zero_ap()
        spmv_gather()
        spmv_scatter()
        spmv_diag()

    @ti.kernel
    def dot_r_z():
        s = 0.0
        for i in range(nc):
            s += r[i] * z[i]
        reduce_out[None] = s

    @ti.kernel
    def dot_p_Ap():
        s = 0.0
        for i in range(nc):
            s += p[i] * Ap[i]
        reduce_out[None] = s

    @ti.kernel
    def norm_l2_b():
        s = 0.0
        for i in range(nc):
            s += b[i] * b[i]
        reduce_out[None] = ti.sqrt(s)

    @ti.kernel
    def norm_l2_r():
        s = 0.0
        for i in range(nc):
            s += r[i] * r[i]
        reduce_out[None] = ti.sqrt(s)

    def get_reduce() -> float:
        return float(reduce_out[None])

    @ti.kernel
    def vec_copy_r_to_z():
        for i in range(nc):
            z[i] = r[i]

    @ti.kernel
    def vec_copy_z_to_p():
        for i in range(nc):
            p[i] = z[i]

    @ti.kernel
    def vec_copy_x_to_p():
        for i in range(nc):
            p[i] = x[i]

    @ti.kernel
    def vec_scaled_add():
        for i in range(nc):
            x[i] += alpha_buf[None] * p[i]

    @ti.kernel
    def vec_update_r():
        for i in range(nc):
            r[i] -= alpha_buf[None] * Ap[i]

    @ti.kernel
    def vec_lincomb():
        for i in range(nc):
            p[i] = z[i] + beta_buf[None] * p[i]

    rzsum_store: list[float] = [0.0]

    @ti.kernel
    def init_r_from_b():
        for i in range(nc):
            r[i] = b[i] - Ap[i]

    @ti.kernel
    def final_residual_r():
        for i in range(nc):
            r[i] = b[i] - Ap[i]

    def init_cg(bnorm_holder: list[float]) -> None:
        # Match EASIER init exactly: r = b - A(x) with x initialized to zeros.
        vec_copy_x_to_p()
        apply_A()
        init_r_from_b()
        vec_copy_r_to_z()
        vec_copy_z_to_p()
        norm_l2_b()
        bnorm_holder[0] = get_reduce()

    def cg_step() -> float:
        apply_A()
        dot_r_z()
        rz = get_reduce()
        dot_p_Ap()
        pAp = get_reduce()
        alpha = rz / pAp
        alpha_buf[None] = alpha
        vec_scaled_add()
        vec_update_r()
        norm_l2_r()
        rn = get_reduce()
        rzsum_store[0] = rz
        return rn

    def cg_update() -> None:
        vec_copy_r_to_z()
        dot_r_z()
        rzn = get_reduce()
        beta = rzn / rzsum_store[0]
        beta_buf[None] = beta
        vec_lincomb()

    def sync():
        if args.arch == "cuda":
            ti.sync()

    if not args.profile:
        bnorm_h = [0.0]
        init_cg(bnorm_h)
        bnorm = bnorm_h[0]
        tol = max(args.rtol * bnorm, args.atol)
        iters = 0
        while True:
            rnorm = cg_step()
            if iters % 10 == 0:
                print(f"CG residual {rnorm} at iteration {iters}")
            if (not np.isnan(rnorm) and rnorm <= tol) or (args.maxiter > 0 and iters >= args.maxiter):
                break
            iters += 1
            cg_update()
        # Final reported residual should be ||b - A(x)||, not ||b - A(p)||.
        vec_copy_x_to_p()
        apply_A()
        final_residual_r()
        norm_l2_r()
        final_r = get_reduce()
        print(f"CG completed: iterations={iters} residual={final_r}")
        return

    # --- profile: one CG iteration = cg_step + cg_update (same as Kokkos) ---
    warmup = args.profile_warmup
    if warmup < 0:
        warmup = 5 if args.arch == "cpu" else 50
    iters = args.profile_iters
    if iters <= 0:
        iters = 20 if args.arch == "cpu" else 100

    bnorm_h = [0.0]
    init_cg(bnorm_h)

    print(f"Warming up {warmup} CG iterations")
    for _ in range(warmup):
        cg_step()
        cg_update()
    sync()

    print(f"Running {iters} CG iterations (step+update)")
    t0 = time.perf_counter()
    for _ in range(iters):
        cg_step()
        cg_update()
    sync()
    t1 = time.perf_counter()
    seconds = t1 - t0
    ms_per = seconds / iters * 1000.0
    print(
        f"Time for {iters} CG iterations: {seconds:.6f} s, {ms_per:.4f} ms/iter (arch={args.arch})"
    )
    csv_path = write_timing_csv(args.output, "simple", args.arch, iters, seconds)
    print(f"Writing profile results to: {csv_path}")


if __name__ == "__main__":
    main()
