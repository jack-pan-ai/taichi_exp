# Poisson + Conjugate Gradient (Taichi)

This folder ports the EASIER Poisson CG tutorial (`tutorial/poisson/poisson_main.py` +
`easier.numeric.solver.CG`) to Taichi, with the **same** operator:

\[
$(Ax)_i = A^c_i x_i + \sum_{e:\,\mathrm{src}(e)=i} A^f_e\, x_{\mathrm{dst}(e)}.$
\]

Two scripts are provided:

| File | SpMV schedule | Kokkos counterpart |
|------|----------------|----------------------|
| `poisson_cg_main.py` | Gather → edge buffer → scatter → diagonal | `poisson_cg.cpp` |
| `poisson_cg_main_fused.py` | Single edge loop: `Af* p[dst]` + atomic to `Ap[src]` | `poisson_cg_fused.cpp` |

The **fusion strategy is identical** between Kokkos and Taichi: avoid materializing the
per-edge product in global memory by fusing the gather and scatter (see section below). CG
vector operations and iteration control match `easier/numeric/solver/cg.py` (identity
preconditioner \(M=I\)).

---

## Prerequisites

- Python 3 with `numpy`, `h5py`, and `taichi` (same stack as `../swe_main.py`).
- Tutorial HDF5 files `~/.easier/triangular_*.hdf5` and `~/.easier/Poisson_*.hdf5`.

---

## Run

```bash
conda activate taichi   # if you use conda (see ../taichi_swe.sh)
python poisson_cg_main.py \
  --arch=cpu --threads 20 \
  ~/.easier/triangular_1000.hdf5 ~/.easier/Poisson_1000.hdf5

python poisson_cg_main_fused.py \
  --arch=cuda \
  ~/.easier/triangular_1000.hdf5 ~/.easier/Poisson_1000.hdf5
```

Profiling (times repeated **CG iterations**, each = one `step` + one `update` from `cg.py`):

```bash
python poisson_cg_main.py --arch=cpu --profile \
  --profile-warmup 5 --profile-iters 20 \
  --output ./prof \
  ~/.easier/triangular_2000.hdf5 ~/.easier/Poisson_2000.hdf5
```

CSV files: `timing_<arch>_taichi.csv` and `timing_<arch>_taichi_fused.csv` under `--output`.

Batch driver (same grid sizes as `kokkos_poisson_cg.sh`): `bash taichi_poisson_cg.sh`.

---

## Why fuse these kernels?

Each CG iteration spends most of its time in the SpMV \(Ap\) and in memory-bound vector ops.

**Simple schedule** (three edge/cell passes for the off-diagonal part):

1. **Gather** — for each edge \(e\), write `edge_buf[e] = Af[e] * p[dst[e]]`.
2. **Scatter** — for each edge \(e\), add `edge_buf[e]` to `Ap[src[e]]` (atomics).
3. **Diagonal** — for each cell \(i\), `Ap[i] += Ac[i] * p[i]`.

**Fused schedule** combines (1) and (2) into one edge parallel loop: compute
`Af[e] * p[dst[e]]` in registers and immediately `atomic_add` to `Ap[src[e]]`, removing the
edge buffer and one full read/write pass over edges—directly analogous to fusing face
reconstruction → flux → scatter in the shallow-water examples.

The diagonal pass remains a separate cell loop (no shared temporaries with edges). Fusion
does not change the discrete operator, only data movement and kernel count; CG convergence
behavior matches the unfused implementation up to floating-point reordering in atomics.
