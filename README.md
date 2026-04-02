# Taichi Shallow Water Equation (SWE) Benchmark

This folder contains a Taichi implementation of the shallow water equation (SWE) time stepper, mirroring the data layout and math used in:

- `EASIER/tutorial/shallow_water_equation/swe_main.py`
- `kokkos/example/shallow_water_equation/shallow_water.cpp`
- `kokkos/example/shallow_water_equation/swe_export_to_binary.py` (conceptually: same datasets)

## Requirements

- A conda environment named `taichi` with `taichi`, `numpy`, and `h5py` installed.

If `h5py` is missing:

```bash
conda activate taichi
conda install -c conda-forge -y h5py
```

## Files in this folder

- **`swe_main.py`**: **simple / non-fused** Taichi SWE (**matched to Kokkos simple**) — reconstruct-to-edge arrays, then separate edge kernels for mass and momentum
- **`swe_main_fused.py`**: **optimized / fused** Taichi SWE (**matched to Kokkos fused**) — fused edge kernel computes reconstruction + velocities + flux + scatter in one pass
- **`taichi_swe.sh`**: sweep script that imitates `kokkos_swe.sh` (same `n` lists, same `dt=0.5/n`, warmup/iters)
- **`compare_snapshots.py`**: compares two `data000.csv` snapshot files (`x,y,h`)

## Run (sample mode, no input files)

This generates a small random-but-consistent mesh/state in memory and runs timing.

```bash
conda activate taichi
python /home/x_panq/taichi/swe_main.py --sample --arch=cpu --profile --output /home/x_panq/taichi/res
```

Fused variant:

```bash
conda activate taichi
python /home/x_panq/taichi/swe_main_fused.py --sample --arch=cpu --profile --output /home/x_panq/taichi/res_fused
```

If CUDA is available:

```bash
conda activate taichi
python /home/x_panq/taichi/swe_main.py --sample --arch=cuda --profile --output /home/x_panq/taichi/res
```

## “simple” vs “fused” modes (Taichi + Kokkos)

We solve the shallow-water equations using an explicit RK4 time integrator. Each RK4 stage
evaluates a discrete residual via edge-based numerical fluxes computed from face-reconstructed
states. For an edge $e$ connecting source cell $s(e)$ and destination cell $d(e)$, we reconstruct
any cell-centered quantity $\phi$ to the face using a linear blend
$\phi_f(e)=(1-\alpha_e)\phi_{s(e)}+\alpha_e\phi_{d(e)}$.

**Simple (non-fused) mode.** We implement one RK4 stage residual evaluation using the following
kernel decomposition (identical in Taichi and Kokkos):

- **Face reconstruction kernels**: compute face arrays $h_f, (uh)_f, (vh)_f$ over all edges.
- **Edge mass kernel**: accumulate the height residual $\Delta h$ from $(uh)_f\,s_x + (vh)_f\,s_y$.
- **Edge momentum kernel**: compute face velocities $u=(uh)_f/h_f$, $v=(vh)_f/h_f$ and accumulate
  $\Delta (uh)$ and $\Delta (vh)$ using the SWE flux expressions.
- **Boundary kernel**: add boundary contributions for $(uh)$ and $(vh)$.
- **Scaling kernel**: divide all residuals by cell area.

**Fused mode.** We fuse the per-edge pipeline (reconstruction $\rightarrow$ velocities $\rightarrow$
flux evaluation $\rightarrow$ scatter) into a single edge kernel, while keeping boundary and area
scaling as separate kernels. This reduces intermediate memory traffic by avoiding materialization of
face arrays and associated global-memory round-trips, and reduces kernel launch overhead.

**Why fusion is physically justified.** The above steps are algebraically one local “edge flux
evaluation” that produces the same numerical flux for each interface. Fusion changes only the
implementation schedule (how intermediate quantities are stored/streamed), not the PDE, the
discretization, or the RK4 stage ordering.
