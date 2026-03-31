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

## Run (HDF5 inputs from EASIER tutorial)

The script expects the same dataset names as the EASIER tutorial HDF5 files:

- From mesh HDF5: `src`, `dst`, `bcells`
- From state HDF5: `alpha`, `area`, `sx`, `sy`, `bsx`, `bsy`, `h`, `x`, `y`

Example:

```bash
conda activate taichi
python /home/x_panq/taichi/swe_main.py --arch=cpu --profile --output /home/x_panq/taichi/res \
  ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5
```

Fused variant:

```bash
conda activate taichi
python /home/x_panq/taichi/swe_main_fused.py --arch=cpu --profile --output /home/x_panq/taichi/res_fused \
  ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5
```

## Run the full sweep (imitates `kokkos_swe.sh`)

This runs the same `n` lists as Kokkos, using:

- **CPU**: warmup=5, iters=20
- **CUDA**: warmup=50, iters=100
- **dt**: `0.5/n`

```bash
conda activate taichi
bash /home/x_panq/taichi/taichi_swe.sh
```

Outputs go to:

- **Simple**: `/home/x_panq/taichi/res_simple/timing_<arch>_taichi.csv`
- **Fused**: `/home/x_panq/taichi/res_fused/timing_<arch>_taichi_fused.csv`

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

## Correctness check vs Kokkos (same input)

This runs **1 step** of Kokkos and Taichi on the same `~/.easier` dataset, writes `data000.csv` from both, then compares.

1) Export HDF5 -> Kokkos binary:

```bash
conda activate taichi
python /home/x_panq/kokkos/example/shallow_water_equation/swe_export_to_binary.py \
  ~/.easier/triangular_500.hdf5 ~/.easier/SW_500.hdf5 \
  --output-dir /home/x_panq/kokkos/data/swe_binary_500
```

2) Run Kokkos for 1 step and write a snapshot:

```bash
mkdir -p /home/x_panq/taichi/correct_kokkos_500
cd /home/x_panq/kokkos/example/shallow_water_equation
./build-omp/shallow_water_pipeline \
  --data /home/x_panq/kokkos/data/swe_binary_500 \
  --steps 1 --dt 0.001 \
  --output /home/x_panq/taichi/correct_kokkos_500 --output-interval 1
```

3) Run Taichi for 1 step and write a snapshot:

```bash
mkdir -p /home/x_panq/taichi/correct_taichi_500
conda activate taichi
python /home/x_panq/taichi/swe_main.py \
  --arch=cpu --dt 0.001 --steps 1 \
  --write-output --output-interval 1 \
  --output /home/x_panq/taichi/correct_taichi_500 \
  ~/.easier/triangular_500.hdf5 ~/.easier/SW_500.hdf5
```

4) Compare:

```bash
python /home/x_panq/taichi/compare_snapshots.py \
  /home/x_panq/taichi/correct_kokkos_500/data000.csv \
  /home/x_panq/taichi/correct_taichi_500/data000.csv
```

## Outputs

- Prints timing to stdout
- Appends a CSV row to:
  - `${output}/timing_taichi_${arch}.csv` if `--output` is set
  - `timing_taichi_${arch}.csv` otherwise

# taichi_exp
