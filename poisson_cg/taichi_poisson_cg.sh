#! /bin/bash
# Profile Taichi Poisson CG (simple vs fused) — mirrors taichi_swe.sh / kokkos_poisson_cg.sh.

set -euo pipefail

THREADS=20
INTEROP_THREADS=1
export OMP_NUM_THREADS=${THREADS}
export OMP_INTEROP_THREADS=${INTEROP_THREADS}

N_CPU=(1000 2000 3000 4000 5000)
N_GPU=(1000 2000 3000 4000 5000)

ROOT_DIR="/home/x_panq/taichi/poisson_cg"
SCRIPT_SIMPLE="${ROOT_DIR}/poisson_cg_main.py"
SCRIPT_FUSED="${ROOT_DIR}/poisson_cg_main_fused.py"

# shellcheck source=/dev/null
if [ -f /home/x_panq/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/x_panq/miniconda3/etc/profile.d/conda.sh
  conda activate taichi
fi

for n in "${N_CPU[@]}"
do
  python "${SCRIPT_SIMPLE}" \
    --arch=cpu \
    --threads "${THREADS}" --interop-threads "${INTEROP_THREADS}" \
    --profile --profile-warmup 5 --profile-iters 20 \
    --output "${ROOT_DIR}/profile_cpu" \
    "/home/x_panq/.easier/triangular_${n}.hdf5" \
    "/home/x_panq/.easier/Poisson_${n}.hdf5"

  python "${SCRIPT_FUSED}" \
    --arch=cpu \
    --threads "${THREADS}" --interop-threads "${INTEROP_THREADS}" \
    --profile --profile-warmup 5 --profile-iters 20 \
    --output "${ROOT_DIR}/profile_cpu" \
    "/home/x_panq/.easier/triangular_${n}.hdf5" \
    "/home/x_panq/.easier/Poisson_${n}.hdf5"
done

for n in "${N_GPU[@]}"
do
  python "${SCRIPT_SIMPLE}" \
    --arch=cuda \
    --threads "${THREADS}" --interop-threads "${INTEROP_THREADS}" \
    --profile --profile-warmup 50 --profile-iters 100 \
    --output "${ROOT_DIR}/profile_cuda" \
    "/home/x_panq/.easier/triangular_${n}.hdf5" \
    "/home/x_panq/.easier/Poisson_${n}.hdf5"

  python "${SCRIPT_FUSED}" \
    --arch=cuda \
    --threads "${THREADS}" --interop-threads "${INTEROP_THREADS}" \
    --profile --profile-warmup 50 --profile-iters 100 \
    --output "${ROOT_DIR}/profile_cuda" \
    "/home/x_panq/.easier/triangular_${n}.hdf5" \
    "/home/x_panq/.easier/Poisson_${n}.hdf5"
done
