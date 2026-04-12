#! /bin/bash

export THREADS=20
export INTEROP_THREADS=1
export OMP_NUM_THREADS=20
export OMP_PLACES="{0:20}"
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=verbose
export OMP_DISPLAY_AFFINITY=TRUE

N_CPU=(1000 2000 3000 4000 5000)
N_GPU=(1000 2000 3000 4000 5000)

ROOT_DIR="${HOME}/taichi/poisson_cg"
SCRIPT_SIMPLE="${ROOT_DIR}/poisson_cg_main.py"
SCRIPT_FUSED="${ROOT_DIR}/poisson_cg_main_fused.py"

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
