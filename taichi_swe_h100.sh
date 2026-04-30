#! /bin/bash
set -euo pipefail

HOEM_DIR=/mnt/local-fast/qpan
ROOT_DIR=/mnt/local-fast/qpan/taichi_exp

export THREADS=40
export OMP_PLACES="{0:40}"
export OMP_PROC_BIND=close
# export OMP_DISPLAY_ENV=verbose
# export OMP_DISPLAY_AFFINITY=TRUE

# N_CPU=(1000 2000 3000 4000)
N_GPU=(1000 2000 3000 4000)

SCRIPT_SIMPLE="${ROOT_DIR}/swe_main.py"
SCRIPT_FUSED="${ROOT_DIR}/swe_main_fused.py"

for n in "${N_CPU[@]}"
do
    delta_t=$(echo "scale=8; 0.5/$n" | bc)

    # Simple (non-fused)
    python "${SCRIPT_SIMPLE}" \
        --arch=cpu \
        --threads "${THREADS}" --interop-threads 1 \
        --profile --dt "${delta_t}" --profile-warmup 5 --profile-iters 20 \
        --output "${ROOT_DIR}/res_simple" \
        "${HOEM_DIR}/.easier/triangular_${n}.hdf5" \
        "${HOEM_DIR}/.easier/SW_${n}.hdf5"

    # Fused / optimized
    python "${SCRIPT_FUSED}" \
        --arch=cpu \
        --threads "${THREADS}" --interop-threads 1 \
        --profile --dt "${delta_t}" --profile-warmup 0 --profile-iters 20 \
        --output "${ROOT_DIR}/res_fused" \
        "${HOEM_DIR}/.easier/triangular_${n}.hdf5" \
        "${HOEM_DIR}/.easier/SW_${n}.hdf5"
done

for n in "${N_GPU[@]}"
do
    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    echo "Running CUDA with n=${n}, delta_t=${delta_t}"

    python "${SCRIPT_SIMPLE}" \
        --arch=cuda \
        --threads "${THREADS}" --interop-threads 1 \
        --profile --dt "${delta_t}" --profile-warmup 50 --profile-iters 100 \
        --output "${ROOT_DIR}/res_simple" \
        "${HOEM_DIR}/.easier/triangular_${n}.hdf5" \
        "${HOEM_DIR}/.easier/SW_${n}.hdf5"

    python "${SCRIPT_FUSED}" \
        --arch=cuda \
        --threads "${THREADS}" --interop-threads 1 \
        --profile --dt "${delta_t}" --profile-warmup 50 --profile-iters 100 \
        --output "${ROOT_DIR}/res_fused" \
        "${HOEM_DIR}/.easier/triangular_${n}.hdf5" \
        "${HOEM_DIR}/.easier/SW_${n}.hdf5"
done

HOEM_DIR=/mnt/local-fast/qpan
ROOT_DIR=/mnt/local-fast/qpan/taichi_exp
SCRIPT_SIMPLE="${ROOT_DIR}/swe_main.py"
SCRIPT_FUSED="${ROOT_DIR}/swe_main_fused.py"

export OMP_NUM_THREADS=40 & nsys profile -o taichi --stats=true  --trace=cuda,osrt,nvtx,openmp,mpi,cublas    --cudabacktrace=all  --force-overwrite=true --delay 5  \
python "${SCRIPT_FUSED}" \
    --arch=cuda \
    --threads 40 --interop-threads 1 \
    --profile --dt 0.0005 --profile-warmup 50 --profile-iters 100 \
    --output "${ROOT_DIR}/res_fused" \
    "${HOEM_DIR}/.easier/triangular_2000.hdf5" \
    "${HOEM_DIR}/.easier/SW_2000.hdf5"
