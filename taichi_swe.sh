#! /bin/bash
set -euo pipefail

export OMP_NUM_THREADS=32
export OMP_PLACES="{0:32}"
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=verbose
export OMP_DISPLAY_AFFINITY=TRUE

N_CPU=(500 1000 1500 2000 2500 3000)
N_GPU=(500 1000 1500 2000 2500 3000)

ROOT_DIR="/home/x_panq/taichi"
SCRIPT_SIMPLE="${ROOT_DIR}/swe_main.py"
SCRIPT_FUSED="${ROOT_DIR}/swe_main_fused.py"

for n in "${N_CPU[@]}"
do
    delta_t=$(echo "scale=8; 0.5/$n" | bc)

    # Simple (non-fused)
    python "${SCRIPT_SIMPLE}" \
        --arch=cpu \
        --threads "${THREADS}" --interop-threads "${INTEROP_THREADS}" \
        --profile --dt "${delta_t}" --profile-warmup 5 --profile-iters 20 \
        --output "${ROOT_DIR}/res_simple" \
        "/home/x_panq/.easier/triangular_${n}.hdf5" \
        "/home/x_panq/.easier/SW_${n}.hdf5"

    # Fused / optimized
    python "${SCRIPT_FUSED}" \
        --arch=cpu \
        --threads "${THREADS}" --interop-threads "${INTEROP_THREADS}" \
        --profile --dt "${delta_t}" --profile-warmup 0 --profile-iters 20 \
        --output "${ROOT_DIR}/res_fused" \
        "/home/x_panq/.easier/triangular_${n}.hdf5" \
        "/home/x_panq/.easier/SW_${n}.hdf5"
done

for n in "${N_GPU[@]}"
do
    delta_t=$(echo "scale=8; 0.5/$n" | bc)

    python "${SCRIPT_SIMPLE}" \
        --arch=cuda \
        --threads "${THREADS}" --interop-threads "${INTEROP_THREADS}" \
        --profile --dt "${delta_t}" --profile-warmup 50 --profile-iters 100 \
        --output "${ROOT_DIR}/res_simple" \
        "/home/x_panq/.easier/triangular_${n}.hdf5" \
        "/home/x_panq/.easier/SW_${n}.hdf5"

    python "${SCRIPT_FUSED}" \
        --arch=cuda \
        --threads "${THREADS}" --interop-threads "${INTEROP_THREADS}" \
        --profile --dt "${delta_t}" --profile-warmup 50 --profile-iters 100 \
        --output "${ROOT_DIR}/res_fused" \
        "/home/x_panq/.easier/triangular_${n}.hdf5" \
        "/home/x_panq/.easier/SW_${n}.hdf5"
done

