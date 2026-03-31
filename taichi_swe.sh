#! /bin/bash

set -euo pipefail

THREADS=20
INTEROP_THREADS=1

export OMP_NUM_THREADS=${THREADS}
export OMP_INTEROP_THREADS=${INTEROP_THREADS}
# export OMP_PROC_BIND=close
# export OMP_PLACES=cores

N_CPU=(500 1000 2000 3000 4000 5000)
N_GPU=(500 750 1000 1250 1500 1750 2000 2250 2500)

# N_CPU=(500 1000 1500 2000 2500 3000)
# N_CPU=(2000)
# N_GPU=(500 1000 1500 2000 2500 3000)


ROOT_DIR="/home/x_panq/taichi"
SCRIPT_SIMPLE="${ROOT_DIR}/swe_main.py"
SCRIPT_FUSED="${ROOT_DIR}/swe_main_fused.py"

source /home/x_panq/miniconda3/etc/profile.d/conda.sh
conda activate taichi


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

