#!/usr/bin/env bash
# Sequential complete FE2 runs: never overlap timing jobs.
set -euo pipefail
cd /home/sares/ML_assisted_CLs_clean
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/coupon_fe2_mpl
export PYTHONPATH=coupon_fe2_paper/.pydeps:/home/sares/Kratos_Eigen_Check/bin/Release
for entry in optimized_r1 baseline_recheck optimized_r2 optimized_r3; do
    implementation=optimized
    if [[ "$entry" == baseline_recheck ]]; then implementation=baseline; fi
    tag="clean_timing_maw10_w20_f100kn_${entry}"
    result="coupon_fe2_paper/06_fe2/hprom_ann_fe2_${tag}"
    if [[ -e "${result}.json" || -e "${result}.npz" || -e "${result}.log" ]]; then
        printf 'Refusing to overwrite existing benchmark: %s\n' "$result" >&2
        exit 1
    fi
    python3 -B coupon_fe2_paper/06_fe2/run_hprom_ann_fe2.py \
        --implementation "$implementation" --workers 20 --tag "$tag" > "${result}.log" 2>&1
done
