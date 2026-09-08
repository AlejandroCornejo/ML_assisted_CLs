#!/usr/bin/env bash
# Fresh complete simulations, sequential and single-threaded inside workers.
set -euo pipefail
cd /home/sares/ML_assisted_CLs_clean
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/coupon_fe2_mpl
export PYTHONPATH=coupon_fe2_paper/.pydeps:/home/sares/Kratos_Eigen_Check/bin/Release
for entry in optimized_r1 baseline_recheck optimized_r2 optimized_r3; do
    implementation=optimized
    if [[ "$entry" == baseline_recheck ]]; then implementation=baseline; fi
    for model in linear direct; do
        if [[ "$model" == linear ]]; then
            runner=run_hprom_fe2.py
            tag="clean_timing_ecm_w20_f100kn_${entry}"
            prefix=hprom_fe2
            extra=()
        else
            runner=run_hprom_ann_fe2.py
            tag="clean_timing_maw10_direct_w20_f100kn_${entry}"
            prefix=dhprom_ann_fe2
            extra=(--direct)
        fi
        result="coupon_fe2_paper/06_fe2/${prefix}_${tag}"
        if [[ -e "${result}.json" || -e "${result}.npz" || -e "${result}.log" ]]; then
            printf 'Refusing to overwrite benchmark: %s\n' "$result" >&2
            exit 1
        fi
        python3 -B "coupon_fe2_paper/06_fe2/${runner}" "${extra[@]}" \
            --implementation "$implementation" --workers 20 --tag "$tag" > "${result}.log" 2>&1
    done
done
