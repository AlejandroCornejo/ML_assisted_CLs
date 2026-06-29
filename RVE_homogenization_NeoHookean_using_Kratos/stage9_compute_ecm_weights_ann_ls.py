#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import stage9_compute_ecm_weights_ann as stage9_base


def _has_option(opt_name):
    for arg in sys.argv[1:]:
        if arg == opt_name or arg.startswith(opt_name + "="):
            return True
    return False


def _inject_default(opt_name, opt_value):
    if not _has_option(opt_name):
        sys.argv.extend([opt_name, str(opt_value)])


if __name__ == "__main__":
    _inject_default("--data-dir", "stage_9_ecm_dataset_ann_ls")
    _inject_default("--out-dir", "stage_9_hprom_ann_data_ls_independent_sum990")
    _inject_default("--ecm-coupling-mode", "independent")
    _inject_default("--constrain-sum-weights", "1")
    stage9_base.main()
