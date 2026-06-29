#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import stage9_build_ecm_dataset_ann as stage9_base


def _has_option(opt_name):
    for arg in sys.argv[1:]:
        if arg == opt_name or arg.startswith(opt_name + "="):
            return True
    return False


def _inject_default(opt_name, opt_value):
    if not _has_option(opt_name):
        sys.argv.extend([opt_name, str(opt_value)])


if __name__ == "__main__":
    _inject_default("--ann-dir", "stage_7_ann_model_ls_newton")
    _inject_default("--out-dir", "stage_9_ecm_dataset_ann_ls")
    stage9_base.main()
