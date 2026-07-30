#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import stage9_build_ecm_dataset_ann as stage9_base

_HERE = Path(__file__).resolve().parent
while not (_HERE / "core").is_dir():
    _HERE = _HERE.parent
_DEFAULT_ANN_DIR = str(_HERE / "prom" / "ann" / "stage_7_ann_model_ls")


def _has_option(opt_name):
    for arg in sys.argv[1:]:
        if arg == opt_name or arg.startswith(opt_name + "="):
            return True
    return False


def _inject_default(opt_name, opt_value):
    if not _has_option(opt_name):
        sys.argv.extend([opt_name, str(opt_value)])


if __name__ == "__main__":
    _inject_default("--ann-dir", _DEFAULT_ANN_DIR)
    _inject_default("--out-dir", "stage_9_ecm_dataset_ann_ls")
    stage9_base.main()
