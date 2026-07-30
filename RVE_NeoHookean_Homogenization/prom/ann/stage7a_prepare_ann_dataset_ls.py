#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convenience wrapper for ANN-LS dataset preparation.
Reuses the LS coordinate construction from stage7a_prepare_rbf_dataset_ls.py
but defaults to ANN-LS output directory naming.
"""

import sys
import runpy


def _has_option(opt_name):
    for arg in sys.argv[1:]:
        if arg == opt_name or arg.startswith(opt_name + "="):
            return True
    return False


def _inject_default(opt_name, opt_value):
    if not _has_option(opt_name):
        sys.argv.extend([opt_name, str(opt_value)])


if __name__ == "__main__":
    _inject_default("--out-dir", "stage_7_ann_data_ls")
    runpy.run_module("stage7a_prepare_rbf_dataset_ls", run_name="__main__")
