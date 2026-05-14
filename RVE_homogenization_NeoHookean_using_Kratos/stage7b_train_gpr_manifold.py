#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backward-compatible entrypoint for Stage 7b GPR.

This now routes to the true sparse-GP pipeline.
Use this script exactly as stage7b_train_sparse_gpr_manifold.py.
"""

from stage7b_train_sparse_gpr_manifold import main


if __name__ == "__main__":
    main()
