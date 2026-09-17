#!/usr/bin/env python3
"""Render diagnostic plots from saved outputs, without additional FOM solves."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_pilot import plots

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
report = json.loads((args.directory / "report.json").read_text())
if report["status"] != "pilot complete":
    raise RuntimeError("Wait for the completed pilot")
mesh = np.load(args.directory / "fine.npz")
endpoint = np.load(args.directory / "fine_combined_x_endpoint.npz")
fields = mesh["xy"], mesh["triangles"], endpoint["node_displacement"], endpoint["P_micro"]
plots(report, fields, args.directory)
