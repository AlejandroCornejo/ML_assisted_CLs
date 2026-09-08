import sys
sys.path.insert(0, ".")
import numpy as np

from run_cruciform_fe2_claude import run_newton_fe2_cruciform

DELTA = 1.2
N_STEPS = 20

# n_body must be a multiple of 3 (arm_width_fraction=2/3 needs an exact
# n_width=2*n_body/3); n_arm_len scaled to keep the same aspect ratio as
# production (n_arm_len = 2/3 * n_body, matching n_body=12/n_arm_len=8).
CONFIGS = [(6, 4), (9, 6), (12, 8), (18, 12), (24, 16)]

print(f"{'n_body':>8}{'n_elem':>8}{'max|E11|':>12}{'max|E22|':>12}{'max|g12|':>12}{'max|sig11|':>14}{'max|sig22|':>14}{'max|sig12|':>14}{'wall(s)':>10}", flush=True)
for n_body, n_arm_len in CONFIGS:
    res = run_newton_fe2_cruciform(
        "pann_ickan", n_body=n_body, n_arm_len=n_arm_len, n_steps=N_STEPS,
        delta_x_final=DELTA, delta_y_final=DELTA, verbose=False, use_line_search=True, save_npz=True,
    )
    d = np.load(f"cruciform_results_pann_ickan_claude.npz")
    e_gp, s_gp = d["e_gp"], d["s_gp"]
    n_elem = d["tris"].shape[0]
    print(f"{n_body:>8}{n_elem:>8}{np.abs(e_gp[:,0]).max():>12.4f}{np.abs(e_gp[:,1]).max():>12.4f}"
          f"{np.abs(e_gp[:,2]).max():>12.4f}{np.abs(s_gp[:,0]).max():>14.4e}{np.abs(s_gp[:,1]).max():>14.4e}"
          f"{np.abs(s_gp[:,2]).max():>14.4e}{res['wall_time']:>10.1f}", flush=True)
    # rename per-resolution so later configs don't overwrite the file before the next iteration reads its own
    import shutil
    shutil.move("cruciform_results_pann_ickan_claude.npz", f"cruciform_results_pann_ickan_nbody{n_body}_claude.npz")

print("CONVERGENCE_STUDY_DONE_MARKER", flush=True)
