"""Pilot-only affine predictor and logged step-halving around the unchanged FOM.

q is the independent TOTAL displacement, not a fluctuation. Updating q by
delta(F) X at independent coordinates updates the affine field everywhere,
including the periodic jumps carried by g. It changes the Newton initial guess,
not equilibrium, constitutive coefficients, constraints or stopping tolerances.
"""
import numpy as np
from run_pilot import cfg, field_stats, pf


class ContinuationFailure(RuntimeError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


def solve(rve, target, max_increment=0.01, min_increment=1e-6,
          start=None, q_start=None):
    from fom_solver_rve import DeformationGradientFromGreenLagrange2D as gradient
    target = np.asarray(target, dtype=float)
    initial = np.zeros(3) if start is None else np.asarray(start, dtype=float)
    q = np.zeros(rve.n_ind) if q_start is None else np.asarray(q_start).copy()
    length = float(np.linalg.norm(target-initial))
    if not (0 < min_increment <= max_increment) or not np.isfinite(length):
        raise ValueError("Invalid increment controls or target")
    attempts, progress = [], 0.0
    step = 1.0 if length == 0 else min(1.0, max_increment/length)
    old_density = pf.SUBSTEPS_PER_UNIT_STRAIN
    pf.SUBSTEPS_PER_UNIT_STRAIN = 1.0  # one Newton increment per external attempt
    try:
        while progress < 1.0:
            if len(attempts) >= 2000:
                raise ContinuationFailure("Continuation attempt budget exhausted", attempts)
            next_progress = min(1.0, progress+step)
            previous = initial + progress*(target-initial)
            trial = initial + next_progress*(target-initial)
            delta_F = gradient(trial)-gradient(previous)
            affine = np.einsum("ij,dj->di", delta_F, rve.ind_xy)[
                np.arange(rve.n_ind), rve.ind_comp]
            guess = q+affine
            # Explicitly verify the affine-predictor identity in full DOF space.
            actual = rve.T @ affine + rve._g(trial)-rve._g(previous)
            expected = np.einsum("ij,dj->di", delta_F, rve.dof_xy)[
                np.arange(rve.n_dof), rve.dof_comp]
            error = float(np.max(np.abs(actual-expected)))
            if error > 1e-10:
                raise ContinuationFailure("Affine predictor does not reproduce delta(F) X", attempts)
            record = dict(strain=trial.tolist(), increment_norm=float(np.linalg.norm(trial-previous)),
                          affine_identity_max_error=error, ok=False)
            attempts.append(record)
            try:
                stress, trial_q = rve.solve(trial, u_ind_init=guess, E_start=previous)
                stats, _u, _pk1 = field_stats(rve, trial, trial_q)
                if stats["relative_reduced_residual"] > 1e-7:
                    raise RuntimeError("Increment converged by displacement but failed residual screen")
                record.update(ok=True, min_micro_J=stats["min_micro_J"],
                              relative_reduced_residual=stats["relative_reduced_residual"])
                q, progress = trial_q, next_progress
                if length > 0:
                    step = min(2*step, max_increment/length)
            except Exception as exc:
                record["error"] = repr(exc)
                step /= 2
                if step*length < min_increment:
                    raise ContinuationFailure("Step halving reached minimum increment", attempts) from exc
        return stress, q, attempts
    finally:
        pf.SUBSTEPS_PER_UNIT_STRAIN = old_density
