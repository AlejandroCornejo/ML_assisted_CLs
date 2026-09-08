"""Batched direct closure and the same seven-point stress/tangent stencil."""
import numpy as np
from direct_hprom_ann_law import MAWDHPROMANN, FD_EPS
from reduced_stress_batch import ReducedStressBatch


class FastMAWDHPROMANN(MAWDHPROMANN):
    def __init__(self, work_dir=None):
        super().__init__(work_dir)
        self.batch_sig = ReducedStressBatch(self.rve, self.stress_assembly)

    def evaluate_stress_batch(self, E):
        from maw_lab import field_weights
        E = np.asarray(E, dtype=float).reshape(-1, 3)
        weights = field_weights(self.field_sig, E).T
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise RuntimeError('invalid adaptive MAW stress weights')
        self.weight_min = min(self.weight_min, float(weights.min()))
        self.weight_sum_error = max(self.weight_sum_error,
            float(np.max(np.abs(weights.sum(axis=1)-self.n_full_elements))))
        dec = self.decoder_sig
        # Only network values enter the stress. The unchanged FD stencil
        # differentiates the whole stress map, so decoder Jacobians are unused.
        h = (E-dec.mu_m)/dec.mu_s
        for W, b in zip(dec.Ws[:-1], dec.bs[:-1]):
            h = np.tanh(h @ W.T+b)
        N = h @ dec.Ws[-1].T+dec.bs[-1]
        d = (E @ dec.PhiMA.T + N @ dec.Phi_S.T)*self.mask_sig
        S = self.batch_sig.evaluate(E, d, weights)
        if not np.all(np.isfinite(S)):
            raise RuntimeError('nonfinite direct stress batch')
        return S

    def evaluate_stress(self, E):
        return self.evaluate_stress_batch(E)[0]

    def stress_and_tangent_batch(self, E):
        """Batch over macro points AND their seven-point FD stencils."""
        E = np.asarray(E, dtype=float).reshape(-1, 3)
        # Bound memory for callers beyond the worker chunks used by FE2.
        if len(E) > 128:
            pieces = [self.stress_and_tangent_batch(E[i:i+128]) for i in range(0, len(E), 128)]
            return tuple(np.concatenate([p[k] for p in pieces]) for k in range(3))
        h = FD_EPS*np.maximum(1., np.abs(E))
        stencil = np.repeat(E[:, None, :], 7, axis=1)
        for j in range(3):
            stencil[:, 1+2*j, j] += h[:, j]
            stencil[:, 2+2*j, j] -= h[:, j]
        S = self.evaluate_stress_batch(stencil.reshape(-1, 3)).reshape(len(E), 7, 3)
        C = np.swapaxes((S[:, 1::2]-S[:, 2::2])/(2*h[:, :, None]), 1, 2)
        if not np.all(np.isfinite(C)):
            raise RuntimeError('nonfinite direct tangent batch')
        return S[:, 0], C, E.copy()

    def stress_and_tangent(self, E, q_init=None, E_start=None, return_state=False):
        del q_init, E_start
        S, C, Q = self.stress_and_tangent_batch(np.asarray(E).reshape(1, 3))
        return (S[0], C[0], Q[0]) if return_state else (S[0], C[0])

    @property
    def ecm_metadata(self):
        return dict(super().ecm_metadata,
                    implementation='batched_closure_and_same_stress_fd_stencil',
                    maximum_macro_points_per_batch=128)
