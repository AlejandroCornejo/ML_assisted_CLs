"""Check the stress-dependent acoustic tensor against W(E)=E:E."""
import unittest
import numpy as np
from protocol.audit_robustness import acoustic, sqrt_c


class AcousticTests(unittest.TestCase):
    def test_geometric_term_and_engineering_shear(self):
        for e in (np.array([-.375, -.375, 0.]),
                  np.array([-.21875, -.21875, 0.]),
                  np.array([.12, -.03, .07])):
            f = sqrt_c(e)[0]
            E = np.array([[e[0], e[2]/2], [e[2]/2, e[1]]])
            stress = np.array([[2*e[0], 2*e[1], e[2]]])
            tangent = np.diag([2., 2., 1.])[None]
            for theta in (0., .4, 1.2):
                value, a, b = acoustic(f[None], stress, tangent, theta)
                H = np.outer(a[0], b)
                Edot = (f.T@H+H.T@f)/2
                direct = 2*np.sum(Edot**2)+np.sum(2*E*(H.T@H))
                self.assertAlmostEqual(value[0], direct, places=12)

    def test_positive_c_required(self):
        with self.assertRaises(ValueError):
            sqrt_c([-.5, 0., 0.])


if __name__ == '__main__':
    unittest.main()
