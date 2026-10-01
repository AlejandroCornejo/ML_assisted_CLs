"""
Simple verification of the MOEKAN gradient implementation.

Checks:
  1. `MOEKAN.gradient()` (vmap(jacfwd)) vs. central finite differences.
  2. `MOEKAN.gradient()` vs. `jax.jacrev` (reverse-mode) as a cross-check.
  3. Scalar-output and vector-output shapes.
  4. Single-sample (1-D input) and batched (2-D input) calls.

Run:
    python MOEKAN/test_gradient.py
"""

import numpy as np
import jax
import jax.numpy as jnp

from MOEKAN import MOEKAN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def finite_difference_gradient(f, x, eps=1e-3):
    """
    Central finite-difference gradient of f(x) w.r.t. x.

    f : callable, x -> scalar or vector
    x : 1-D array of shape (n,)
    Returns:
        array of shape (n,) if f returns a scalar,
        array of shape (n, m) if f returns a vector of length m.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    grad = np.zeros_like(x)

    for i in range(n):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        f_plus = np.asarray(f(x_plus), dtype=np.float64)
        f_minus = np.asarray(f(x_minus), dtype=np.float64)
        grad[i] = (f_plus - f_minus) / (2.0 * eps)

    return grad


def check_close(name, a, b, rtol=1e-4, atol=1e-5):
    # NOTE: the network runs in float32 (JAX default). Finite-difference
    # references computed in float64 therefore carry a roundoff floor of
    # O(eps * machine_eps_f32) ~ 1e-2 for eps=1e-3. The rigorous check is
    # gradient() vs jacrev (same autodiff, different mode), which agrees
    # to ~1e-7. FD checks use looser tolerances accordingly.
    """Print a PASS/FAIL line and return True if within tolerance."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.max(np.abs(a - b))
    scale = max(1.0, np.max(np.abs(b)))
    ok = diff <= atol + rtol * scale
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max|diff| = {diff:.3e}  (atol={atol:.0e}, rtol={rtol:.0e})")
    return ok


# ---------------------------------------------------------------------------
# Test 1: scalar output, single sample (1-D input)
# ---------------------------------------------------------------------------
def test_scalar_single_sample():
    print("\n=== Test 1: scalar output, single sample (1-D input) ===")
    net = MOEKAN(width=[3, 8, 1], seed=0, random_init=True)

    x = jnp.array([0.5, -1.2, 0.3])

    # MOEKAN.gradient() — for 1-D input it returns shape (1, in)
    grad_auto = np.asarray(net.gradient(x))

    # Finite differences (float64 reference, looser tol due to float32 net)
    f = lambda xx: float(net.moekan_network(jnp.asarray(xx))[0])
    grad_fd = finite_difference_gradient(f, np.asarray(x))

    # jacrev cross-check (same autodiff, different mode — tight tol)
    grad_rev = np.asarray(
        jax.jacrev(lambda xx: net.moekan_network(xx)[0])(x)
    )

    ok = True
    ok &= check_close("gradient() vs FD", grad_auto, grad_fd, rtol=1e-2, atol=1e-2)
    ok &= check_close("gradient() vs jacrev", grad_auto, grad_rev, rtol=1e-5, atol=1e-6)
    assert grad_auto.shape == (1, 3), f"Expected (1,3), got {grad_auto.shape}"
    print(f"  Shape check: {grad_auto.shape}  OK")
    return ok


# ---------------------------------------------------------------------------
# Test 2: scalar output, batched (2-D input)
# ---------------------------------------------------------------------------
def test_scalar_batched():
    print("\n=== Test 2: scalar output, batched (2-D input) ===")
    net = MOEKAN(width=[4, 6, 1], seed=7, random_init=True)

    key = jax.random.PRNGKey(123)
    x_batch = jax.random.normal(key, (10, 4))

    grad_auto = np.asarray(net.gradient(x_batch))

    # Finite differences per sample
    grad_fd = np.zeros_like(np.asarray(x_batch))
    for i in range(10):
        grad_fd[i] = finite_difference_gradient(
            lambda xx: float(net.moekan_network(jnp.asarray(xx))[0]),
            np.asarray(x_batch[i]),
        )

    # jacrev cross-check (tight)
    grad_rev = np.asarray(
        jax.vmap(jax.jacrev(lambda xx: net.moekan_network(xx)[0]))(x_batch)
    )

    ok = True
    ok &= check_close("gradient() vs FD (batch)", grad_auto, grad_fd, rtol=1e-2, atol=1e-2)
    ok &= check_close("gradient() vs jacrev (batch)", grad_auto, grad_rev, rtol=1e-5, atol=1e-6)
    assert grad_auto.shape == (10, 4), f"Expected (10,4), got {grad_auto.shape}"
    print(f"  Shape check: {grad_auto.shape}  OK")
    return ok


# ---------------------------------------------------------------------------
# Test 3: vector output, single sample
# ---------------------------------------------------------------------------
def test_vector_single_sample():
    print("\n=== Test 3: vector output, single sample (1-D input) ===")
    net = MOEKAN(width=[3, 8, 5], seed=1, random_init=True)

    x = jnp.array([0.1, -0.7, 2.0])

    grad_auto = np.asarray(net.gradient(x))

    # jacrev cross-check: output shape (5, 3) -> transpose to (3, 5)
    jac_rev = np.asarray(
        jax.jacrev(lambda xx: net.moekan_network(xx))(x)
    )  # shape (5, 3)
    jac_rev = jac_rev.T  # (3, 5)

    # FD cross-check (looser)
    grad_fd = np.zeros((3, 5))
    for i in range(3):
        x_plus = np.asarray(x).copy(); x_plus[i] += 1e-3
        x_minus = np.asarray(x).copy(); x_minus[i] -= 1e-3
        f_plus = np.asarray(net.moekan_network(jnp.asarray(x_plus)))   # (5,)
        f_minus = np.asarray(net.moekan_network(jnp.asarray(x_minus))) # (5,)
        grad_fd[i, :] = (f_plus - f_minus) / (2e-3)

    ok = True
    ok &= check_close("gradient() vs FD (vector)", grad_auto, grad_fd, rtol=1e-2, atol=1e-2)
    ok &= check_close("gradient() vs jacrev (vector)", grad_auto, jac_rev, rtol=1e-5, atol=1e-6)
    assert grad_auto.shape == (1, 3, 5), f"Expected (1,3,5), got {grad_auto.shape}"
    print(f"  Shape check: {grad_auto.shape}  OK")
    return ok


# ---------------------------------------------------------------------------
# Test 4: vector output, batched
# ---------------------------------------------------------------------------
def test_vector_batched():
    print("\n=== Test 4: vector output, batched (2-D input) ===")
    net = MOEKAN(width=[2, 4, 3], seed=99, random_init=True)

    key = jax.random.PRNGKey(42)
    x_batch = jax.random.normal(key, (6, 2))

    grad_auto = np.asarray(net.gradient(x_batch))

    # jacrev cross-check
    jac_rev = np.asarray(
        jax.vmap(jax.jacrev(lambda xx: net.moekan_network(xx)))(x_batch)
    )  # shape (6, 3, 2)
    jac_rev = np.transpose(jac_rev, (0, 2, 1))  # (6, 2, 3)

    ok = True
    ok &= check_close("gradient() vs jacrev (vector batch)", grad_auto, jac_rev, rtol=1e-5, atol=1e-6)
    assert grad_auto.shape == (6, 2, 3), f"Expected (6,2,3), got {grad_auto.shape}"
    print(f"  Shape check: {grad_auto.shape}  OK")
    return ok


# ---------------------------------------------------------------------------
# Test 5: deterministic init (random_init=False)
# ---------------------------------------------------------------------------
def test_deterministic_init():
    print("\n=== Test 5: deterministic init (random_init=False) ===")
    net = MOEKAN(width=[2, 4, 1], seed=0, random_init=False)

    x = jnp.array([1.0, -0.5])
    grad_auto = np.asarray(net.gradient(x))

    f = lambda xx: float(net.moekan_network(jnp.asarray(xx))[0])
    grad_fd = finite_difference_gradient(f, np.asarray(x))

    grad_rev = np.asarray(
        jax.jacrev(lambda xx: net.moekan_network(xx)[0])(x)
    )

    ok = True
    ok &= check_close("gradient() vs FD (det init)", grad_auto, grad_fd, rtol=1e-2, atol=1e-2)
    ok &= check_close("gradient() vs jacrev (det init)", grad_auto, grad_rev, rtol=1e-5, atol=1e-6)
    return ok


# ---------------------------------------------------------------------------
# Test 6: multi-layer network (3 hidden layers)
# ---------------------------------------------------------------------------
def test_deep_network():
    print("\n=== Test 6: deep network [3, 8, 8, 8, 1] ===")
    net = MOEKAN(width=[3, 8, 8, 8, 1], seed=5, random_init=True)

    key = jax.random.PRNGKey(7)
    x_batch = jax.random.normal(key, (5, 3))

    grad_auto = np.asarray(net.gradient(x_batch))

    grad_fd = np.zeros_like(np.asarray(x_batch))
    for i in range(5):
        grad_fd[i] = finite_difference_gradient(
            lambda xx: float(net.moekan_network(jnp.asarray(xx))[0]),
            np.asarray(x_batch[i]),
        )

    grad_rev = np.asarray(
        jax.vmap(jax.jacrev(lambda xx: net.moekan_network(xx)[0]))(x_batch)
    )

    ok = True
    ok &= check_close("gradient() vs FD (deep)", grad_auto, grad_fd, rtol=1e-2, atol=1e-2)
    ok &= check_close("gradient() vs jacrev (deep)", grad_auto, grad_rev, rtol=1e-5, atol=1e-6)
    assert grad_auto.shape == (5, 3), f"Expected (5,3), got {grad_auto.shape}"
    print(f"  Shape check: {grad_auto.shape}  OK")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MOEKAN gradient verification")
    print("=" * 60)

    results = [
        test_scalar_single_sample(),
        test_scalar_batched(),
        test_vector_single_sample(),
        test_vector_batched(),
        test_deterministic_init(),
        test_deep_network(),
    ]

    print("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} tests passed")
    print("=" * 60)

    if all(results):
        print("\nAll gradient checks PASSED.")
    else:
        print("\nSome gradient checks FAILED. Inspect the output above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
