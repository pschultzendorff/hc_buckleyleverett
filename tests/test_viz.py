"""Tests for :mod:`hc_sandbox.buckley_leverett.viz`."""

import jax.numpy as jnp
import numpy as np
from hc_sandbox.buckley_leverett.viz import segments_arclengths


def _expected_segment_lengths(betas, solutions):
    """Helper function for :meth:`test_segments_arclengths_...`: compute ||[q_i -
    q_{i-1}, beta_i - beta_{i-1}]||.

    """
    n = len(betas)
    lengths = np.empty(n - 1)
    for i in range(1, n):
        dq = np.asarray(solutions[i]) - np.asarray(solutions[i - 1])
        db = np.asarray(betas[i]) - np.asarray(betas[i - 1])
        lengths[i - 1] = np.sqrt(np.sum(dq**2) + db**2)
    return lengths


def test_segments_arclengths_identical_points_gives_zero():
    """All segment lengths are zero when every point is the same."""
    betas = jnp.array([0.5, 0.5, 0.5])
    solutions = jnp.ones((3, 4))
    result = segments_arclengths(betas, solutions)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_segments_arclengths_known_distances():
    """Check against hand-computed distances (3-4-5 triangle and others)."""
    betas = jnp.array([0.5, 0.5, 0.5])
    solutions = jnp.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],  # distance from previous = 5
            [6.0, 4.0],  # distance from previous = 3
        ]
    )
    result = segments_arclengths(betas, solutions)
    np.testing.assert_allclose(float(result[0]), 5.0, atol=1e-7)
    np.testing.assert_allclose(float(result[1]), 3.0, atol=1e-7)


def test_segments_arclengths_random_vs_reference():
    """Arbitrary points: compare with a brute-force reference."""
    rng = np.random.default_rng(42)
    n_points, n_vars = 20, 5
    betas_np = np.sort(rng.random(n_points))[::-1]
    solutions_np = rng.standard_normal((n_points, n_vars))

    result = segments_arclengths(jnp.array(betas_np), jnp.array(solutions_np))
    expected = _expected_segment_lengths(betas_np, solutions_np)
    np.testing.assert_allclose(result, expected, atol=1e-6)
