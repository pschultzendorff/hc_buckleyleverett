r"""Tests for the tangent, curvature_vector, and curvature methods of HCAnalysis.

Uses a mock model whose homotopy curve is the helix

.. math::
    c(\lambda) = (\cos \lambda, \sin \lambda, \lambda), \quad \lambda \in (0, \pi).

The residual function
:math:`\mathcal{H}(\mathbf{q}, \lambda) = (q_1^2 + q_2^2 - 1,\; q_1 - \cos \lambda) = 0`
implicitly defines this helix. The Jacobian :math:`\nabla_{\mathbf{q}} \mathcal{H}`, derivative
:math:`\partial_\lambda \mathcal{H}`, and full :math:`(N+1) \times (N+1)` Hessian tensor are
provided analytically so the test is independent of the automatic-differentiation
machinery.

Analytical properties of the helix:

- :math:`\|c'(\lambda)\| = \sqrt{2}` (constant speed)
- arclength unit tangent:
  :math:`(-\sin\lambda / \sqrt{2},\; \cos\lambda / \sqrt{2},\; 1 / \sqrt{2})`
- :math:`\beta` tangent (:math:`\mathbf{q}`-part):
  :math:`(-\sin\lambda,\; \cos\lambda)`
- arclength curvature vector:
  :math:`(-\cos\lambda / 2,\; -\sin\lambda / 2,\; 0)`
- :math:`\beta` curvature vector (:math:`\mathbf{q}`-part):
  :math:`(-\cos\lambda / 2,\; -\sin\lambda / 2)`
- Curvature :math:`\kappa = 1/2` (helix with radius 1 and pitch :math:`2\pi \cdot 1`)

"""

import jax.numpy as jnp
import numpy as np
import pytest
from hc_sandbox.buckley_leverett.hc_analysis import HCAnalysisMixin


class MockHelixModel(HCAnalysisMixin):
    r"""Mock model whose homotopy curve is the helix
    :math:`(\cos\lambda, \sin\lambda, \lambda)`.

    Residual:
    :math:`\mathcal{H}(\mathbf{q}, \lambda) = (q_1^2 + q_2^2 - 1,\; q_1 - \cos\lambda)`.

    Building blocks (Jacobian, ``h_beta_deriv``, ``hc_hessian_tensor_fn``) are provided
    analytically so the test does not depend on JAX tracing / AD.
    """

    def __init__(self):
        # Intentionally skipping HCAnalysis.__init__ (which calls hc_hessian_tensor
        # and requires a full BuckleyLeverettModel).
        pass

    def jacobian(self, q, dt, q_prev=None, **kwargs):
        r""":math:`\nabla_{\mathbf{q}} \mathcal{H} = [[2\mathbf{q}_1, 2\mathbf{q}_2], [1, 0]]`."""
        return jnp.array([[2.0 * q[0], 2.0 * q[1]], [1.0, 0.0]])

    def h_beta_deriv(self, q, dt, q_prev=None, **kwargs):
        r""":math:`\partial \mathcal{H} / \partial \lambda = (0, \sin\lambda)`."""
        return jnp.array([0.0, jnp.sin(kwargs["beta"])])

    def hc_hessian_tensor_fn(self, q, dt, q_prev=None, **kwargs):
        r"""Full :math:`(N+1) \times (N+1)` Hessian tensor per residual component.

        Component 0 (:math:`\mathcal{H}_0 = \mathbf{q}_1^2 + \mathbf{q}_2^2 - 1`):
            :math:`\partial^2 \mathcal{H}_0 / \partial \mathbf{q}^2 = \operatorname{diag}(2, 2)`;
            mixed and :math:`\lambda\lambda` blocks are zero.

        Component 1 (:math:`\mathcal{H}_1 = \mathbf{q}_1 - \cos\lambda`):
            :math:`\partial^2 \mathcal{H}_1 / \partial \mathbf{q}^2 = 0`;
            :math:`\partial^2 \mathcal{H}_1 / \partial \mathbf{q} \partial \lambda = 0`;
            :math:`\partial^2 \mathcal{H}_1 / \partial \lambda^2 = \cos\lambda`.

        """
        H = jnp.zeros((2, 3, 3))
        H = H.at[0, 0, 0].set(2.0)
        H = H.at[0, 1, 1].set(2.0)
        H = H.at[1, 2, 2].set(jnp.cos(kwargs["beta"]))
        return H


# Test at several interior points of :math:`(0, \pi)` where :math:`\sin(\lambda) \neq 0`
# so the Jacobian is non-singular.
TEST_BETA_VALUES = [
    np.pi / 6,
    np.pi / 4,
    np.pi / 3,
    np.pi / 2,
    2 * np.pi / 3,
    3 * np.pi / 4,
    5 * np.pi / 6,
]


@pytest.fixture(
    params=TEST_BETA_VALUES,
    ids=[f"beta={beta:.4f}" for beta in TEST_BETA_VALUES],
    scope="module",
)
def helix_point(request):
    """Return model and state for a point on the helix."""
    beta = request.param
    model = MockHelixModel()  # type: ignore
    q = jnp.array([jnp.cos(beta), jnp.sin(beta)])
    dt = 1.0  # dt is unused by the mock but required by the API
    return model, q, dt, beta


# Tolerance for floating-point comparison.
ATOL = 1e-6


# Tangent tests:


def test_arclength_tangent_values(helix_point):
    r"""arclength tangent matches analytical
    :math:`(-\sin\lambda / \sqrt{2},\; \cos\lambda / \sqrt{2},\; 1/\sqrt{2})`.
    """
    model, q, dt, beta = helix_point
    t = model.tangent(beta, q, dt, parametrization="arclength")

    expected = jnp.array(
        [
            -jnp.sin(beta) / jnp.sqrt(2.0),
            jnp.cos(beta) / jnp.sqrt(2.0),
            1.0 / jnp.sqrt(2.0),
        ]
    )
    np.testing.assert_allclose(t, expected, atol=ATOL)


def test_arclength_tangent_is_unit_vector(helix_point):
    """Test that the arclength tangent has unit norm."""
    model, q, dt, beta = helix_point
    t = model.tangent(beta, q, dt, parametrization="arclength")
    np.testing.assert_allclose(jnp.linalg.norm(t), 1.0, atol=ATOL)


def test_beta_tangent_values(helix_point):
    r"""Test that the :math:`\beta` tangent matches analytical
    :math:`(-\sin\lambda,\; \cos\lambda)`.
    """
    model, q, dt, beta = helix_point
    t = model.tangent(beta, q, dt, parametrization="lambda")

    expected = jnp.array([-jnp.sin(beta), jnp.cos(beta)])
    np.testing.assert_allclose(t, expected, atol=ATOL)


def test_beta_tangent_norm(helix_point):
    r"""Test that the :math:`\beta` tangent norm equals :math:`1` (radius of the unit
    circle).
    """
    model, q, dt, beta = helix_point
    t = model.tangent(beta, q, dt, parametrization="lambda")
    np.testing.assert_allclose(jnp.linalg.norm(t), 1.0, atol=ATOL)


def test_tangent_invalid_parametrization_raises(helix_point):
    """Test that an invalid parametrization raises :class:`NotImplementedError`."""
    model, q, dt, beta = helix_point
    with pytest.raises(NotImplementedError):
        model.tangent(beta, q, dt, parametrization="invalid")


# Curvature-vector tests:


def test_arclength_curvature_vector_values(helix_point):
    r"""arclength curvature vector matches analytical
    :math:`(-\cos\lambda / 2,\; -\sin\lambda / 2,\; 0)`.
    """
    model, q, dt, beta = helix_point
    kv = model.curvature_vector(beta, q, dt, parametrization="arclength")

    expected = jnp.array([-jnp.cos(beta), -jnp.sin(beta), 0.0]) / 2.0
    np.testing.assert_allclose(kv, expected, atol=ATOL)


def test_beta_curvature_vector_values(helix_point):
    r"""Test that the partial :math:`\beta` curvature vector matches analytical
    :math:`(-\cos\lambda,\; -\sin\lambda)`.
    """
    model, q, dt, beta = helix_point
    kv = model.curvature_vector(beta, q, dt, parametrization="lambda")

    expected = jnp.array([-jnp.cos(beta), -jnp.sin(beta)])
    np.testing.assert_allclose(kv, expected, atol=ATOL)


def test_arclength_curvature_vector_orthogonal_to_tangent(helix_point):
    r"""Test that the arclength curvature vector :math:`\ddot{c}(s)` is orthogonal to
    the tangent :math:`\dot{c}(s)`.
    """
    model, q, dt, beta = helix_point
    t = model.tangent(beta, q, dt, parametrization="arclength")
    kv = model.curvature_vector(beta, q, dt, parametrization="arclength")
    np.testing.assert_allclose(jnp.dot(t, kv), 0.0, atol=ATOL)


def test_curvature_vector_invalid_parametrization_raises(helix_point):
    """Test that an invalid parametrization raises :class:`NotImplementedError`."""
    model, q, dt, beta = helix_point
    with pytest.raises(NotImplementedError):
        model.curvature_vector(beta, q, dt, parametrization="invalid")


# Curvature (scalar) tests:


def test_arclength_curvature_value(helix_point):
    r"""Test that the arclength curvature is :math:`\kappa = 1/2` for the unit helix."""
    model, q, dt, beta = helix_point
    kappa = model.curvature(beta, q, dt, parametrization="arclength")
    np.testing.assert_allclose(float(kappa), 0.5, atol=ATOL)


def test_beta_curvature_value(helix_point):
    r"""Test that the partial :math:`\lambda` curvature is :math:`\kappa_\mathbf{q} = 1`
    for the unit helix.
    """
    model, q, dt, beta = helix_point
    kappa = model.curvature(beta, q, dt, parametrization="lambda")
    np.testing.assert_allclose(float(kappa), 1.0, atol=ATOL)


# Curvature and tangent test with precomputed Jacobian / tangent:


def test_tangent_with_precomputed_jac(helix_point):
    """Passing a precomputed Jacobian gives the same tangent."""
    model, q, dt, beta = helix_point
    jac = model.jacobian(q, dt, beta=beta)
    t1 = model.tangent(beta, q, dt, parametrization="arclength")
    t2 = model.tangent(beta, q, dt, jac=jac, parametrization="arclength")
    np.testing.assert_allclose(t1, t2, atol=ATOL)


def test_curvature_vector_with_precomputed_jac_and_tangent(helix_point):
    """Passing precomputed jac and tangent gives the same curvature vector."""
    model, q, dt, beta = helix_point
    jac = model.jacobian(q, dt, beta=beta)
    t = model.tangent(beta, q, dt, jac=jac, parametrization="arclength")
    kv1 = model.curvature_vector(beta, q, dt, parametrization="arclength")
    kv2 = model.curvature_vector(
        beta, q, dt, jac=jac, tangent=t, parametrization="arclength"
    )
    np.testing.assert_allclose(kv1, kv2, atol=ATOL)
