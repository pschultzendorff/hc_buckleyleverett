"""Tests for :mod:`hc_sandbox.buckley_leverett.hc.DiffusionHCMixin`."""

import math

import jax.numpy as jnp
import numpy as np
from hc_sandbox.buckley_leverett.hc import ConHullHCMixin, DiffusionHCMixin
from hc_sandbox.buckley_leverett.model import BuckleyLeverettModel

# ---------------------------------------------------------------------------
# Concrete model with DiffusionHCMixin for testing
# ---------------------------------------------------------------------------


class DiffusionHCModel(DiffusionHCMixin, BuckleyLeverettModel): ...


def _make_model(**overrides) -> DiffusionHCModel:
    """Create a minimal DiffusionHCModel with sensible defaults."""
    params = {
        "num_cells": 10,
        "domain_size": 1.0,
        "total_flow": 1.0,
        "rp_model": "Corey",
        "nw": 2.0,
        "nn": 2.0,
        "permeability": 1.0,
        "porosity": 0.2,
        "rho_w": 1.0,
        "rho_n": 1.0,
        "G": 0.0,
        "theta": 0.0,
        "s_inlet": 1.0,
        "s_initial": jnp.full(10, 0.5),
        "mu_w": 1.0,
        "mu_n": 1.0,
    }
    params.update(overrides)
    # Ignore the type checker. At runtime the model has all required methods.
    return DiffusionHCModel(params)  # type: ignore


def test_f_max_grad_linear_relperm_is_constant():
    """For linear relative permeabilities f_w(s) = s, so f'(s) = 1 everywhere."""
    model = _make_model(rp_model="linear")
    result = model.f_max_grad(n_points=200)
    np.testing.assert_allclose(float(result), 1.0, atol=1e-4)


def test_f_max_grad_is_positive():
    """The max absolute gradient of f_w must be strictly positive for any valid
    rel-perm model."""
    model = _make_model(rp_model="Corey", nw=3.0, nn=2.0)
    result = model.f_max_grad()
    assert float(result) > 0


def test_f_max_grad_increases_with_nw():
    """Higher Corey wetting exponent produces a steeper fractional flow curve
    and thus a larger maximum gradient."""
    grad_low = float(_make_model(nw=2.0, nn=2.0).f_max_grad())
    grad_high = float(_make_model(nw=4.0, nn=2.0).f_max_grad())
    assert grad_high > grad_low


def test_update_adaptive_diffusion_coeff_formula():
    """Verify the coefficient equals omega * (u_t * dt) / phi * max|f'|."""
    model = _make_model()
    dt = 0.5
    dx = model.domain_size / model.num_cells

    model.update_adaptive_diffusion_coeff(dt)

    omega = 2.0e-3
    expected = (
        omega * (model.total_flow * dt) / (model.porosity * dx) * model.f_max_grad()
    )
    np.testing.assert_allclose(
        float(model.adaptive_diffusion_coeff), float(expected), rtol=1e-7
    )


def test_update_adaptive_diffusion_coeff_scales_with_dt():
    """The coefficient should scale linearly with the time-step size."""
    model = _make_model()
    model.update_adaptive_diffusion_coeff(dt=1.0)
    coeff_1 = float(model.adaptive_diffusion_coeff)

    model.update_adaptive_diffusion_coeff(dt=2.0)
    coeff_2 = float(model.adaptive_diffusion_coeff)

    np.testing.assert_allclose(coeff_2, 2.0 * coeff_1, rtol=1e-7)


class ConvexHullHCModel(ConHullHCMixin, BuckleyLeverettModel): ...


def _make_convex_hull_gravity_model(**overrides) -> ConvexHullHCModel:
    """Create a ConvexHullHCModel with nonzero gravity for testing."""
    params = {
        "num_cells": 10,
        "domain_size": 1.0,
        "total_flow": 1.0,
        "rp_model": "Corey",
        "nw": 3.0,
        "nn": 3.0,
        "permeability": 1.0,
        "porosity": 0.2,
        "rho_w": 1000.0,
        "rho_n": 900.0,
        "G": 9.81,
        "theta": math.pi / 4,
        "s_inlet": 0.0,
        "s_initial": jnp.full(10, 0.5),
        "mu_w": 10.0,
        "mu_n": 1.0,
    }
    params.update(overrides)
    return ConvexHullHCModel(params, con_hull_side="lower")  # type: ignore


def test_convex_hull_gravity_mobility_reproduces_fractional_flow():
    r"""Verify that the back-calculated gravity mobility satisfies the fractional flow
    identity.

    For the convex-hull wetting mobility :math:`\lambda_{w,\mathrm{conv}}` and
    :math:`\lambda_{n,\mathrm{conv}} = \lambda_t - \lambda_{w,\mathrm{conv}}`:

    .. math::
        f_{w,\mathrm{conv}}(s) = \frac{\lambda_{w,\mathrm{conv}}}{\lambda_t}
            - \frac{\lambda_{w,\mathrm{conv}} \lambda_{n,\mathrm{conv}}}
                   {\lambda_t \mu_n} \frac{C_g}{u_t}

    """
    model = _make_convex_hull_gravity_model()

    # Evaluate on a fine grid, avoiding exact boundaries where mobilities vanish.
    s = jnp.linspace(0.01, 0.99, 100)

    m_w_conv = model.mobility_w_con_hull_gravity(s)
    m_w_phys = BuckleyLeverettModel.mobility_w(model, s)  # type: ignore[arg-type]
    m_n_phys = BuckleyLeverettModel.mobility_n(model, s)  # type: ignore[arg-type]
    m_t = m_w_phys + m_n_phys
    m_n_conv = m_t - m_w_conv

    # Reconstruct fractional flow from the back-calculated mobilities.
    f_w_reconstructed = (m_w_conv / m_t) - (
        (m_w_conv * m_n_conv)
        / (m_t * model.mu_n)
        * model.gravity_number
        / model.total_flow
    )

    # Expected: the convex hull fractional flow.
    f_w_expected = model.fractional_flow_con_hull(s)

    np.testing.assert_allclose(
        np.array(f_w_reconstructed),
        np.array(f_w_expected),
        atol=5e-3,
        err_msg="Reconstructed f_w from gravity mobility does not match convex hull f_w",
    )
