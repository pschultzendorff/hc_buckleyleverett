"""Homotopy continuation mixins for the Buckley-Leverett model class.

Three auxiliary problems are considered:
- Linear relative permeabilities: The initial problem has linear relative
permeabilities, i.e. the phase mobilities are linear functions of saturation.

- Vanishing diffusion: The initial problem has an additional diffusion term in the
flux, which is gradually reduced to zero during the homotopy continuation.

- Convex/concave hull of the flux function: The initial problem has a modified flux
  function, which is the convex/concave hull of the original flux function.

"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt

from hc_buckleyleverett.buckley_leverett.model import BuckleyLeverettModel
from hc_buckleyleverett.buckley_leverett.protocol import (
    BuckleyLeverettModelProtocol,
    HCProtocol,
)
from hc_buckleyleverett.utils.con_hull import HullSide, con_hull

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sns.set_style("whitegrid")


class HCMixin(HCProtocol, BuckleyLeverettModelProtocol):
    """Base mixin for homotopy continuation models."""

    def __init__(self, params: dict[str, float]):
        super().__init__(params=params)  # type: ignore

        # Lists to store curve data.
        self.betas: list[float] = []
        self.intermediate_solutions: list[jnp.ndarray] = []

    def reset(self) -> None:
        """Reset the homotopy continuation model to its initial state and empty the
        statistics.

        """
        # Ignore pylance error; concrete method provided by BuckleyLeverettModel via MRO.
        super().reset()  # type: ignore
        self.betas = []
        self.intermediate_solutions = []

    def store_curve_data(
        self,
        beta: float,
        q: jnp.ndarray,
        dt: float,
        q_prev: Optional[jnp.ndarray] = None,
    ):
        r"""Store :math:`\lambda` values and intermediate solutions along the HC curve."""
        self.betas.append(beta)
        self.intermediate_solutions.append(q)


class LinearRelPermHCMixin(HCMixin):
    """Linear relative permeability homotopy continuation for the two-phase flow problem.

    Auxiliary problem is Buckley-Leverett with linear relative permeabilities.

    von Schultzendorff, P.; Both, J.W.; Nordbotten, J.M.; Sandve, T.H.; Vohralík, M.,
    Adaptive homotopy continuation solver for incompressible two-phase flow in porous
    media, (in preparation)


    """

    def mobility_w(self, s, **kwargs):
        beta = kwargs["beta"]
        # Ingore pylance error; self is a valid BuckleyLeverettModel at runtime via MRO.
        return beta * BuckleyLeverettModel.mobility_w(
            self,  # type: ignore[arg-type]
            s,
            rp_model="linear",
        ) + (1 - beta) * BuckleyLeverettModel.mobility_w(self, s)  # type: ignore[arg-type]

    def mobility_n(self, s, **kwargs):
        beta = kwargs["beta"]
        # Ingore pylance error; self is a valid BuckleyLeverettModel at runtime via MRO.
        return beta * BuckleyLeverettModel.mobility_n(
            self,  # type: ignore[arg-type]
            s,
            rp_model="linear",
        ) + (1 - beta) * BuckleyLeverettModel.mobility_n(self, s)  # type: ignore[arg-type]


class DiffusionHCMixin(HCMixin):
    r"""Vanishing diffusion homotopy continuation for the two-phase flow problem.

    Cf. Jiang, J. and Tchelepi, H.A. (2018) ‘Dissipation-based continuation method for
    multiphase flow in heterogeneous porous media’, Journal of Computational Physics,
    375, pp. 307–336. Available at: https://doi.org/10.1016/j.jcp.2018.08.044.

    The parameter controlling the strength of the diffusion, called :math:`\beta` in
    the paper, is called `diffusion_coeffficient` here (the continuation parameter
    :math:`\lambda` is called :math:`\kappa` in the paper).

    """

    def __init__(self, params: dict[str, float], **kwargs):
        super().__init__(params)
        self.adaptive_diffusion_coeff = kwargs.get("adaptive_diffusion_coeff", 1.0)
        self.omega = kwargs.get("omega", 2.0e-3)

    def compute_face_fluxes(self, s, **kwargs):
        """Compute total and wetting phase fluxes at cell faces with vanishing
        diffusion.

        At domain boundaries the diffusion is nulled.

        """
        beta = kwargs["beta"]

        # Target flux:
        # Ignore pylance error; concrete method provided by BuckleyLeverettModel via MRO.
        F_w = super().compute_face_fluxes(s)  # type: ignore[misc]

        # Diffusive flux:
        s = jnp.concatenate(
            [jnp.array([self.s_inlet]), s, jnp.array([s[-1]])]
        )  # zero-gradient ghost on the right

        # Diffusive flux = -1 * saturation gradient * diffusion coefficient
        s_gradient = s[1:] - s[:-1]
        diffusive_flux = -self.adaptive_diffusion_coeff * s_gradient

        # No diffusion at the boundaries.
        diffusive_flux = diffusive_flux.at[0].set(0.0)
        diffusive_flux = diffusive_flux.at[-1].set(0.0)

        # HC flux:
        F_w += beta * diffusive_flux

        return F_w

    def update_adaptive_diffusion_coeff(self, dt: float) -> None:
        r"""Update the adaptive diffusion coefficient for the current time step.

        We follow the approach from Jiang and Tchelepi (2018), which defines the
        vanishing artificial diffusion flux for the fully coupled flow and transport
        problem as

        .. math::
            \beta = \omega \frac{u_t \Delta t}{\Delta x} \max |(\frac{\lambda_{w}}{\lambda_{t}})'|,

        where :math:`\omega` is set to :math:`2.0 \times 10^{-3}` in the 1D scalar
        transport problem.

        Parameters:


            dt: Time step size.
        """
        max_flow_gradient = self.f_max_grad()
        dx = self.domain_size / self.num_cells

        self.adaptive_diffusion_coeff = (
            self.omega
            * (self.total_flow * dt)
            / (self.porosity * dx)
            * max_flow_gradient
        )

        if jnp.isnan(self.adaptive_diffusion_coeff).any():
            raise ValueError(
                "Adaptive diffusion coefficient contains NaN values. "
                "Check the input data and the model parameters."
            )

    def f_max_grad(self, n_points=100) -> jax.Array:
        """Calculate the maximum gradient of the flow function.

        Parameters:
            n_points: _description_. Defaults to 100.

        Returns:
            _description_

        """
        # Sample saturations. Disregarding S=0 and S=1, as mobilities are cut off
        # there and gradients may return NaN.
        s = jnp.linspace(0.0, 1.0, n_points)[1:-1]

        # Compute the gradient of the fractional flow for the sampled saturations.
        batched_grad = jax.vmap(jax.grad(self.fractional_flow))
        f_prime = batched_grad(s)  # ``shape=(n_points - 2)``
        assert f_prime.shape == (s.shape[0],), (
            f"Expected shape {(s.shape[0],)}, got {f_prime.shape}"
        )

        return jnp.abs(f_prime).max(axis=-1)


class ConHullHCMixin(HCMixin):
    r"""Convex/concave hull homotopy continuation for the two-phase flow problem.

    Take the convex/concave hull of the wetting flow function
    :math:`f_w = \frac{\lambda_w}{\lambda_t}` as the initial wetting flow function.

    Note: This assumes **ZERO** buoyancy and capillary forces, s.t. the numerical
    wetting flow function :math:`\frac{F_w}{F_t}` is a one-dimensional function of only
    the saturation in upstream direction.

    """

    def __init__(self, params: dict[str, float], **kwargs):
        super().__init__(params=params)
        self.hull_side: HullSide = kwargs["con_hull_side"]
        self.hull_saturation_interval: tuple[float, float] = kwargs.get(
            "hull_saturation_interval", (0.0, 1.0)
        )
        self.initialize_con_hull()

    def initialize_con_hull(self):
        """Compute the convex/concave hull of the wetting flow function.

        When gravity is nonzero, also pre-computes the convex-hull wetting mobility
        as an interpolation table so that the quadratic inversion only runs once.

        """

        def fractional_flow(s):
            return self.fractional_flow(s)

        self.fractional_flow_con_hull = con_hull(  # type: ignore  # returns interpolated callable
            fractional_flow,
            self.hull_saturation_interval,
            self.hull_side,
            xp=jnp,
        )

        if self.G != 0:
            self._initialize_gravity_con_hull_mobilities()

    def _initialize_gravity_con_hull_mobilities(self, num_points: int = 200) -> None:
        r"""Pre-compute the convex-hull wetting mobility for the gravity case.

        Samples saturations on a fine grid, solves the quadratic equation relating the
        convex-hull fractional flow :math:`f_{w,\mathrm{conv}}` to the wetting mobility
        :math:`\lambda_{w,\mathrm{conv}}`:

        .. math::
            G_r \, x^2 + (1 - \lambda_t G_r) \, x
            - f_{w,\mathrm{conv}} \lambda_t = 0

        where :math:`G_r = C_g / (u_t \mu_n)`, and stores the result as an
        interpolation table.  At runtime, :meth:`mobility_w_con_hull_gravity`
        evaluates via :func:`jnp.interp`.

        Parameters:
            num_points: Number of saturation samples for the interpolation table.

        """
        s_samples = jnp.linspace(
            float(self.hull_saturation_interval[0]),
            float(self.hull_saturation_interval[1]),
            num_points,
        )

        # Physical total mobility (base model, independent of beta).
        m_w_phys = BuckleyLeverettModel.mobility_w(self, s_samples)  # type: ignore[arg-type]
        m_n_phys = BuckleyLeverettModel.mobility_n(self, s_samples)  # type: ignore[arg-type]
        m_t = m_w_phys + m_n_phys

        # Convex-hull fractional flow at each sample.
        f_w_conv = self.fractional_flow_con_hull(s_samples)

        # Solve the quadratic for lambda_w,conv.
        Gr = self.gravity_number / (self.total_flow * self.mu_n)
        A = Gr
        B = 1.0 - m_t * Gr
        C = -f_w_conv * m_t

        discriminant = jnp.maximum(B**2 - 4 * A * C, 0.0)
        m_w_conv = jnp.where(
            jnp.abs(A) > 1e-30,
            (-B + jnp.sqrt(discriminant)) / (2.0 * A),
            f_w_conv * m_t,
        )
        m_w_conv = jnp.clip(m_w_conv, 0.0, m_t)  # type: ignore

        # Store the interpolation table.
        self._gravity_conv_hull_s = s_samples
        self._gravity_conv_hull_m_w = m_w_conv

    def mobility_w_con_hull_gravity(self, s: jnp.ndarray) -> jnp.ndarray:
        """Interpolate the pre-computed convex-hull wetting mobility (gravity case).

        Parameters:
            s: Saturation values at which to evaluate.

        Returns:
            Wetting-phase mobility consistent with the convex-hull fractional flow and
            buoyancy.

        """
        return jnp.interp(s, self._gravity_conv_hull_s, self._gravity_conv_hull_m_w)

    def compute_face_fluxes(self, s, **kwargs):
        """Compute total and wetting phase fluxes at cell faces with convex/concave hull
        homotopy continuation.

        The phase fluxes use the convex/concave hull of the fractional flow with
        phase-potential upwinding (when gravity is present), while the total flux is the
        physically correct flux from the base model. The two are combined as a convex
        combination weighted by ``beta``.

        """
        beta = kwargs["beta"]

        # Prepend the inlet boundary saturation.
        s = jnp.concatenate([jnp.array([self.s_inlet]), s])

        if self.G == 0:
            # Total-flux upwinding: mobilities evaluated at the left cell.
            m_w_target = self.mobility_w(s, **kwargs)
            m_n_target = self.mobility_n(s, **kwargs)
            m_t = m_w_target + m_n_target

            m_w_conv = self.fractional_flow_con_hull(s) * m_t
            m_n_conv = m_t - m_w_conv

            # Blend with beta.
            m_w_hc = beta * m_w_conv + (1 - beta) * m_w_target
            m_n_hc = beta * m_n_conv + (1 - beta) * m_n_target

        else:
            raise NotImplementedError(
                "Convex/concave hull homotopy continuation with gravity is not implemented yet."
            )

        # Viscous wetting flux.
        F_w_viscous = (self.total_flow / self.porosity) * (m_w_hc / m_t)

        # Buoyancy wetting flux.
        dx = self.domain_size / self.num_cells
        transmissibilities = self.face_transmissibility()

        delta_rho = self.rho_w - self.rho_n
        buoyancy_potential = self.G * jnp.sin(self.theta) * dx * delta_rho

        F_w_buoyancy = (m_w_hc * m_n_hc / m_t) * (
            transmissibilities * buoyancy_potential
        )

        return F_w_viscous - F_w_buoyancy

    def plot_con_hull(
        self, color_f: str, color_hull: str, color_linear: str, **kwargs
    ) -> plt.Figure:
        """Plot the convex/concave hull of the wetting flow function."""
        label_fontsize = kwargs.get("label_fontsize", 14)
        tick_fontsize = kwargs.get("tick_fontsize", 10)

        s_vals = jnp.linspace(0, 1, 100)
        fw_vals = self.fractional_flow(s_vals)
        fw_hull_vals = self.fractional_flow_con_hull(s_vals)
        fw_linear_vals = self.fractional_flow(s_vals, rp_model="linear")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(s_vals, fw_vals, label="$f(S)$", color=color_f, ls="-")
        ax.plot(
            s_vals,
            fw_hull_vals,
            label=r"$f_\mathrm{c}(S)$",
            color=color_hull,
            ls="--",
        )
        ax.plot(
            s_vals,
            fw_linear_vals,
            label=r"$f_\mathrm{lin}(S)$",
            color=color_linear,
            ls="-.",
        )
        ax.set_xlabel("$S$", fontsize=label_fontsize)
        ax.legend(fontsize=label_fontsize)

        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

        return fig
