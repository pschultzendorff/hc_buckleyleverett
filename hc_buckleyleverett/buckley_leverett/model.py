r"""Implicit finite volume model for the hyperbolic conservation law, in particular for
the Buckley-Leverett problem.

An example geometry with the domain discretized into two cells looks like this:

          |-------------------|-------------------|
          |                   |                   |
Neu bc    F0                  F1                  F2      Dir bc
       ------>    s0       ------>      s1    ------>       s2
          |                   |                   |
          |-------------------|-------------------|
          e0        c0        e1        c1        e2

The Buckley-Leverett problem is derived from the two-phase flow equations in porous
media assuming, immiscible incompressible fluids, zero capillary pressure, and
one-dimensional flow. The governing equations for the combined viscous-buoyancy case
are:

.. math::

    \phi \frac{\partial S_w}{\partial t} + \frac{\partial F_w}{\partial x} = 0

where :math:`S_w` is the wetting phase saturation, :math:`\phi` is porosity, and
:math:`F_w` is the wetting phase flux given by:

.. math::

    F_w = u_t f_w(S_w) - K \lambda_w(S_w) \Delta \rho g \sin(theta)

with the fractional flow function:

.. math::

    f_w(S_w) = \frac{\lambda_w(S_w)}{\lambda_w(S_w) + \lambda_n(S_w)}

Here, :math:`u_t` is the total Darcy velocity, :math:`K` is permeability,
:math:`\lambda_i = k_{ri}/\mu_i` are the phase mobilities,
:math:`\Delta \rho = \rho_w - \rho_n`
is the density difference, :math:`g` is gravitational acceleration, and :math:`\theta`
is the angle of inclination.

To solve the problem numerically, we
- discretize with implicit Euler in time
- discretize with cell-centered finite volumes in space, where face fluxes are evaluated
with two-point flux approximation and phase-potential upwinding
- solve the resulting nonlinear system with a Newton-Raphson method, where the Jacobian
is computed with automatic differentiation.


"""

import logging
from typing import Any, Optional

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class BuckleyLeverettModel:
    def __init__(self, params: dict[str, Any]) -> None:
        r"""_summary_

        Note:
            If neither ``total_flow`` nor ``porosity`` are set, it holds
            :math:`\frac{u_t}{\phi} = 1`, i.e., the hyperbolic conservation law is

            .. math::

                \partial_t S + \partial_x f_w(S_w) = 0

        Parameters:
            params: _description_

        Raises:
            ValueError: _description_
            NotImplementedError: _description_

        """
        # General model and discretization parameters:
        self.domain_size: float = params["domain_size"]
        self.num_cells: int = params["num_cells"]
        self.total_flow: float = params.get("total_flow", 1.0)
        self.porosity: float | jnp.ndarray = params.get("porosity", 1.0)
        self.permeability: float | jnp.ndarray = params["permeability"]

        # Initial and boundary conditions:
        self.s_inlet: float = params["s_inlet"]

        self.s_initial: jnp.ndarray = params["s_initial"]
        if self.s_initial.shape != (self.num_cells,):
            raise ValueError(
                f"Initial saturation shape {self.s_initial.shape} does not match "
                f"number of cells {self.num_cells}."
            )

        # General fluid parameters:
        self.rp_model: str = params["rp_model"]
        self.mu_w: float = params["mu_w"]
        self.mu_n: float = params["mu_n"]

        # Brooks-Corey-Mualem rel. perm. model:
        self.nb: float = params.get("nb", 2)
        self.n1: float = params.get("n1", 2)
        self.n2: float = params.get("n2", 1 + 2 / self.nb)
        self.n3: float = params.get("n3", 1)

        # Corey rel. perm. model:
        self.nw: float = params.get("nw", 2)
        self.nn: float = params.get("nn", 2)

        # Buoyancy parameters:
        self.rho_w: float = params["rho_w"]
        self.rho_n: float = params["rho_n"]
        self.G: float = params["G"]
        self.theta: float = params["theta"]
        self.compute_gravity_number()

        # Initialize a placeholder for the linear system (Jacobian, residual).
        self.linear_system: tuple = (None, None)

    def reset(self) -> None:
        """Reset the model to its initial state, clearing any stored intermediate
        solutions or statistics."""
        self.linear_system = (None, None)

    def compute_gravity_number(self) -> None:
        if isinstance(self.permeability, jnp.ndarray):
            raise NotImplementedError(
                "Gravity number computation and upwinding with heterogeneous"
                " permeability is not yet implemented."
            )

        delta_rho = self.rho_w - self.rho_n
        delta_x = self.domain_size / self.num_cells
        delta_h = jnp.sin(self.theta) * delta_x

        self.gravity_number: float = float(
            self.permeability * delta_rho * self.G * delta_h / self.mu_n
        )
        logger.info(f"Gravity number: {self.gravity_number:.4f}")

    def mobility_w(
        self, s: jnp.ndarray, rp_model: Optional[str] = None, **kwargs
    ) -> jnp.ndarray:
        """Mobility of the wetting phase."""
        if rp_model is None:
            rp_model = self.rp_model
        if rp_model == "Brooks-Corey":
            k_w = jnp.where(s >= 0, s ** (self.n1 + self.n2 * self.n3), 0.0)
        elif rp_model == "Corey":
            k_w = jnp.where(s >= 0, s**self.nw, 0.0)
        elif rp_model == "linear":
            k_w = s
        else:
            raise ValueError(f"Unknown relative permeability model: {rp_model}")
        # Handle NaN values.
        return jnp.nan_to_num(k_w / self.mu_w, nan=0.0)  # type: ignore  # jnp.nan_to_num returns ndarray

    def mobility_n(
        self, s: jnp.ndarray, rp_model: Optional[str] = None, **kwargs
    ) -> jnp.ndarray:
        """Mobility of the nonwetting phase."""
        if rp_model is None:
            rp_model = self.rp_model
        if rp_model == "Brooks-Corey":
            k_n = jnp.where(
                s <= 1, (1 - s) ** self.n1 * (1 - s**self.n2) ** self.n3, 0.0
            )
        elif rp_model == "Corey":
            k_n = jnp.where(s <= 1, (1 - s) ** self.nn, 0.0)
        elif rp_model == "linear":
            k_n = 1 - s
        else:
            raise ValueError(f"Unknown relative permeability model: {rp_model}")
        # Handle NaN values.
        return jnp.nan_to_num(k_n / self.mu_n, nan=0.0)  # type: ignore  # jnp.nan_to_num returns ndarray

    def fractional_flow(self, s: jnp.ndarray, **kwargs) -> jnp.ndarray:
        r"""Compute the fractional flow function including buoyancy.

        .. math::
            f_w(S) = \frac{\lambda_w}{\lambda_t}
                     - \frac{\lambda_w \lambda_n}{\lambda_t \mu_n}
                       \frac{C_g}{u_t}

        where :math:`C_g` is the gravity number.

        Parameters:
            s: Saturation values.

        Returns:
            Fractional flow values (dimensionless).

        """
        m_w = self.mobility_w(s, **kwargs)
        m_n = self.mobility_n(s, **kwargs)
        m_t = m_w + m_n
        viscous_flow = m_w / m_t
        buoyancy_flow = (
            (m_w * m_n) / (m_t * self.mu_n) * self.gravity_number / self.total_flow
        )
        return viscous_flow - buoyancy_flow

    def face_transmissibility(self) -> jnp.ndarray:
        """Compute face transmissibilities using TPFA assuming uniform grid size and
        assuming unit area.

        Returns:
            T: Face transmissibilities.

        """
        area = 1.0
        dx = self.domain_size / self.num_cells

        if isinstance(self.permeability, jnp.ndarray):
            # Compute interface transmissibilities with harmonic averaging.
            transmissibilities = (
                (2 * area / dx) * (self.permeability[:-1] * self.permeability[1:])
            ) / (self.permeability[:-1] + self.permeability[1:])
        else:
            transmissibilities = self.permeability * jnp.ones(self.num_cells + 1)

        return transmissibilities

    def compute_face_fluxes(self, s: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Compute wetting phase fluxes at cell interfaces.

        When gravity is zero, total-flux upwinding is used (mobilities evaluated at the
        left cell).  When gravity is nonzero, phase-potential upwinding is applied via
        :meth:`upwind`.

        Args:
            s: Approximate cell saturations.

        Returns:
            F_w: Wetting phase fluxes at cell interfaces.

        """
        # Prepend the inlet boundary saturation.
        s = jnp.concatenate([jnp.array([self.s_inlet]), s])

        if self.G == 0:
            # Total-flux upwinding: mobilities evaluated at the left cell.
            m_w = self.mobility_w(s, **kwargs)
            m_n = self.mobility_n(s, **kwargs)
        else:
            # Phase-potential upwinding (Brenier & Jaffré).
            m_w, m_n = self.upwind(s, **kwargs)

        m_t = m_n + m_w

        # Viscous wetting flux.
        F_w_viscous = (self.total_flow / self.porosity) * (m_w / m_t)

        # Buoyancy wetting flux.
        dx = self.domain_size / self.num_cells
        transmissibilities = self.face_transmissibility()

        delta_rho = self.rho_w - self.rho_n
        buoyancy_potential = self.G * jnp.sin(self.theta) * dx * delta_rho

        F_w_buoyancy = (m_w * m_n / m_t) * (transmissibilities * buoyancy_potential)

        return F_w_viscous - F_w_buoyancy

    def upwind(self, s: jnp.ndarray, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Determine upwinded face mobilities via phase-potential upwinding.

        The upwind direction is determined per-phase using the fractional flow
        function :math:`f_w`:

        - **Wetting phase**: upwind from left when :math:`f_w(S_i) \geq 0`, i.e.
          viscous + buoyancy both push the wetting phase rightward.
        - **Non-wetting phase**: upwind from left when :math:`f_w(S_i) \leq 1`, i.e.
          the non-wetting fraction :math:`1 - f_w \geq 0`.

        This is equivalent to the Brenier & Jaffré formulation (Jiang & Tchelepi, 2018,
        §4) expressed in terms of the fractional flow instead of gravity number and
        mobility ratio.

        Selection is implemented with :func:`jnp.where` so that the function is
        compatible with JAX automatic differentiation (no integer indexing).

        Parameters:
            s: Padded saturation array ``[s_inlet, s_0, ..., s_{N-1}]`` of shape
                ``(N+1,)``.

        Returns:
            m_w: Upwinded wetting-phase mobility at each face, shape ``(N+1,)``.
            m_n: Upwinded non-wetting-phase mobility at each face, shape ``(N+1,)``.

        Raises:
            NotImplementedError: If permeability is heterogeneous (array-valued).

        """
        if isinstance(self.permeability, jnp.ndarray):
            raise NotImplementedError(
                "Phase-potential upwinding with heterogeneous permeability "
                "is not yet implemented."
            )

        # Extend with a zero-gradient ghost cell at the outlet boundary.
        s_extended = jnp.concatenate([s, s[-1:]])  # shape (N+2,)
        s_left = s_extended[:-1]  # shape (N+1,)
        s_right = s_extended[1:]  # shape (N+1,)

        # Mobilities evaluated at left and right cells (4 evaluations total).
        m_w_left = self.mobility_w(s_left, **kwargs)
        m_w_right = self.mobility_w(s_right, **kwargs)
        m_n_left = self.mobility_n(s_left, **kwargs)
        m_n_right = self.mobility_n(s_right, **kwargs)

        # Derive fractional flow at the left cell from already-computed mobilities
        # (avoids a redundant fractional_flow call that would re-evaluate mobilities).
        m_t_left = m_w_left + m_n_left
        f_w_left = m_w_left / m_t_left - (
            (m_w_left * m_n_left)
            / (m_t_left * self.mu_n)
            * self.gravity_number
            / self.total_flow
        )

        # Wetting phase: upwind from left when f_w >= 0.
        m_w = jnp.where(f_w_left >= 0.0, m_w_left, m_w_right)
        # Non-wetting phase: upwind from left when f_w <= 1.
        m_n = jnp.where(f_w_left <= 1.0, m_n_left, m_n_right)

        return m_w, m_n

    def residual(
        self, q: jnp.ndarray, dt: float, q_prev: Optional[jnp.ndarray] = None, **kwargs
    ) -> jnp.ndarray:
        """Compute the residual of the Buckley-Leverett system at (q, t + dt)

        Parameters:
            q: Current solution values.
            dt: Time step size.
            q_prev: Previous time step solution. Defaults to the initial condition if
                not provided.

        Returns:
            r: Residual vector.

        """

        if q_prev is None:
            q_prev = self.s_initial.copy()

        # Compute fluxes.
        F_w = self.compute_face_fluxes(q, **kwargs)

        # Residuals for flow and transport equations.
        r = (q - q_prev) / dt + F_w[1:] - F_w[:-1]

        return r

    def jacobian(
        self, q: jnp.ndarray, dt: float, q_prev: Optional[jnp.ndarray] = None, **kwargs
    ) -> jnp.ndarray:
        """Compute the Jacobian of the system at (q, t + dt) using automatic
        differentiation.

        The ``jacrev`` function is cached on the instance to avoid re-creating the
        closure on every call, which would trigger JAX re-tracing.

        Parameters:
            q: Current saturation values.
            dt: Time step size.
            q_prev: Previous saturation values. Defaults to the initial condition if not
            provided.

        Returns:
            J: Jacobian matrix.

        """
        if not hasattr(self, "_jacrev_fn"):
            self._jacrev_fn = jax.jacrev(self.residual, argnums=0)
        return self._jacrev_fn(q, dt, q_prev=q_prev, **kwargs)
