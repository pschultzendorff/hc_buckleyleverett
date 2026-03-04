import typing
from typing import Any, Callable, Optional, Protocol

import jax.numpy as jnp

# Conditional importing to avoid runtime issues:
if not typing.TYPE_CHECKING:

    class BuckleyLeverettModelProtocol: ...

    class HCProtocol: ...


else:

    class BuckleyLeverettModelProtocol(Protocol):
        domain_size: float
        num_cells: int
        total_flow: float
        s_inlet: float
        s_initial: jnp.ndarray
        linear_system: tuple
        porosity: float | jnp.ndarray
        permeability: float | jnp.ndarray

        mu_n: float

        G: float
        rho_w: float
        rho_n: float
        theta: float
        gravity_number: float

        def __init__(self, params: dict[str, Any]) -> None: ...

        def reset(self) -> None:
            """Reset the model to its initial state, clearing any stored intermediate
            solutions or statistics."""
            ...

        def mobility_w(
            self, s: jnp.ndarray, rp_model: Optional[str] = None
        ) -> jnp.ndarray:
            """Mobility of the wetting phase."""
            ...

        def mobility_n(
            self, s: jnp.ndarray, rp_model: Optional[str] = None
        ) -> jnp.ndarray:
            """Mobility of the nonwetting phase."""
            ...

        def fractional_flow(self, s: jnp.ndarray, **kwargs) -> jnp.ndarray:
            """Compute the fractional flow function including buoyancy."""
            ...

        def face_transmissibility(self) -> jnp.ndarray:
            """Compute face transmissibilities using TPFA assuming uniform grid size and
            assuming unit area.

            """
            ...

        def compute_face_fluxes(self, s: jnp.ndarray) -> jnp.ndarray:
            """Compute wetting phase fluxes at cell interfaces.

            Args:
                s: Approximate cell saturations.

            Returns:
                F_w: Wetting phase fluxes at cell interfaces.

            """
            ...

        def residual(
            self, q: jnp.ndarray, dt: float, q_prev: jnp.ndarray | None = None, **kwargs
        ) -> jnp.ndarray:
            """Compute the residual of the Buckley-Leverett system at (q, t + dt)"""
            ...

        def jacobian(
            self, q: jnp.ndarray, dt: float, q_prev: jnp.ndarray | None = None, **kwargs
        ) -> jnp.ndarray:
            """Compute the Jacobian of the system at (q, t + dt) using automatic
            differentiation."""
            ...

    class HCProtocol(Protocol):
        beta: float
        betas: list[float]
        intermediate_solutions: list[jnp.ndarray]

        def reset(self) -> None:
            """Reset the homotopy continuation model to its initial state and empty the
            statistics.

            """
            ...

        def store_curve_data(
            self,
            beta: float,
            q: jnp.ndarray,
            dt: float,
            q_prev: Optional[jnp.ndarray] = None,
        ) -> None:
            """Store the homotopy curve data at the current point."""
            ...

        def mobility_w_con_hull_gravity(self, s: jnp.ndarray) -> jnp.ndarray:
            """Interpolate the pre-computed convex-hull wetting mobility (gravity case)."""
            ...

    class HCModelProtocol(HCProtocol, BuckleyLeverettModelProtocol):
        """Combined protocol for HC models that satisfy both the Buckley-Leverett
        model interface and the HC interface."""

    class HCAnalysisProtocol(Protocol):
        curvature_vectors: list[jnp.ndarray]
        newton_rs: list[float]
        curvature_lambda_vectors: list[jnp.ndarray]

        def hessian_tensor(self, f: Callable) -> Callable:
            """Compute the Hessian tensor of a scalar-valued function."""
            ...

    class HCModelandAnalysisProtocol(HCModelProtocol, HCAnalysisProtocol):
        """Full homotopy continuation protocol for HC modesl that satisfy the
        Buckley-Leverett model interface, the basic HC interface, and the HC analysis
        interface."""
