"""Newton and homotopy continuation solver for two-phase flow problems."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from hc_buckleyleverett.buckley_leverett.hc import DiffusionHCMixin, HCMixin
from hc_buckleyleverett.utils.ui import (
    DummyProgressBar,
    logging_redirect_tqdm,
    progressbar_class,
)

if TYPE_CHECKING:
    from hc_buckleyleverett.buckley_leverett.protocol import (
        BuckleyLeverettModelProtocol,
        HCModelProtocol,
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def newton(
    model: BuckleyLeverettModelProtocol,
    q_init: jnp.ndarray,
    q_prev: jnp.ndarray,
    dt: float,
    max_iter: int = 20,
    tol: float = 1e-5,
    appleyard: bool = False,
    **kwargs,
) -> tuple[jnp.ndarray, bool, int]:
    """Solve the system using Newton"s method.

    Returns:
        q: The solution at the new time step.
        converged: Whether the solver converged within the maximum number of iterations.
        num_iterations: The number of iterations taken by the solver.

    Raises:
        ValueError: If the Newton step results in NaN or Inf values, which may indicate
        divergence or numerical issues. In this case, the solver should be terminated
        to prevent further issues. The user should check the input data and model
        parameters

    """
    q = q_init.copy()
    converged = False

    if kwargs.get("progressbars", True):
        newton_progressbar: DummyProgressBar | Any = progressbar_class(  # type: ignore[assignment]
            range(max_iter), desc="Newton iterations", position=0, leave=False
        )
    else:
        newton_progressbar = DummyProgressBar(max_iter)

    with logging_redirect_tqdm([logger]):
        for i in newton_progressbar:
            r = model.residual(q, dt=dt, q_prev=q_prev, **kwargs)
            residual_norm = float(jnp.linalg.norm(r) / jnp.sqrt(r.size))
            newton_progressbar.set_postfix({"residual_norm": residual_norm})

            if residual_norm < tol:
                converged = True
                break

            # Newton step:
            J = model.jacobian(q, dt=dt, q_prev=q_prev, **kwargs)
            model.linear_system = (J, r)
            dx = jnp.linalg.solve(J, -r)

            if jnp.isnan(dx).any() or jnp.isinf(dx).any():
                raise ValueError("Newton step resulted in NaN or Inf values.")

            # Appleyard damping: Limit cellwise saturation updates to [-0.2, 0.2].
            if appleyard:
                dx = jnp.clip(dx, -0.2, 0.2)

            q += dx

            # Physical damping: Restrict saturation values to [0,1] (plus small epsilon
            # to avoid numerical issues).
            q = jnp.clip(q, 1e-15, 1 - 1e-15)

    logger.info(
        f"Newton solver for model {model.__class__.__name__} "
        + ("converged" if converged else "did not converge")
        + f" in {i} iterations."
    )

    return q, converged, i


def hc(
    model: HCModelProtocol,
    q_prev: jnp.ndarray,
    q_init: jnp.ndarray,
    dt: float,
    hc_decay: float = 1 / 30,
    hc_max_iter: int = 31,
    initial_grid_search: bool = False,
    **kwargs,
) -> tuple[jnp.ndarray, bool, int]:
    """Solve the system using homotopy continuation with Newton"s method.

    Parameters:
        model: The model to solve.
        q_prev: Previous solution, used as the initial guess for Newton's method.
        q_init: Initial guess for the homotopy continuation.
        dt: Time step size.
        hc_decay: Linear decay for the homotopy parameter :math:`beta`.
        hc_max_iter: Maximum number of homotopy continuation iterations.
        initial_grid_search: Whether to perform an initial grid search to find a good
            starting point for the homotopy continuation instead of using ``q_init``.
        **kwargs: Additional keyword arguments for the Newton solver.

    Returns:
        q: The solution at the new time step.
        converged: Whether the solver converged within the maximum number of iterations.
        num_iterations: The number of homotopy continuation iterations taken by the
            solver.


    Raises:
        ValueError: If the Newton step results in NaN or Inf values.

    """
    beta: float = 1.0

    if kwargs.get("progressbars", True):
        hc_progressbar: DummyProgressBar | Any = progressbar_class(  # type: ignore[assignment]
            range(hc_max_iter), desc="Homotopy continuation", position=1, leave=False
        )
    else:
        hc_progressbar = DummyProgressBar(hc_max_iter)

    with logging_redirect_tqdm([logger]):
        for i in hc_progressbar:
            hc_progressbar.set_postfix({r"$\lambda$": beta})

            if i == 0:
                # Initial guess either via grid search or provided initial value.
                if initial_grid_search:
                    q = grid_search(model, q_prev, dt, beta=beta, **kwargs)
                else:
                    q = q_init.copy()

            # Previous solution for the predictor step. Newton's method for the
            # corrector step.
            try:
                q, converged, _ = newton(
                    model,
                    q,
                    q_prev,
                    dt,
                    beta=beta,
                    **kwargs,
                )
            except ValueError as _:
                converged = False
                raise _

            if converged:
                # Store data for the homotopy curve BEFORE updating beta.
                model.store_curve_data(beta, q, dt, q_prev=q_prev)

                # Update the homotopy parameter beta only now.
                beta -= hc_decay

                # For convenience, ensure beta is non-negative and set to zero if it is
                # very close to zero.
                if abs(beta) < 1e-3 or beta < 0:
                    beta = 0.0

            else:
                logger.info(
                    f"Model {model.__class__.__name__} did not converge at continuation"
                    + f" step {i + 1}, lambda={beta}."
                )
                break

    return q, converged, i


def grid_search(
    model: HCModelProtocol,
    q_prev: jnp.ndarray,
    dt: float,
    **kwargs,
) -> jnp.ndarray:
    """Perform  grid search to find a good starting point for HC.

    Note:
        This works only for the 2-cell problem.

    Parameters:
        model: The model to solve.
        q_prev: Previous solution, used as the initial guess for Newton's method.
        dt: Time step size.
        **kwargs: Additional keyword arguments for the Newton solver.

    Returns:
        q_best: The best initial guess found from the grid search.

    """
    num_s_initial = kwargs.get("num_s_initial", 10)

    s_inits = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0.0, 1.0, num=num_s_initial),
            jnp.linspace(0.0, 1.0, num=num_s_initial),
        ),
        axis=-1,
    ).reshape(-1, 2)

    best_residual_norm = jnp.inf
    q_best = q_prev

    # Loop through all possible initial guesses, check if they converge with Newton,
    # pick the one with the lowest initial residual norm.
    for s_init in s_inits:
        q_init = s_init.flatten()
        try:
            q_candidate, converged, _ = newton(
                model,
                q_init,
                q_prev,
                dt,
                **kwargs,
            )
            if converged:
                r_candidate = model.residual(
                    q_candidate, dt=dt, q_prev=q_prev, **kwargs
                )
                residual_norm = jnp.linalg.norm(r_candidate) / jnp.sqrt(
                    r_candidate.size
                )
                if residual_norm < best_residual_norm:
                    best_residual_norm = residual_norm
                    q_best = q_candidate
        except ValueError:
            continue

    return q_best


def solve(
    model: BuckleyLeverettModelProtocol,
    final_time: float,
    num_time_steps: int,
    **kwargs,
):
    # Setup the simulation.
    dt = final_time / num_time_steps
    solutions: list[jnp.ndarray] = [model.s_initial]
    model.reset()
    solver = hc if isinstance(model, HCMixin) else newton

    time_progressbar_position = 2 if hasattr(model, "beta") else 1
    if kwargs.get("progressbars", True):
        time_progressbar: DummyProgressBar | Any = progressbar_class(  # type: ignore[assignment]
            range(num_time_steps),
            desc="Time steps",
            position=time_progressbar_position,
            leave=False,
        )
    else:
        time_progressbar = DummyProgressBar(num_time_steps)

    with logging_redirect_tqdm([logger]):
        for i in time_progressbar:
            time_progressbar.set_postfix({"time_step": i + 1})

            # Previous solution is the initial guess for the solver.
            q_prev = solutions[-1]

            # For diffusion based HC, update the diffusion coefficient, which depends on
            # the time step size.
            if isinstance(model, DiffusionHCMixin):
                model.update_adaptive_diffusion_coeff(dt)
                logger.info(
                    f"Model {model.__class__.__name__} updated adaptive diffusion"
                    + f" cofficient.\n New values: {model.adaptive_diffusion_coeff}."
                )

            try:
                # Ignore pylance complaining; solver is either hc or newton, both accept
                # model class.
                q_next, converged, _ = solver(model, q_prev, q_prev, dt=dt, **kwargs)  # type: ignore
            except ValueError as _:
                converged = False
                raise _

            if converged:
                solutions.append(q_next)
            else:
                logger.info(
                    f"Model {model.__class__.__name__} did not converge at time step"
                    + f" {i + 1}."
                )
                break

    return solutions, converged
