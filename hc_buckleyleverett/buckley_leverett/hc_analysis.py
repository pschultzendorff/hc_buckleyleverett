r"""Homotopy continuation evaluation methods.

For more information on evaluating the curvature of homotopy solution curves, see

Brown, D.A. and Zingg, D.W. (2016) ‘Efficient numerical differentiation of
implicitly-defined curves for sparse systems’, Journal of Computational and Applied
Mathematics, 304, pp. 138–159. Available at: https://doi.org/10.1016/j.cam.2016.03.002.

and

Brown, D.A. and Zingg, D.W. (2017) ‘Design and evaluation of homotopies for efficient
and robust continuation’, Applied Numerical Mathematics, 118, pp. 150–181. Available at:
https://doi.org/10.1016/j.apnum.2017.03.001.

We follow the notation from these papers. Due to the overlap of the Python ``lambda``
keyword and the homotopy continuation parameter, we use ``beta`` instead of ``lambda``
in the code. In the mathematical formulas, we use `\lambda` to be consisten with the
papers.

"""

import logging
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from hc_sandbox.buckley_leverett.protocol import (
    BuckleyLeverettModelProtocol,
    HCProtocol,
)
from hc_sandbox.buckley_leverett.solvers import newton

logger = logging.getLogger(__name__)


def hessian_tensor(f: Callable) -> Callable:
    """Returns a function that computes the Hessian tensor of the multi-variable
    function f at a point q.

    Note: Uses ``jacfwd(jacrev(f))`` to compute all per-component Hessians in a single
    pass, avoiding a Python for-loop over the C output components.

    Returns:
        ``shape=(C, N, N)``: Where C = number of output components, N = len(q).

    """

    @jax.jit
    def hessian_tensor_fn(q, *args, **kwargs):
        return jax.jacfwd(jax.jacrev(f, argnums=0), argnums=0)(q, *args, **kwargs)

    return hessian_tensor_fn


def hc_hessian_tensor(h: Callable, h_beta_deriv: Callable) -> Callable:
    r"""Extends :math:`hessian_tensor` to the homotopy case, where the function h is a
    convex combination of :math:`F` and :math:`G`.

    With :math:`\mathcal{H}(\mathbf{q}, \lambda) = \lambda G(\mathbf{q}) + (1 - \lambda) F(\mathbf{q})`,
    we have

    .. math::
        \nabla^2 \mathcal{H} =
        \begin{pmatrix}
            \nabla^2_\mathbf{q} \mathcal{H} & \nabla_\mathbf{q} \partial_\lambda \mathcal{H} \\
            \nabla_\mathbf{q} \partial_\lambda \mathcal{H} & 0
        \end{pmatrix}
        = 
        \begin{pmatrix}
            \nabla^2_\mathbf{q} \mathcal{H} & \nabla_\mathbf{q} G - \nabla_\mathbf{q} F \\
            \nabla_\mathbf{q} G - \nabla_\mathbf{q} F & 0
        \end{pmatrix}.

    Parameters:
        h: Residual function of the homotopy problem at the current beta.
        h_beta_deriv: Function that computes the derivative of the homotopy residual
            w.r.t. beta.
            

    Returns:
        per_component_hessian: Function that computes the Hessian tensor of the homotopy
            function at a point (q, beta). The tensor has the block form 
            [[nabla_q^2 H_i,   nabla_q (\partial_lambda H_i) ],
            [ nabla_q (\partial_lambda H_i).T, 0            ]]
            for each component i.


    """

    # Get the function to calculate :math:`\nabla^2_\mathbf{q} H`.
    q_hessian_func = hessian_tensor(h)

    @jax.jit
    def hc_hessian_tensor_fn(q, *args, **kwargs):
        # Get the Hessian w.r.t q. ``shape=(C, N, N)``, where C is components of H, N is
        # len(q).
        H_q = q_hessian_func(q, *args, **kwargs)

        # Get the mixed partials :math:`\nabla_\mathbf{q} \partial_\lambda \mathcal{H}`
        # ``shape=(C, N)``, where C is components of H, N is len(q).
        mixed_partials = jax.jacobian(h_beta_deriv, argnums=0)(q, *args, **kwargs)

        # Build the full (C, N+1, N+1) block Hessian in a vectorized manner.
        # Top block: [H_q (C,N,N) | mixed_partials (C,N,1)]
        top_block = jnp.concatenate([H_q, mixed_partials[:, :, None]], axis=2)
        # Bottom row: [mixed_partials (C,1,N) | zeros (C,1,1)]
        zeros = jnp.zeros((H_q.shape[0], 1))
        bottom_row = jnp.concatenate([mixed_partials, zeros], axis=1)[:, None, :]
        # Full Hessian: (C, N+1, N+1)
        return jnp.concatenate([top_block, bottom_row], axis=1)

    return hc_hessian_tensor_fn


def apply_hessian(H: jax.Array, u: jax.Array, v: jax.Array) -> jax.Array:
    """Apply the Hessian tensor H to vectors u and v."""
    return jnp.einsum("ijk,j,k->i", H, u, v)


class HCAnalysisMixin(HCProtocol, BuckleyLeverettModelProtocol):
    r"""Mixin for analyzing curvature and Newton convergence along the HC curve.

    Note: To compute the tangent and curvature, it would be cleaner to define two
    residual functions, ``residual_initial`` and ``residual_target``, and compute
    Jacobian and Hessian for both of them. However, this would require duplicating the
    code for the residuals for all concrete homotopies.
    Instead, we leave the concrete implementation of the homotopy to the subclasses and
    compute tangent and curvatures by passing `beta = 1` and `beta = 0` to the residual
    and Jacobian functions, respectively.

    """

    def __init__(self, params: dict[str, Any], **kwargs) -> None:
        super().__init__(params=params, **kwargs)  # type: ignore

        # Precompute the function to compute the Hessian tensor of the homotopy
        # residual.
        self.hc_hessian_tensor_fn = hc_hessian_tensor(self.residual, self.h_beta_deriv)

        # Lists to store curve data.
        self.tangents: list[jnp.ndarray] = []
        self.curvature_vectors: list[jnp.ndarray] = []
        self.curvature_lambda_vectors: list[jnp.ndarray] = []
        self.newton_rs: list[float] = []

    def reset(self) -> None:
        """Reset tangent and curvature lists."""
        # Mixin; concrete reset provided by HCMixin via MRO.
        super().reset()  # type: ignore
        self.tangents = []
        self.curvature_vectors = []
        self.curvature_lambda_vectors = []
        self.newton_rs = []

    def store_curve_data(
        self,
        beta: float,
        q: jnp.ndarray,
        dt: float,
        q_prev: Optional[jnp.ndarray] = None,
        parametrization: str = "arclength",
    ) -> None:
        """Store tangent and curvature vectors for the homotopy curve."""
        # Mixin; concrete store_curve_data provided by HCMixin via MRO.
        super().store_curve_data(beta, q, dt, q_prev=q_prev)  # type: ignore

        self.tangents.append(
            self.tangent(
                self.betas[-1],
                q,
                dt,
                q_prev=q_prev,
                jac=self.linear_system[0],
                parametrization=parametrization,
            )  # type: ignore  # linear_system[0] is the Jacobian matrix
        )
        self.curvature_vectors.append(
            self.curvature_vector(
                self.betas[-1],
                q,
                dt,
                q_prev=q_prev,
                # Make use of the precomputed Jacobian and tangent.
                jac=self.linear_system[0],  # type: ignore  # linear_system[0] is the Jacobian matrix
                tangent=self.tangents[-1],
                parametrization=parametrization,
            )
        )
        self.curvature_lambda_vectors.append(
            self.curvature_vector(
                self.betas[-1],
                q,
                dt,
                q_prev=q_prev,
                # Make use of the precomputed Jacobian and tangent.
                jac=self.linear_system[0],  # type: ignore  # linear_system[0] is the Jacobian matrix
                parametrization="lambda",
            )
        )

        if q_prev is not None:
            self.newton_rs.append(
                self.convergence_metric(
                    self.betas[-1],
                    q,
                    dt,
                    q_prev,
                    tangent=self.tangents[-1],
                    parametrization=parametrization,
                )
            )

    def h_beta_deriv(
        self, q: jnp.ndarray, dt: float, q_prev: Optional[jnp.ndarray] = None, **kwargs
    ) -> jnp.ndarray:
        r"""Compute the derivative of the homotopy problem with respect to
         :math:`\lambda`.

        .. math::
            \frac{\partial \mathcal{R}_H}{\partial \lambda} = \mathcal{R}_g - \mathcal{R}_F,


        where :math:`\mathcal{R}_G` is the residual of the simpler system and
        :math:`\mathcal{r}_F` is the residual of the actual system.

        """
        r_g = self.residual(q, dt, q_prev=q_prev, beta=1.0)
        r_f = self.residual(q, dt, q_prev=q_prev, beta=0.0)
        return r_g - r_f

    def tangent(
        self,
        beta: float,
        q: jnp.ndarray,
        dt: float,
        q_prev: Optional[jnp.ndarray] = None,
        jac: Optional[jnp.ndarray] = None,
        parametrization: str = "arclength",
    ) -> jnp.ndarray:
        r"""Compute the tangent vector of the homotopy curve at :math:`(\mathbf{q},
         \lambda)`.

        For the arclength parametrization, the unit tangent vector is given by
        .. math::
            \dot{c}(s)  = \xi \frac{\tau}{\|\tau\|},

        where :math:`\tau = \begin{pmatrix} \mathbf{z} \\ -1 \end{pmatrix}` is the
        unnormalized tangent vector and :math:`\xi` is a scaling factor to ensure
        tracing in a consisten direction. The unnormalized tangent vector is found by
        solving the linear system

        .. math::

            \nabla_{\mathbf{q}} \mathcal{H}(\mathbf{q}, \lambda) \mathbf{z} =
            \partial_{\lambda} \mathcal{H}(\mathbf{q}, \lambda).

        Note: Assuming that no bifurcation points are present, we can set :math:`\xi =
        -1`.

        The partial unit tangent vector for the :math:`\lambda` parametrization is given
        by

        .. math::
            \dot{\mathbf{q}} = \sqrt(\mathbf{z}\cdot\mathbf{z} + 1) \xi \frac{\mathbf{z}}{\|\tau\|}.

        Note: In the :math:`\lambda` parametrization, the full tangent vector includes
        the derivative :math:`\lambda' = -1`, which is **not** included by this method.
        :meth:`curvature_vector` computes only the partial curvature
        :math:`\mathbf{q}''` for the :math:`\lambda` parametrization.

        Parameters:
            beta:
            q: Current time step solution.
            dt: Time step size.
            q_prev: Previous time step solution values. Defaults to the initial
                condition if not provided.
            jac: Jacobian matrix of the system at (q, t + dt). If not provided, it will
                be computed using automatic differentiation.
            parametrization: Parametrization of the homotopy curve. Either "arclength"
                or "lambda". Defaults to "arclength".

        Returns:
            tangent: Unit tangent vector of the homotopy curve at (q, t + dt) w.r.t. the
                given parametrization.

        """
        xi = -1.0

        # Note: The beta pass to :meth:`h_beta_deriv` is only needed for testing.
        b = self.h_beta_deriv(q, dt, q_prev=q_prev, beta=beta)
        A = self.jacobian(q, dt, q_prev=q_prev, beta=beta) if jac is None else jac

        z = jnp.linalg.solve(A, b)
        tau = jnp.concatenate([z, jnp.array([-1.0])])

        if parametrization == "arclength":
            return xi * tau / jnp.linalg.norm(tau)

        elif parametrization == "lambda":
            return xi * z / jnp.linalg.norm(tau) * jnp.sqrt(z @ z + 1)

        else:
            raise NotImplementedError(
                f"Parametrization {parametrization} not implemented. Only 'arclength'"
                "and 'lambda' are supported."
            )

    def curvature_vector(
        self,
        beta: float,
        q: jnp.ndarray,
        dt: float,
        q_prev: Optional[jnp.ndarray] = None,
        jac: Optional[jnp.ndarray] = None,
        tangent: Optional[jnp.ndarray] = None,
        parametrization: str = "arclength",
    ) -> jnp.ndarray:
        r"""Compute the curvature vector of the homotopy curve at :math:`(\mathbf{q},
        \lambda)`.

        For the arclength parametrization, the curvature vector :math:`\ddot{c}(s)` is
        found by solving

        .. math::

            \nabla \mathcal{H}(c(s)) \ddot{c}(s) = -\mathbf{w}_2,

        where :math:`\mathbf{w}_2` is the Hessian of the residual applied to the tangent
        vector :math:`\dot{c}(s)`, i.e. :math:`\mathbf{w}_2 = \nabla^2 \mathcal{H}(c(s))
        [\dot{c}(s), \dot{c}(s)]`.

        In practice, one solves

        .. math::

            \nabla \mathcal_\mathbf{q}\mathcal{H}(c(s)) \mathbf{z}_2 = -\mathbf{w}_2,

        s.t. :math:`\mathbf{z}_2 = \ddot{\mathbf{q}} + \ddot{\lambda} \mathbf{z}`.
        :math:`\ddot{\lambda}` and :math:`\ddot{\mathbf{q}}` are then found via

        .. math::
            \ddot{\lambda} = -\frac{\mathbf{z}_2 \cdot \mathbf{z}_2}{\dot{\lambda}^2}
            \text{ and }
            \ddot{\mathbf{q}} = \mathbf{z}_2 - \ddot{\lambda} \mathbf{z}.


        For the :math:`\lambda` parametrization, defined via :math:`\lambda'(r) = -1`,
        the partial curvature vector :math:`\ddot{\mathbf{q}}(r)` is obtained by solving
        :math:
            \nabla_\mathbf{q} \mathcal{H}(c(r)) \ddot{\mathbf{q}}(r) = -\mathbf{w}'_2,

        where :math:`\mathbf{w}'_2 = \nabla^2 \mathcal{H}(c(r)) [\dot{c}(r),
        \dot{c}(r)]` is the Hessian of the residual applied to the partial tangent
        vector :math:`\dot{c}(r) = (\dot{\mathbf{q}}(r), -1)`.


        Parameters:
            q: Current time step solution.
            dt: Time step size.
            q_prev: Previous time step solution values. Defaults to the initial
                condition if not provided.
            jac: Jacobian matrix of the system at (q, t + dt). If not provided, it will
                be computed using automatic differentiation.
            tangent: Tangent vector of the homotopy curve at (q, t + dt) w.r.t. the
                arclength parametrization. If not provided, it will be computed using
                the :meth:`tangent` method.
            parametrization: Parametrization of the homotopy curve. Either "arclength"
                or "lambda". Defaults to "arclength".

        """
        # Compute jacobian and tangent if not provided.
        if jac is None:
            jac = self.jacobian(q, dt, q_prev=q_prev, beta=beta)
        if tangent is None:
            tangent = self.tangent(
                beta, q, dt, q_prev=q_prev, jac=jac, parametrization=parametrization
            )

        if parametrization == "arclength":
            # Recover some intermediate variables from the tangent vector.
            q_dot = tangent[:-1]
            beta_dot = tangent[-1]
            z = q_dot / (-beta_dot)

            # Compute the Hessian w.r.t. to :math:`c(s)` and apply it to the
            # corresponding tangent vector :math:`\dot{c}(s)`.
            H = self.hc_hessian_tensor_fn(q, dt, q_prev=q_prev, beta=beta)
            w2 = apply_hessian(H, tangent, tangent)

            z2 = jnp.linalg.solve(jac, -w2)

            beta_ddot = (z2 @ z) / beta_dot**2
            q_ddot = z2 - beta_ddot * z

            return jnp.concatenate([q_ddot, jnp.array([beta_ddot])])

        elif parametrization == "lambda":
            # Append the derivative of :math:`\lambda` w.r.t. :math:`r` to the tangent
            # vector.
            tangent = jnp.concatenate([tangent, jnp.array([-1.0])])

            # Compute the Hessian w.r.t. to :math:`c(r)` and apply it to the
            # corresponding tangent vector :math:`\dot{c}(r)`.
            H = self.hc_hessian_tensor_fn(q, dt, q_prev=q_prev, beta=beta)
            w2_prime = apply_hessian(H, tangent, tangent)

            q_ddot = jnp.linalg.solve(jac, -w2_prime)

            return q_ddot

        else:
            raise NotImplementedError(
                f"Parametrization {parametrization} not implemented. Only 'arclength'"
                "and 'lambda' are supported."
            )

    def curvature(
        self,
        beta: float,
        q: jnp.ndarray,
        dt: float,
        q_prev: Optional[jnp.ndarray] = None,
        jac: Optional[jnp.ndarray] = None,
        tangent: Optional[jnp.ndarray] = None,
        parametrization: str = "arclength",
    ):
        r"""Compute the curvature vector of the homotopy curve at :math:`(\mathbf{q},
        \lambda)`.

        For an arclength parametrization, this is the full curvature. For the
        :math:`\lambda` parametrization, this is the partial curvature in the components
        :math:`\mathbf{q}`.


        Parameters:
            beta:
            q: Current time step solution.
            dt: Time step size.
            q_prev: Previous time step solution values. Defaults to the initial
                condition if not provided.
            jac: Jacobian matrix of the system at (q, t + dt). If not provided, it will
                be computed using automatic differentiation.
            tangent: Tangent vector of the homotopy curve at (q, t + dt) w.r.t. the
                arclength parametrization. If not provided, it will be computed using
                the :meth:`tangent` method.
            parametrization: Parametrization of the homotopy curve. Either "arclength"
                or "lambda". Defaults to "arclength".

        Returns:
            curvature: Curvature of the homotopy curve at (q, t + dt) w.r.t. the given
                parametrization.

        """
        curvature_vector = self.curvature_vector(
            beta,
            q,
            dt,
            q_prev=q_prev,
            jac=jac,
            tangent=tangent,
            parametrization=parametrization,
        )
        return jnp.linalg.norm(curvature_vector)

    def curvature_fd(self) -> jnp.ndarray:
        """Compute the curvature of the homotopy curve at the stored points using finite
        differences."""
        hh = (jnp.asarray(self.betas[1:]) - jnp.asarray(self.betas[:-1]))[..., None]
        curvatures_approx = (
            jnp.asarray(self.tangents[1:]) - jnp.asarray(self.tangents[:-1])
        ) * (2 / (hh + hh))
        return jnp.linalg.norm(curvatures_approx, axis=-1)

    def convergence_metric(
        self,
        beta: float,
        q: jnp.ndarray,
        dt: float,
        q_prev: jnp.ndarray,
        tangent: Optional[jnp.ndarray] = None,
        parametrization: str = "arclength",
    ) -> float:
        r"""Compute a metric of Newton convergence along the homotopy curve. For example,
        this could be the average curvature or the maximum curvature along the curve.


        Given a previous solution along the curve, :math:`\mathbf{q}(s)`, the initial
        guess of the Newton corrector step lies along the line
        :math:`\mathbf{q}_\mr{pred}(s)(\gamma) = \{\mathbf{q}(s) + \gamma
        \dot{\mathbf{q}}(s) \mid \gamma \in \mathbb{R}^+\}`. The measures for Newton
        convergence are

        .. math::

            r(s) = \max_r \{r \mid \text{Newton converges from }
            \mathbf{q}_\mr{pred}(s)(\gamma) \, \forall \, 0 < \gamma \leq r\},

            \hat{r}(s) = r(s) \cdot \dot{\lambda}(s),

        where the latter is the projection of the maximum admissible step size onto the
        homotopy parameter axis.

        Returns:
            r * tangent_beta: Maximum admissible step size projected onto the homotopy
                parameter axis.

        """
        current_beta = beta
        current_solution = q.copy()  # ``shape=(N,)``
        if tangent is None:
            tangent = self.tangent(
                current_beta,
                current_solution,
                dt,
                q_prev=q_prev,
                jac=self.linear_system[0],
                parametrization=parametrization,
            )  # ``shape=(N + 1,)``, includes beta.
        tangent_q = tangent[:-1]
        tangent_beta = float(tangent[-1])
        assert tangent_beta > 0, f"tangent_beta is negative: {tangent_beta = }"

        # Evaluate the maximum step size along the tangent before lambda becomes
        # negative.
        max_gamma = current_beta / tangent_beta
        small_gamma = max_gamma / 20.0

        # Use binary search to find the maximum admissible gamma.
        lo, hi = small_gamma, max_gamma
        converged_gamma = 0.0
        num_evals = 0

        # Check whether the largest admissible step size converges.
        trial_beta = current_beta - max_gamma * tangent_beta
        trial_q_init = current_solution + max_gamma * tangent_q
        num_evals += 1
        if check_newton_convergence(self, trial_beta, trial_q_init, dt, q_prev) >= 1:
            logger.info(
                f"Evaluated Newton convergence for {num_evals} candidate gammas"
            )
            return max_gamma * tangent_beta

        # Verify that the smallest step converges at all.
        trial_beta = current_beta - small_gamma * tangent_beta
        trial_q_init = current_solution + small_gamma * tangent_q

        num_evals += 1
        if check_newton_convergence(self, trial_beta, trial_q_init, dt, q_prev) == -1:
            # Even the smallest step fails — converged_gamma stays 0.
            logger.info(
                f"Evaluated Newton convergence for {num_evals} candidate gammas"
            )
            return converged_gamma * tangent_beta

        converged_gamma = small_gamma

        # Binary search over [small_gamma, max_gamma] for the boundary.
        max_bisection_steps = 5
        for _ in range(max_bisection_steps):
            mid = (lo + hi) / 2.0
            trial_beta = current_beta - mid * tangent_beta

            # FIXME Is the sign correct here or should it be minus?
            trial_q_init = current_solution + mid * tangent_q
            num_evals += 1
            trial_beta = current_beta - mid * tangent_beta
            if (
                check_newton_convergence(self, trial_beta, trial_q_init, dt, q_prev)
                != -1
            ):
                converged_gamma = mid
                lo = mid
            else:
                hi = mid

        logger.info(f"Evaluated Newton convergence for {num_evals} candidate gammas")

        return converged_gamma * tangent_beta

    # def find_convergence_region(
    #     model: HCModel,
    #     x_prev: jnp.ndarray,
    #     dt: float,
    #     **kwargs,
    # ) -> tuple[jnp.ndarray, jnp.ndarray]:
    #     """Find region of Newton convergence for a given model."""
    #     num = kwargs.get("num_s_initial", 10)

    #     s_inits = jnp.stack(
    #         jnp.meshgrid(
    #             jnp.linspace(0.0, 1.0, num=num),
    #             jnp.linspace(0.0, 1.0, num=num),
    #         ),
    #         axis=-1,
    #     ).reshape(-1, 2)

    #     num_iters = jnp.zeros((s_inits.shape[0],), dtype=jnp.int32)

    #     # No jax.vmap here because we have python control flow inside the Newton solver.
    #     # TODO Fix this with jax.lax.cond?
    #     for s_init in s_inits:
    #         num_iters = num_iters.at[
    #             jnp.where(jnp.all(s_inits == s_init, axis=-1))[0][0]
    #         ].set(check_newton_convergence(model, s_init, x_prev, dt))

    #     return num_iters, s_inits

    # def convergence_region_along_curve(
    #     model: HCModel,
    #     x_prev: jnp.ndarray,
    #     dt: float,
    #     **kwargs,
    # ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    #     """Find region of Newton convergence along the homotopy curve."""
    #     betas = jnp.linspace(0.0, 1.0, num=kwargs.get("num_betas", 10))

    #     num_iters_over_beta = []
    #     s_inits_over_beta = []

    #     # No jax.vmap here because we mutate model.beta.
    #     for beta in betas:
    #         logger.info(f"Finding convergence region for beta={beta:.3f}.")
    #         model.reset()
    #         model.beta = beta
    #         num_iters, s_inits = find_convergence_region(model, x_prev, dt)
    #         num_iters_over_beta.append(num_iters)
    #         s_inits_over_beta.append(s_inits)

    #     num_iters_per_beta = jnp.stack(num_iters_over_beta)
    #     s_inits_over_beta = jnp.stack(s_inits_over_beta)

    #     return num_iters_per_beta, s_inits_over_beta, betas


def check_newton_convergence(
    model: BuckleyLeverettModelProtocol,
    beta: float,
    q_init: jnp.ndarray,
    dt: float,
    q_prev: jnp.ndarray,
) -> int:
    """Check if Newton's method converges for a given model and initial value.

    Logging is suppressed during the convergence check to avoid overhead from
    formatting JAX arrays in log messages.

    Parameters:
        model: The model to solve.
        beta:
        q_init: Initial guess for Newton's method.
        dt: Time step size.
        q_prev: Previous time step solution.

    Returns:
        i: Number of iterations taken to converge, -1 if not converged.

    """
    # Suppress solver logging to avoid expensive JAX array __repr__ calls.
    solver_logger = logging.getLogger("hc_sandbox.buckley_leverett.solvers")
    prev_level = solver_logger.level
    solver_logger.setLevel(logging.WARNING)
    try:
        _, converged, i = newton(
            model, q_init, q_prev, dt=dt, progressbars=False, beta=beta
        )
    except ValueError:
        converged = False
    finally:
        solver_logger.setLevel(prev_level)

    if not converged:
        i = -1
    return i
