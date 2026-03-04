"""Visualization for homotopy curves of the 1D two-phase flow problem.


References:
- Brown, D.A. and Zingg, D.W. (2017) ‘Design and evaluation of homotopies for
efficient and robust continuation’, Applied Numerical Mathematics, 118, pp. 150–181.
Available at: https://doi.org/10.1016/j.apnum.2017.03.001.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeAlias

import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax.typing import ArrayLike as ArrayLike_jax
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

# Ignore mypy complaining about missing library stubs.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
from numpy.typing import ArrayLike as ArrayLike_np
from scipy.ndimage import map_coordinates

from hc_buckleyleverett.buckley_leverett.solvers import solve

if TYPE_CHECKING:
    from hc_buckleyleverett.buckley_leverett.protocol import (
        HCModelandAnalysisProtocol,
        HCModelProtocol,
    )

try:
    from skimage.measure import marching_cubes
except Exception:
    # Ignore mypy complaining about wrong type for marching_cubes.
    marching_cubes = None  # type: ignore

ArrayLike: TypeAlias = ArrayLike_jax | ArrayLike_np

sns.set_theme(style="whitegrid")
plt.style.use("seaborn-v0_8-colorblind")


def plot_solution(solutions: ArrayLike) -> None:
    """Plot the solution of the Buckley Leverett problem with an interactive time
    slider.

    Parameters:
        solutions: Array-like structure containing the solution data.

    """
    sol = jnp.array(solutions)
    n_time_steps, n_vars = sol.shape
    n_cells = n_vars // 2

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)  # Make room for slider.

    xs = jnp.linspace(0, n_cells, n_cells)  # Evenly spaced cell-centre positions.

    (s_line,) = ax.plot(xs, sol, "v-", color="tab:orange", label="$S$")

    ax.set_ylim(0, 1)

    ax.set_xlabel("x")
    ax.set_ylabel("$S$", color="tab:orange")

    ax.set_title("Solution (time step: 0)")
    ax.legend()
    ax.grid(True)

    # Add slider
    ax_slider = plt.axes((0.15, 0.1, 0.7, 0.03))
    time_slider = Slider(ax_slider, r"$t$", 0, n_time_steps, valinit=0, valstep=1)

    # Update function for slider
    def update(val):
        time_idx = int(time_slider.val)
        s_line.set_ydata(sol[time_idx])
        ax.set_title(f"Solution (time step: {time_idx})")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def weighted_curvature(
    curvature_vectors: ArrayLike, betas: ArrayLike, intermediate_solutions: ArrayLike
) -> jnp.ndarray:
    r"""Compute arclength-weighted curvature as a traceability measure.

    Following Brown & Zingg (2017), the weighted curvature is defined as

    .. math::
        \kappa_w = \|\mathbf{\kappa}\| \, s_{\mathrm{tot}}^2

    where :math:`\|\mathbf{\kappa}\|` is the Euclidean norm of the curvature vector
    and :math:`s_{\mathrm{tot}}` is the total arclength of the homotopy curve.


    Parameters:
        curvature_vectors: Curvature vectors at each point along the homotopy curve.
        betas: Homotopy parameter values at each curve point.
        intermediate_solutions: Solution vectors at each curve point.

    Returns:
        Arclength-weighted curvature at each curve point.

    """
    curvatures = jnp.linalg.norm(jnp.asarray(curvature_vectors), axis=-1)

    # Calculate the total arclength of the homotopy curve.
    segment_lengths = segments_arclengths(betas, intermediate_solutions)
    total_arclength = jnp.sum(segment_lengths, axis=0)

    return curvatures * total_arclength**2


def segments_arclengths(
    betas: ArrayLike, intermediate_solutions: ArrayLike
) -> jnp.ndarray:
    r"""Approximate the arclength along the solution curve.

    The segment arclengths :math:`\Delta s_i` is approximated by the length of the
    polygonal line segments connecting the points along the curve, as in Brown & Zingg
    (2017):

    .. math::
        \Delta s_i \approx \sqrt{\|\mathbf{q}_i - \mathbf{q}_{i - 1}\|^2 + \|\lambda_i - \lambda_{i - 1}\|^2}.

    The cumulative arclength up to each curve point is then given by:

    .. math::
        s_i = \sum_{j=1}^{i} \Delta s_j.

    Parameters:
        betas: Homotopy parameter values at each curve point.
        intermediate_solutions: Solution vectors at each curve point.

    Returns:
        Segment arclengths :math:`\Delta s_i` between consecutive curve points.

    """
    betas = jnp.asarray(betas)
    intermediate_solutions = jnp.asarray(intermediate_solutions)

    # Polygonal line segments connecting consecutive (q_i, beta_i) points.
    curve_segments = (
        jnp.concatenate([intermediate_solutions, betas[:, None]], axis=-1)[1:]
        - jnp.concatenate([intermediate_solutions, betas[:, None]], axis=-1)[:-1]
    )
    segments_lengths = jnp.linalg.norm(curve_segments, axis=-1)

    return segments_lengths


def relative_arclengths(
    betas: ArrayLike, intermediate_solutions: ArrayLike
) -> jnp.ndarray:
    r"""Approximate the relative arclength along the solution curve.

    The total arclength :math:`s_{tot}` is approximated by summing the polygonal line
    segments connecting the points along the curve, as in Brown & Zingg (2017):

    .. math::
        s_{tot} \approx \sum_{i=1}^{n} \Delta s_i

    The relative arclength at each point is then given by the cumulative arclength up to
    that point divided by the total arclength

    .. math::
        s_i = \frac{\sum_{j=1}^{i} \Delta s_j}{s_{tot}}.

    Parameters:
        betas: Homotopy parameter values at each curve point.
        intermediate_solutions: Solution vectors at each curve point.

    Returns:
        Relative (normalised) arclength at each curve point, ranging from 0 to 1.

    """
    betas = jnp.asarray(betas)

    segment_lengths = segments_arclengths(betas, intermediate_solutions)
    total_arclength = jnp.sum(segment_lengths, axis=0)

    relative_arclengths = jnp.zeros_like(betas)
    relative_arclengths = relative_arclengths.at[1:].set(
        jnp.cumsum(segment_lengths) / total_arclength
    )

    return relative_arclengths


def plot_curvature(
    curve_parametrization: ArrayLike,
    curvatures: ArrayLike,
    fig: Optional[Figure] = None,
    **kwargs,
):
    """Plot arclength-weighted curvature.

    Parameters:
        curve_parametrization: Relative arclengths (or a different curve
            parametrization) for the x-axis.
        curvatures: Weighted curvature values to plot.
        fig: Existing figure to draw into. A new one is created when ``None``.
        **kwargs: Accepts ``color``, ``ls``, ``linewidth``, ``marker``, ``label``, and
            other matplotlib plot arguments.

    Returns:
        The figure containing the curvature plot.

    """
    curve_parametrization = jnp.asarray(curve_parametrization)
    curvatures = jnp.asarray(curvatures)

    label_fontsize = kwargs.pop("label_fontsize", 14)
    tick_fontsize = kwargs.pop("tick_fontsize", 10)

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        ax = fig.axes[0]

    ax.plot(curve_parametrization, curvatures, **kwargs)

    ax.set_xlabel(r"$s / s_\mathrm{tot}$", fontsize=label_fontsize)
    ax.set_ylabel(r"$s_\mathrm{tot}^2 \kappa$", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_yscale("log")

    ax.set_ylim(0.01, 20)

    # Light grid for good data visibility.
    ax.grid(True, which="both", linestyle=":", alpha=0.3)

    # Clean up old legends
    for leg in fig.legends:
        leg.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()  # type: ignore[union-attr]

    ax.legend(fontsize=label_fontsize)

    fig.tight_layout()
    return fig


def plot_curvature_lambda(
    curve_parametrization: ArrayLike,
    curvatures: ArrayLike,
    fig: Optional[Figure] = None,
    **kwargs,
):
    """Plot arclength-weighted curvature.

    Parameters:
        curve_parametrization: Relative arclengths (or a different curve
            parametrization) for the x-axis.
        curvatures: Weighted curvature values to plot.
        fig: Existing figure to draw into. A new one is created when ``None``.
        **kwargs: Accepts ``color``, ``ls``, ``linewidth``, ``marker``, ``label``, and
            other matplotlib plot arguments.

    Returns:
        The figure containing the curvature plot.

    """
    curve_parametrization = jnp.asarray(curve_parametrization)
    curvatures = jnp.asarray(curvatures)

    label_fontsize = kwargs.pop("label_fontsize", 14)
    tick_fontsize = kwargs.pop("tick_fontsize", 10)

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        ax = fig.axes[0]

    ax.plot(curve_parametrization, curvatures, **kwargs)

    ax.set_xlabel(r"$\lambda$", fontsize=label_fontsize)
    ax.set_ylabel(r"$\kappa_r$", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_yscale("log")
    ax.set_xlim(1, 0)

    # Light grid for good data visibility.
    ax.grid(True, which="both", linestyle=":", alpha=0.3)

    # Clean up old legends
    for leg in fig.legends:
        leg.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()  # type: ignore[union-attr]

    ax.legend(fontsize=label_fontsize)

    fig.tight_layout()
    return fig


def plot_convergence_metric(
    curve_parametrization: ArrayLike,
    scaled_newton_rs: ArrayLike,
    fig: Optional[Figure] = None,
    **kwargs,
):
    """Plot Newton convergence metric.

    Parameters:
        curve_parametrization: Relative arclengths (or a different curve
            parametrization) for the x-axis.
        scaled_newton_rs: Newton convergence metric to plot.
        fig: Existing figure to draw into. A new one is created when ``None``.
        **kwargs: Accepts ``color``, ``ls``, ``linewidth``, ``marker``, ``label``, and
            other matplotlib plot arguments.

    Returns:
        The figure containing the convergence metric plot.

    """
    curve_parametrization = jnp.asarray(curve_parametrization)
    scaled_newton_rs = jnp.asarray(scaled_newton_rs)

    label_fontsize = kwargs.pop("label_fontsize", 14)
    tick_fontsize = kwargs.pop("tick_fontsize", 10)

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        ax = fig.axes[0]

    ax.plot(curve_parametrization, scaled_newton_rs, **kwargs)

    ax.set_xlabel(r"$s / s_\mathrm{tot}$", fontsize=label_fontsize)
    ax.set_ylabel(r"$\tilde{r}(s)$", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    # Format y-axis ticks to use decimal notation instead of scientific notation.
    ax.ticklabel_format(style="plain", axis="y")

    # Light grid for good data visibility.
    ax.grid(True, which="both", linestyle=":", alpha=0.3)

    # Clean up old legends
    for leg in fig.legends:
        leg.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()  # type: ignore[union-attr]

    ax.legend(fontsize=label_fontsize)

    fig.tight_layout()
    return fig


def plot_solution_curve(
    solutions: ArrayLike, betas: ArrayLike, model: HCModelProtocol, **kwargs
) -> Figure:
    r"""Plot saturation along the homotopy path as a 3-D surface.

    The x-axis shows the homotopy parameter :math:`\lambda`, the y-axis shows spatial
    position, and the z-axis shows the water saturation :math:`S`.

    Parameters:
        solutions: Solution vectors at each homotopy step.
        betas: Homotopy parameter values at each step.
        model: Model providing domain geometry (``domain_size``, ``num_cells``).

    Returns:
        The figure containing the 3-D surface plot.

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={"projection": "3d"})

    solutions = jnp.asarray(solutions)

    label_fontsize = kwargs.pop("label_fontsize", 12)
    tick_fontsize = kwargs.pop("tick_fontsize", 10)

    X = jnp.asarray(betas)
    Y = jnp.linspace(0, model.domain_size, model.num_cells)
    X, Y = jnp.meshgrid(X, Y)

    # Saturation plot.
    surf2 = ax.plot_surface(  # type: ignore  # Axes3D methods not in matplotlib stubs
        X,
        Y,
        solutions.swapaxes(0, 1),
        cmap="plasma",
        edgecolor="k",
        linewidth=0.2,
    )
    ax.set_xlabel(r"$\lambda$", fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel("$x$", fontsize=label_fontsize, labelpad=10)
    ax.set_zlabel("$S$", fontsize=label_fontsize, labelpad=10)  # type: ignore  # Axes3D method
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.tick_params(axis="z", labelsize=tick_fontsize)  # type: ignore  # Axes3D method
    ax.set_xlim(1, 0)
    ax.view_init(elev=30, azim=-45)  # type: ignore  # Axes3D method
    fig.colorbar(surf2, ax=ax, shrink=0.6, aspect=10, pad=0.075)

    fig.tight_layout()
    return fig


def plot_residual_curve(
    solutions: ArrayLike,
    betas: ArrayLike,
    model: HCModelProtocol,
    dt: float,
    q_prev: Optional[jnp.ndarray] = None,
    **kwargs,
) -> Figure:
    r"""Plot cell-wise transport residuals along the homotopy path as a 3-D surface.

    Residuals are evaluated at :math:`\lambda = 0` (the target problem) for every
    intermediate solution along the homotopy curve.

    Parameters:
        solutions: Solution vectors at each homotopy step.
        betas: Homotopy parameter values at each step.
        model: Model providing residual evaluation and domain geometry.
        dt: Time-step size used for computing the residual.
        q_prev: Solution at the previous time step.

    Returns:
        The figure containing the 3-D residual surface plot.

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={"projection": "3d"})

    solutions = jnp.asarray(solutions)

    label_fontsize = kwargs.pop("label_fontsize", 12)
    tick_fontsize = kwargs.pop("tick_fontsize", 10)

    # Evaluate the residual of the *original* system (beta=0) at each HC solution.
    residuals = jnp.abs(
        jnp.asarray(
            [
                model.residual(solution, dt, q_prev=q_prev, beta=0.0)
                for solution in solutions
            ]
        )
    )

    X = jnp.asarray(betas)
    Y = jnp.linspace(0, model.domain_size, model.num_cells)
    X, Y = jnp.meshgrid(X, Y)

    # Transport residuals.
    surf = ax.plot_surface(  # type: ignore  # Axes3D methods not in matplotlib stubs
        X,
        Y,
        residuals.swapaxes(0, 1),
        cmap="plasma",
        edgecolor="k",
        linewidth=0.2,
    )
    ax.set_xlabel(r"$\lambda$", fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel("$x$", fontsize=label_fontsize, labelpad=10)
    ax.set_zlabel(r"$\mathcal{R}$", fontsize=label_fontsize, labelpad=10)  # type: ignore  # Axes3D method
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.tick_params(axis="z", labelsize=tick_fontsize)  # type: ignore  # Axes3D method
    ax.set_xlim(1, 0)
    ax.view_init(elev=30, azim=-45)  # type: ignore  # Axes3D method
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.075)

    fig.tight_layout()
    return fig


def solve_and_plot(
    model: HCModelandAnalysisProtocol,
    final_time: float,
    curvature_fig: Optional[Figure] = None,
    curvature_lambda_fig: Optional[Figure] = None,
    convergence_metric_fig: Optional[Figure] = None,
    solver_kwargs: Optional[dict] = None,
    plotting_kwargs: Optional[dict] = None,
) -> tuple[
    Optional[Figure],
    Optional[Figure],
    Optional[Figure],
    Optional[Figure],
    Optional[Figure],
]:
    """Solve with homotopy continuation and plot diagnostics.

    Runs one time step of the HC solver, then produces:

    - a 3-D solution-curve figure,
    - a 3-D residual-curve figure,
    - an arclength-weighted curvature figure, and
    - a Newton convergence metric figure.

    Parameters:
        model: HC model with analysis capabilities.
        final_time: Simulation end time (a single time step of this size is taken).
        curvature_fig: Existing curvature figure to draw into.  A new one is
            created when ``None``.
        curvature_lambda_fig: Existing curvature figure for lambda-parametrization to
            draw into. A new one is created when ``None``.
        convergence_metric_fig: Existing convergence metric figure to draw into.
            A new one is created when ``None``.
        solver_kwargs: Keyword arguments to pass to :func:`solve`.
        plotting_kwargs: Keyword arguments to pass to :func:`plot_curvature` and
            :func:`plot_convergence_metric`.

    Returns:
        ``(solution_curve_fig, residual_curve_fig, curvature_fig, convergence_metric_fig)``.
        The first two entries are ``None`` when the solver converges in a single step.

    """
    solver_kwargs = solver_kwargs or {}
    plotting_kwargs = plotting_kwargs or {}

    _, converged = solve(
        model, final_time=final_time, num_time_steps=1, **solver_kwargs
    )
    intermediate_solutions = jnp.asarray(model.intermediate_solutions)

    # Print statistics about the solution path.
    if converged:
        rel_distance = jnp.linalg.norm(
            intermediate_solutions[-1] - intermediate_solutions[0]
        ) / jnp.linalg.norm(intermediate_solutions[-1])
        print(
            f"{model.__class__.__name__}: Relative distance between solutions:"
            + f" {rel_distance}"
        )

    # Plotting:
    if len(model.betas) > 1:
        solution_curve_fig = plot_solution_curve(
            intermediate_solutions, model.betas, model
        )
        residual_curve_fig = plot_residual_curve(
            intermediate_solutions, model.betas, model, final_time, model.s_initial
        )

        arclengths = relative_arclengths(model.betas, intermediate_solutions)
        curvatures = weighted_curvature(
            jnp.asarray(model.curvature_vectors), model.betas, intermediate_solutions
        )
        curvatures_lambda = jnp.linalg.norm(
            jnp.asarray(model.curvature_lambda_vectors), axis=-1
        )

        scaled_newton_rs = jnp.asarray(model.newton_rs) / jnp.asarray(model.betas)

        curvature_fig = plot_curvature(
            arclengths,
            curvatures,
            fig=curvature_fig,
            **plotting_kwargs,
        )
        curvature_lambda_fig = plot_curvature_lambda(
            model.betas,
            curvatures_lambda,
            fig=curvature_lambda_fig,
            **plotting_kwargs,
        )

        convergence_metric_fig = plot_convergence_metric(
            arclengths,
            scaled_newton_rs,
            fig=convergence_metric_fig,
            **plotting_kwargs,
        )

        return (
            solution_curve_fig,
            residual_curve_fig,
            curvature_fig,
            curvature_lambda_fig,
            convergence_metric_fig,
        )

    # If only the initial or none HC step converged, skip plotting.
    else:
        return None, None, curvature_fig, curvature_lambda_fig, convergence_metric_fig


def plot_convergence_tube_with_path(
    num_iters_over_beta: jnp.ndarray,
    solutions_over_beta: jnp.ndarray,  # (B2, 2): [s_left, s_right]
    s_initial: jnp.ndarray | None = None,
    **kwargs,
) -> Figure:
    r"""Render a 3D convergence tube colored by iterations, with solution path.

    Parameters:
        num_iters_over_beta ``shape=(B1, N^2)``: Number of iterations to converge for
            each (s_left, s_right) initial condition at each beta value. -1 indicates
            non-convergence. Assumed to be over a homogeneous N×NxB1 grid in
            :math:`(s_left,s_right,\lambda)`.
        solutions_over_beta ``shape=(B2, 2)``: Exact HC solutions
            :math:`(s_left, s_right)` at each :math:`\lambda` value. Assumed to be over a
            homogeneous B2 grid in :math:`(\lambda)`.
        s_initial ``shape=(2,)``: Initial saturation. Default is None.

    Raises:
        ValueError: If the solutions have more than 2 dimensions.

    """
    # Convert inputs to NumPy.
    it = np.asarray(num_iters_over_beta)  # (B1, N^2), -1 = non-converged
    sol = np.asarray(solutions_over_beta)  # (B2, 2)

    if sol.shape[1] != 2:
        raise ValueError(
            "solutions_over_beta must have shape (B2, 2) for (s_left, s_right)."
        )

    B1, N2 = it.shape
    B2 = sol.shape[0]

    N = int(np.sqrt(N2))
    if N * N != N2:
        raise ValueError("Per-β data must form an N×N grid.")
    if sol.shape != (0,) and sol.shape != (B2, 2):
        raise ValueError("solutions_over_beta must have shape (B2, 2).")

    # Reshape flat per-beta data into a 3-D (s_left, beta, s_right) grid.
    iters = it.reshape(B1, N, N).transpose(1, 0, 2)
    iters_masked = np.where(iters >= 0, iters, np.nan)

    # Binary convergence indicator: 1 where Newton converged, 0 otherwise.
    binary = np.isfinite(iters_masked).astype(float)

    # Pad the volume with zeros so marching cubes produces a closed surface
    # even when the convergence region touches the domain boundary.
    pad = ((1, 1), (1, 1), (1, 1))
    binary_padded = np.pad(binary, pad, constant_values=0)
    iters_padded = np.pad(iters_masked, pad, constant_values=np.nan)

    # Grid spacing: maps integer indices onto the unit cube [0, 1]^3.
    dx = 1.0 / (N - 1)
    dy = 1.0 / (B1 - 1)
    dz = 1.0 / (N - 1)

    # Figure/axes
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Extract the smooth boundary of the convergence region.  Marching cubes
    # finds the level=0.5 isosurface of the binary field, producing a smooth
    # triangulated surface instead of blocky voxels.
    if marching_cubes is None:
        raise ImportError(
            "scikit-image is required for marching_cubes. "
            "Install it with: pip install scikit-image"
        )
    verts, faces, _, _ = marching_cubes(binary_padded, level=0.5, spacing=(dx, dy, dz))

    # Undo padding offset (only affects geometry, not beta meaning).
    verts[:, 0] -= dx
    verts[:, 1] -= dy
    verts[:, 2] -= dz

    # Sample iteration counts at surface vertices for colouring.
    # ``map_coordinates`` interpolates on the integer-indexed padded grid;
    # we convert physical coordinates back to indices and add 1 for the padding.
    ix = verts[:, 0] / dx
    iy = verts[:, 1] / dy
    iz = verts[:, 2] / dz
    coords = np.vstack([ix + 1, iy + 1, iz + 1])  # +1 to offset for the padding layer.

    iter_vals = map_coordinates(iters_padded, coords, order=0, mode="nearest")

    # Colour scale: 1 (best case) to max_iter Newton iterations.
    norm = colors.LogNorm(vmin=1, vmax=kwargs.get("max_iter", 20))
    cmap = plt.colormaps["viridis"]
    face_colors = cmap(norm(iter_vals))[faces].mean(axis=1)

    # Make faces semi-transparent so the interior solution path is visible.
    face_colors[:, 3] = 0.3

    mesh = Poly3DCollection(verts[faces], facecolor=face_colors, edgecolor="none")
    ax.add_collection3d(mesh)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, pad=0.1, label="Newton iterations")

    # Overlay the exact HC solution path as a black line.
    if sol.shape != (0,):
        b2 = np.linspace(1, 0, B2)
        ax.plot(sol[:, 0], b2, sol[:, 1], color="black", linewidth=2)
        ax.scatter(sol[:, 0], b2, zs=sol[:, 1], color="black", s=14)  # type: ignore[ArgumentType]

    # Mark the initial condition at both ends of the homotopy curve.
    if s_initial is not None:
        init = np.asarray(s_initial)
        ax.scatter(init[0], 0.0, init[1], s=500, c="red", marker="o")
        ax.scatter(init[0], 1.0, init[1], s=500, c="red", marker="o")

    # Axis limits with a small margin for visual clarity.
    eps = 0.05
    ax.set_xlabel(r"$s_\mathrm{left}$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_zlabel(r"$s_\mathrm{right}$")
    ax.set_xlim(-eps, 1.0 + eps)
    ax.set_ylim(1.0 + eps, -eps)
    ax.set_zlim(-eps, 1.0 + eps)
    ax.view_init(elev=25, azim=-30)

    fig.tight_layout()
    return fig
