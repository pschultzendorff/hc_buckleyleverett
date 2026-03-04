import logging
import pathlib
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
from hc_sandbox.buckley_leverett.hc import (
    ConHullHCMixin,
    DiffusionHCMixin,
    LinearRelPermHCMixin,
)
from hc_sandbox.buckley_leverett.hc_analysis import HCAnalysisMixin
from hc_sandbox.buckley_leverett.model import BuckleyLeverettModel
from hc_sandbox.buckley_leverett.viz import solve_and_plot

logging.basicConfig(level=logging.INFO)

results_dir = pathlib.Path(__file__).parent.parent.parent / "results" / "viscous"
results_dir.mkdir(exist_ok=True, parents=True)


# Model params are taken from Jiang & Tchelepi (2018), section 7.
# - :math:`\Omega = [0,1]`
# - :math:`u_t = 0.01`
# - :math:`k_{r,\mathrm{w}} = S^2, k_{r,\mathrm{n}} = (1-S)^2`
# - :math:`N = 500`
# - :math:`\Delta t \approx 5`
# - Appleyard damping is applied

NUM_CELLS: int = 100

# NOTE By default, total_flow = porosity = 1.0.
model_params: dict[str, Any] = {
    "num_cells": NUM_CELLS,
    "domain_size": 1.0,
    # Rel. perm. model parameters:
    "rp_model": "Corey",
    "nw": 2.0,
    "nn": 2.0,
    # Permeability, which is ignored since gravity is zero:
    "permeability": 1.0,
    # Buoyancy parameters; NO GRAVITY:
    "rho_w": 1.0,
    "rho_n": 1.0,
    "G": 0.0,
    "theta": 0.0,
}


# Mixin the analysis functionality into the HC model classes.
class LinearRelPermHCAnalysis(
    HCAnalysisMixin, LinearRelPermHCMixin, BuckleyLeverettModel
): ...


class ConHullHCAnalysis(HCAnalysisMixin, ConHullHCMixin, BuckleyLeverettModel): ...


class DiffusionHCAnalysis(HCAnalysisMixin, DiffusionHCMixin, BuckleyLeverettModel): ...


def compare_solvers(model_params: dict[str, Any], run_case: str) -> None:
    case_dir = results_dir / run_case
    case_dir.mkdir(exist_ok=True)

    # Determine the convex hull side and rotation based on the structure of the Riemann problem.
    match run_case:
        case "case1a_sL00_sR10_sin00_M1":
            hull_side = "lower"
            rotate_solution = False
        case "case1b_sL10_sR00_sin10_M1":
            hull_side = "upper"
            rotate_solution = True
        case "case1c_sL08_sR02_sin1_M1":
            hull_side = "upper"
            rotate_solution = True
        case "case1d_sL02_sR08_sin00_M1":
            hull_side = "lower"
            rotate_solution = False
        case "case1e_sL05_sR05_sin1_M1":
            hull_side = "upper"
            rotate_solution = True

        case "case2a_sL00_sR10_sin00_M10":
            hull_side = "lower"
            rotate_solution = False
        case "case2b_sL10_sR00_sin10_M10":
            hull_side = "upper"
            rotate_solution = True
        case "case2c_sL08_sR02_sin10_M10":
            hull_side = "upper"
            rotate_solution = True
        case "case2d_sL02_sR08_sin00_M10":
            hull_side = "lower"
            rotate_solution = False
        case "case2e_sL05_sR05_sin1_M10":
            hull_side = "upper"
            rotate_solution = True

        case "case3a_sL00_sR10_sin00_M01":
            hull_side = "lower"
            rotate_solution = False
        case "case3b_sL10_sR00_sin10_M01":
            hull_side = "upper"
            rotate_solution = True
        case "case3c_sL08_sR02_sin1_M01":
            hull_side = "upper"
            rotate_solution = True
        case "case3d_sL02_sR08_sin00_M01":
            hull_side = "lower"
            rotate_solution = False
        case "case3e_sL05_sR05_sin1_M01":
            hull_side = "upper"
            rotate_solution = True
        case _:
            raise ValueError(f"Unknown case: {run_case}")

    hc_lin = LinearRelPermHCAnalysis(model_params)  # type: ignore
    hc_conv = ConHullHCAnalysis(model_params, con_hull_side=hull_side)  # type: ignore
    hc_diff1 = DiffusionHCAnalysis(model_params, omega=1e-5)  # type: ignore
    hc_diff2 = DiffusionHCAnalysis(model_params, omega=2e-3)  # type: ignore
    hc_diff3 = DiffusionHCAnalysis(model_params, omega=1e-1)  # type: ignore

    convex_hull_fig = hc_conv.plot_con_hull(
        color_f="black",
        color_hull="xkcd:sky blue",
        color_linear="xkcd:magenta",
        label_fontsize=20,
        tick_fontsize=14,
    )
    convex_hull_fig.savefig(case_dir / "flux_functions.png", bbox_inches="tight")

    FINAL_TIME: float = 25.0

    case_dir_final_time = case_dir / f"T_{FINAL_TIME}"
    case_dir_final_time.mkdir(exist_ok=True)

    curvature_fig, _ = plt.subplots(1, 1, figsize=(10, 6))
    curvature_lambda_fig, _ = plt.subplots(1, 1, figsize=(10, 6))
    convergence_metric_fig, _ = plt.subplots(1, 1, figsize=(10, 6))

    for model, color, ls, linewidth in zip(
        [hc_lin, hc_conv, hc_diff1, hc_diff2, hc_diff3],
        [
            "xkcd:magenta",
            "xkcd:sky blue",
            "xkcd:orange",
            "xkcd:light orange",
            "xkcd:amber",
        ],
        ["-", "--", ":", "-.", (0, (3, 5, 1, 5))],
        [4.0, 4.0, 2.5, 4.0, 2.5],
    ):
        model_name = model.__class__.__name__[:-8]
        if isinstance(model, DiffusionHCAnalysis):
            label = r"$\mathbf{H}_\mathrm{d}$" + rf" $\omega= {model.omega:.2e}$"
            file_name = model_name + f"_omega_{model.omega}"
        elif isinstance(model, ConHullHCAnalysis):
            label = r"$\mathbf{H}_\mathrm{c}$"
            file_name = model_name
        else:
            label = r"$\mathbf{H}_\mathrm{lin}$"
            file_name = model_name

        (
            solution_fig,
            residual_fig,
            curvature_fig,
            curvature_lambda_fig,
            convergence_metric_fig,
        ) = solve_and_plot(
            model,  # type: ignore
            final_time=FINAL_TIME,
            curvature_fig=curvature_fig,
            curvature_lambda_fig=curvature_lambda_fig,
            convergence_metric_fig=convergence_metric_fig,
            solver_kwargs={
                "appleyard": True,
                "progressbars": False,
            },
            plotting_kwargs={
                "color": color,
                "linestyle": ls,
                "linewidth": linewidth,
                "label": label,
                "label_fontsize": 20,
                "tick_fontsize": 14,
            },
        )

        if solution_fig is not None and residual_fig is not None:
            # Rotate the solution figure if needed (when left saturation > right saturation).
            if rotate_solution:
                ax = solution_fig.axes[0]
                ax.view_init(elev=30, azim=-45 + 90)  # type: ignore  # Axes3D method

                # Adjust the z-label position after rotation to avoid it being cut off.
                ax.set_zlabel("$S$", labelpad=-2)  # type: ignore # Axed3D method
                solution_fig.tight_layout()

            solution_fig.savefig(
                case_dir_final_time / f"solution_curve_{file_name}.png",
                bbox_inches="tight",
            )
            residual_fig.savefig(
                case_dir_final_time / f"residual_curve_{file_name}.png",
                bbox_inches="tight",
            )

    curvature_fig.savefig(  # type: ignore
        case_dir_final_time / "curvature.png",
        bbox_inches="tight",
    )
    curvature_lambda_fig.savefig(  # type: ignore
        case_dir_final_time / "curvature_lambda.png",
        bbox_inches="tight",
    )
    convergence_metric_fig.savefig(  # type: ignore
        case_dir_final_time / "convergence_metric.png",
        bbox_inches="tight",
    )


# Case 1a: s_L = 0.0, s_R = 1.0, s_inlet = 0.0, M = mu_w/mu_n = 1.0
model_params.update(
    {
        "s_inlet": 0.0,
        "s_initial": jnp.array([0.0] * (NUM_CELLS // 2) + [1.0] * (NUM_CELLS // 2)),
        "mu_w": 1.0,
        "mu_n": 1.0,
    }
)
compare_solvers(model_params, "case1a_sL00_sR10_sin00_M1")

# Case 1b: s_L = 1.0, s_R = 0.0, s_inlet = 1.0, M = mu_w/mu_n = 1.0
model_params.update(
    {
        "s_inlet": 1.0,
        "s_initial": jnp.array([1.0] * (NUM_CELLS // 2) + [0.0] * (NUM_CELLS // 2)),
        "mu_w": 1.0,
        "mu_n": 1.0,
    }
)
# compare_solvers(model_params, "case1b_sL10_sR00_sin10_M1")

# Case 1c: s_L = 0.8, s_R = 0.2, s_inlet = 1.0, M = mu_w/mu_n = 1.0
model_params.update(
    {
        "s_inlet": 0.8,
        "s_initial": jnp.array([0.8] * (NUM_CELLS // 2) + [0.2] * (NUM_CELLS // 2)),
        "mu_w": 1.0,
        "mu_n": 1.0,
    }
)
# compare_solvers(model_params, "case1c_sL08_sR02_sin1_M1")

# Case 1d: s_L = 0.2, s_R = 0.8, s_inlet = 0.0, M = mu_w/mu_n = 1.0
model_params.update(
    {
        "s_inlet": 0.2,
        "s_initial": jnp.array([0.2] * (NUM_CELLS // 2) + [0.8] * (NUM_CELLS // 2)),
        "mu_w": 1.0,
        "mu_n": 1.0,
    }
)
# compare_solvers(model_params, "case1d_sL02_sR08_sin00_M1")

# Case 1e: s_L = 0.5, s_R = 0.5, s_inlet = 1.0, M = mu_w/mu_n = 1.0
model_params.update(
    {
        "s_inlet": 1.0,
        "s_initial": jnp.array([0.5] * NUM_CELLS),
        "mu_w": 1.0,
        "mu_n": 1.0,
    }
)
# compare_solvers(model_params, "case1e_sL05_sR05_sin1_M1")


# Case 2a: s_L = 0.0, s_R = 1.0, s_inlet = 0.0, M = mu_w/mu_n = 10.0
model_params.update(
    {
        "s_inlet": 0.0,
        "s_initial": jnp.array([0.0] * (NUM_CELLS // 2) + [1.0] * (NUM_CELLS // 2)),
        "mu_w": 10.0,
        "mu_n": 1.0,
    }
)
compare_solvers(model_params, "case2a_sL00_sR10_sin00_M10")

# Case 2b: s_L = 1.0, s_R = 0.0, s_inlet = 1.0, M = mu_w/mu_n = 10.0
model_params.update(
    {
        "s_inlet": 1.0,
        "s_initial": jnp.array([1.0] * (NUM_CELLS // 2) + [0.0] * (NUM_CELLS // 2)),
        "mu_w": 10.0,
        "mu_n": 1.0,
    }
)
compare_solvers(model_params, "case2b_sL10_sR00_sin10_M10")

# Case 2c: s_L = 0.8, s_R = 0.2, s_inlet = 1.0, M = mu_w/mu_n = 10.0
model_params.update(
    {
        "s_inlet": 0.8,
        "s_initial": jnp.array([0.8] * (NUM_CELLS // 2) + [0.2] * (NUM_CELLS // 2)),
        "mu_w": 10.0,
        "mu_n": 1.0,
    }
)
compare_solvers(model_params, "case2c_sL08_sR02_sin10_M10")

# Case 2d: s_L = 0.2, s_R = 0.8, s_inlet = 0.0, M = mu_w/mu_n = 10.0
model_params.update(
    {
        "s_inlet": 0.2,
        "s_initial": jnp.array([0.2] * (NUM_CELLS // 2) + [0.8] * (NUM_CELLS // 2)),
        "mu_w": 10.0,
        "mu_n": 1.0,
    }
)
# compare_solvers(model_params, "case2d_sL02_sR08_sin00_M10")

# Case 2e: s_L = 0.5, s_R = 0.5, s_inlet = 1.0, M = mu_w/mu_n = 10.0
model_params.update(
    {
        "s_inlet": 1.0,
        "s_initial": jnp.array([0.5] * NUM_CELLS),
        "mu_w": 10.0,
        "mu_n": 1.0,
    }
)
# compare_solvers(model_params, "case2e_sL05_sR05_sin1_M10")

# Case 3a: s_L = 0.0, s_R = 1.0, s_inlet = 0.0, M = mu_w/mu_n = 0.1
model_params.update(
    {
        "s_inlet": 0.0,
        "s_initial": jnp.array([0.0] * (NUM_CELLS // 2) + [1.0] * (NUM_CELLS // 2)),
        "mu_w": 1.0,
        "mu_n": 10.0,
    }
)
# compare_solvers(model_params, "case3a_sL00_sR10_sin00_M01")

# Case 3b: s_L = 1.0, s_R = 0.0, s_inlet = 1.0, M = mu_w/mu_n = 0.1
model_params.update(
    {
        "s_inlet": 1.0,
        "s_initial": jnp.array([1.0] * (NUM_CELLS // 2) + [0.0] * (NUM_CELLS // 2)),
        "mu_w": 1.0,
        "mu_n": 10.0,
    }
)
# compare_solvers(model_params, "case3b_sL10_sR00_sin10_M01")

# Case 3c: s_L = 0.8, s_R = 0.2, s_inlet = 1.0, M = mu_w/mu_n = 0.1
model_params.update(
    {
        "s_inlet": 0.8,
        "s_initial": jnp.array([0.8] * (NUM_CELLS // 2) + [0.2] * (NUM_CELLS // 2)),
        "mu_w": 1.0,
        "mu_n": 10.0,
    }
)
# compare_solvers(model_params, "case3c_sL08_sR02_sin1_M01")

# Case 3d: s_L = 0.2, s_R = 0.8, s_inlet = 0.0, M = mu_w/mu_n = 0.1
model_params.update(
    {
        "s_inlet": 0.2,
        "s_initial": jnp.array([0.2] * (NUM_CELLS // 2) + [0.8] * (NUM_CELLS // 2)),
        "mu_w": 1.0,
        "mu_n": 10.0,
    }
)
# compare_solvers(model_params, "case3d_sL02_sR08_sin00_M01")

# Case 3e: s_L = 0.5, s_R = 0.5, s_inlet = 1.0, M = mu_w/mu_n = 0.1
model_params.update(
    {
        "s_inlet": 1.0,
        "s_initial": jnp.array([0.5] * NUM_CELLS),
        "mu_w": 1.0,
        "mu_n": 10.0,
    }
)
# compare_solvers(model_params, "case3e_sL05_sR05_sin1_M01")
