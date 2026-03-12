# hc_buckleyleverett

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18863360.svg)](https://doi.org/10.5281/zenodo.18863360)
[![arXiv](https://img.shields.io/badge/arXiv-2603.10730-b31b1b.svg)](https://arxiv.org/abs/2603.10730)

Companion code for *Efficient design of continuation methods for hyperbolic transport*
*problems in porous media*.

A collection of modules and run-scripts for analysing and designing homotopy
continuation (HC) methods applied to the Buckley-Leverett problem. Includes:
- An implicit finite-volume discretisation of the 1D Buckley-Leverett problem
- Newton and HC solvers
- Three homotopy strategies (diffusion-based, convex-hull flux, linear relative
  permeability)
- Homotopy curve traceability analysis via curvature and Newton convergence evaluation

## Installation & Reproducing Results

**Local setup (requires Python 3.12+):**
```bash
git clone https://github.com/pschultzendorff/hc_buckleyleverett -b reproducable
cd hc_buckleyleverett
pip install -e .
python hc_buckleyleverett/scripts/buckley_leverett/viscous.py
```

**Docker:**
```bash
git clone https://github.com/pschultzendorff/hc_buckleyleverett -b reproducable
cd hc_buckleyleverett
docker-compose up
```
Results are saved to `hc_buckleyleverett/results/viscous/`.

## References

If you use this code, please cite:

> von Schultzendorff, P.; Both, J.W.; Nordbotten, J.M.; Sandve, T.H.
> *Efficient design of continuation methods for hyperbolic transport problems in porous*
> *media.*
> https://arxiv.org/abs/2603.10730

### Used references

- Brown, D.A. and Zingg, D.W. (2017). *Design and evaluation of homotopies for*
  *efficient and robust continuation.* Applied Numerical Mathematics,
  118, 150-181. https://doi.org/10.1016/j.apnum.2017.03.001
- Brown, D.A. and Zingg, D.W. (2017). *Efficient numerical differentiation of*
  *implicitly-defined curves for sparse systems.* Journal of Computational and Applied
  Mathematics, 304, 138–159. https://doi.org/10.1016/j.cam.2016.03.002
- Jiang, J., & Tchelepi, H. A. (2018). *Dissipation-based continuation method for*
  *multiphase flow in heterogeneous porous media.* Journal of Computational Physics,
  375, 307–336. https://doi.org/10.1016/j.jcp.2018.08.044
- von Schultzendorff, P.; Both, J.W.; Nordbotten, J.M.; Sandve, T.H.; Vohralík, M. 
  *Adaptive homotopy continuation solver for incompressible two-phase flow in porous*
  *media* (in preparation).

## Dependencies

jax, matplotlib, seaborn, scikit-image, tqdm, PyQt6

Dev: pytest, mypy, ruff

## AI disclosure

Generative AI (GitHub Copilot in VS Code with Claude Opus 4.6) was used to assist with 
speeding up ``hc_analysis.hessian_tensor`` and
``hc_analysis.HCAnalysisMixin.convergence_metric`` as well as writing documentation,
tests, and type hinting.
