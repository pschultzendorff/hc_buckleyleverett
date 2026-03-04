from setuptools import find_packages, setup

setup(
    name="hc_buckleyleverett",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["jax", "matplotlib", "PyQt6", "scikit-image", "seaborn", "tqdm"],
    extras_require={
        "dev": ["mypy", "pytest", "ruff", "types-tqdm", "types-seaborn", "scipy-stubs"],
    },
    author="Peter von Schultzendorff",
    author_email="peter.schultzendorff@uib.no",
    description="Homotopy continuation methods for nonlinear problems with convergence and curvature analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
