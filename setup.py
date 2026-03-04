from setuptools import find_packages, setup

setup(
    name="hc_buckleyleverett",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax==0.9.0.1",
        "matplotlib==3.10.8",
        "PyQt6==6.10.2",
        "scikit-image==0.26.0",
        "seaborn==0.13.2",
        "tqdm==4.67.3",
    ],
    extras_require={
        "dev": [
            "mypy==1.19.1",
            "pytest==9.0.2",
            "ruff==0.15.2",
            "types-tqdm==4.67.3.20260205",
            "types-seaborn==0.13.2.20251221",
            "scipy-stubs==1.17.1.0",
        ],
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
    python_requires="==3.12.12",
)
