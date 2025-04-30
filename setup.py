# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="neural-stpp",               # ← must match #egg=neural-stpp
    version="0.1.0",                  # ← pick a version
    description="Neural STPP implementation from facebookresearch",
    packages=find_packages(),         # ← auto-discovers the neural_stpp package
    ext_modules=cythonize("data_utils_fast.pyx"),
    include_dirs=[np.get_include()],
    install_requires=[
        "numpy>=1.18",                # ensure runtime NumPy
    ],
    python_requires=">=3.7",
)
