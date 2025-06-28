import numpy as np
from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="neural-stpp",
    version="0.1.0",
    description="Neural STPP implementation from facebookresearch",
    # ─── THIS is the magic ──────────────────────────────────────────────
    packages=find_packages(),  
    ext_modules=cythonize("neural_stpp/data_utils_fast.pyx"),
    include_dirs=[np.get_include()],
    install_requires=["numpy>=1.18"],
    python_requires=">=3.7",
)
