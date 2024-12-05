#!/usr/bin/env python

import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(name="c_pyquasoare",
        sources=[
        "src/pyquasoare/c_pyquasoare.pyx",
        "src/pyquasoare/c_quasoare_utils.c",
        "src/pyquasoare/c_quasoare_core.c",
        "src/pyquasoare/c_nonlinrouting.c",
        "src/pyquasoare/c_gr4jprod.c"
        ],
        include_dirs=[numpy.get_include()])
]

setup(
    name="pyquasoare",
    ext_modules=cythonize(extensions, \
            compiler_directives={"language_level": 3, "profile": False})
)


