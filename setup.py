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
        "src/pyquasoare/c_quasoare_steady.c"
        ],
        include_dirs=[numpy.get_include()]),
    Extension(name="c_pyquasoare_grad",
        sources=[
        "src/pyquasoare/c_pyquasoare_grad.pyx",
        "src/pyquasoare/c_quasoare_grad.c",
        "src/pyquasoare/c_quasoare_utils.c"
        ],
        include_dirs=[numpy.get_include()]),
    Extension(name="c_pyquasoare_benchmarks",
        sources=[
        "src/pyquasoare/benchmarks/c_pyquasoare_benchmarks.pyx",
        "src/pyquasoare/benchmarks/c_nonlinrouting.c",
        "src/pyquasoare/benchmarks/c_gr4jprod.c",
        "src/pyquasoare/benchmarks/c_gradbench_linrouting.c",
        "src/pyquasoare/benchmarks/c_gradbench_linrouting_d.c",
        "src/pyquasoare/benchmarks/c_gradbench_quadrouting.c",
        "src/pyquasoare/benchmarks/c_gradbench_quadrouting_d.c",
        "src/pyquasoare/benchmarks/c_gradbench_rainfall_runoff.c",
        "src/pyquasoare/benchmarks/c_gradbench_rainfall_runoff_d.c",
        "src/pyquasoare/benchmarks/c_gradbench_semidist.c",
        "src/pyquasoare/benchmarks/c_gradbench_semidist_d.c"
        ],
        include_dirs=[numpy.get_include()])
]

setup(
    name="pyquasoare",
    ext_modules=cythonize(extensions, \
            compiler_directives={"language_level": 3, "profile": False})
)


