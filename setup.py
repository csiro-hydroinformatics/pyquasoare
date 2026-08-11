#!/usr/bin/env python

import numpy

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

# Build directives. Aggressive means
# faster, but less safe compilation
AGGRESSIVE = True

directives = {
   # Bounds/overflow/safety checks removed
   "boundscheck": not AGGRESSIVE,
   "wraparound": not AGGRESSIVE,
   "initializedcheck": not AGGRESSIVE,
   "nonecheck": not AGGRESSIVE,
   "cdivision": AGGRESSIVE,
   "cdivision_warnings": not AGGRESSIVE,

   # Python-interaction overhead removed
   "profile": not AGGRESSIVE,
   "linetrace": not AGGRESSIVE,
   "embedsignature": not AGGRESSIVE,

   # Aggressive numeric/control-flow behavior
   "overflowcheck": not AGGRESSIVE,
   "overflowcheck.fold": not AGGRESSIVE,

   # Compilation mode
   "language_level": "3",
   "binding": False,

   # Loop/iteration optimizations
   "infer_types": True
   }

# Customize compilation to take into account aggressive options
class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        is_msvc = self.compiler.compiler_type == "msvc"
        if AGGRESSIVE:
            if is_msvc:
                ext.extra_compile_args += ["/O2"]  # MSVC has no /O3
            else:
                # march=native : tells GCC/Clang to detect the CPU architecture
                # of the machine doing the compiling and generate machine
                # code that uses every instruction set extension that CPU supports
                # — rather than targeting a generic, portable baseline.
                ext.extra_compile_args += ["-O3", "-march=native"]

        super().build_extension(ext)


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
    ext_modules=cythonize(extensions,
                          compiler_directives=directives),
    cmdclass={"build_ext": CustomBuildExt}
)


