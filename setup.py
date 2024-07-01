#!/usr/bin/env python

import os
import numpy
from pathlib import Path

from setuptools import setup, Extension, find_packages

from Cython.Distutils import build_ext

import versioneer

def read(fname):
    thisfolder = Path(__file__).resolve().parent
    with (thisfolder / fname).open() as fo:
        return fo.read()

# C extensions
ext_rezeq=Extension(name="c_pyrezeq",
    sources=[
        "src/pyrezeq/c_pyrezeq.pyx",
        "src/pyrezeq/c_utils.c",
        "src/pyrezeq/c_integ.c",
        "src/pyrezeq/c_run.c",
        "src/pyrezeq/c_steady.c",
        "src/pyrezeq/c_quadrouting.c"
    ],
    extra_cflags=["-O3"],
    extra_compile_args=["-ffast-math"],
    include_dirs=[numpy.get_include()])

cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = build_ext

# Package config
setup(
    name="pyrezeq",
    author= "Julien Lerat",
    author_email= "julien.lerat@csiro.au",
    url= "https://github.com/csiro-hydroinformatics/pyrezeq",
    download_url= "hhttps://github.com/csiro-hydroinformatics/pyrezeq/tags",
    version=versioneer.get_version(),
    description= "Solve the reservoir equation",
    long_description= read("README.md"),
    packages=find_packages(),
    package_dir={"": "src"},
    package_data={
        "pyrezeq": [
            "tests/*.zip"
        ],
    },
    install_requires= [
        "cython",
        "hydrodiy",
        "numpy >= 1.8.0",
        "scipy (>=0.14.0)",
        "pandas >= 0.16"
    ],
    cmdclass=cmdclass,
    ext_modules=[ext_rezeq],
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License"
    ]
)


