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
ext_quasoare=Extension(name="c_pyquasoare",
    sources=[
        "src/pyquasoare/c_pyquasoare.pyx",
        "src/pyquasoare/c_quasoare_utils.c",
        "src/pyquasoare/c_quasoare_steady.c",
        "src/pyquasoare/c_quasoare_quad.c",
        "src/pyquasoare/c_nonlinrouting.c",
        "src/pyquasoare/c_gr4jprod.c"
    ],
    extra_cflags=["-O3"],
    extra_compile_args=["-ffast-math"],
    include_dirs=[numpy.get_include()])

cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = build_ext

# Package config
setup(
    name="pyquasoare",
    author= "Julien Lerat",
    author_email= "julien.lerat@csiro.au",
    url= "https://github.com/csiro-hydroinformatics/pyquasoare",
    download_url= "hhttps://github.com/csiro-hydroinformatics/pyquasoare/tags",
    version=versioneer.get_version(),
    description= "Solve the reservoir equation",
    long_description= read("README.md"),
    packages=find_packages(),
    package_dir={"": "src"},
    package_data={
        "pyquasoare": [
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
    ext_modules=[ext_quasoare],
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License"
    ]
)


