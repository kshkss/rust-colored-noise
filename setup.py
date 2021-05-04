#!/bin/env python 

import sys

from setuptools import setup
from setuptools_rust import RustExtension

setup(rust_extensions=[RustExtension("perlin_noise.rust_ext")],
      packages=["perlin_noise"],
      include_package_data=True)

