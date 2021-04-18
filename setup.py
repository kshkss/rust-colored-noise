#!/bin/env python 

import sys

from setuptools import setup
from setuptools_rust import RustExtension

setup(rust_extensions=[RustExtension("colored_noise.rust_ext")],
      packages=["colored_noise"],
      include_package_data=True)

