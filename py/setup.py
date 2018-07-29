from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

# Get pybind11 include.
pybind_include = pybind11.get_include(True)

ext_modules=[
    Extension(
        'gplib',
        [pybind_include, 'bindings/bindings.cpp'],
        language='c++'
    )
]