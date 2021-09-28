import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

extension = Extension(
    name="*",
    sources=["flaco/*.pyx"],
    libraries=["flaco"],
    include_dirs=[np.get_include(), "flaco"],
    library_dirs=["target/release"]
)

setup(
    name="flaco",
    version="0.1.0",
    ext_modules = cythonize(extension),
    include_dirs=[np.get_include(), "flaco"],
    zip_safe=False
)
