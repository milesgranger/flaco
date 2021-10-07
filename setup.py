import pathlib
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

extension = Extension(
    name="*",
    sources=["flaco/*.pyx"],
    libraries=["flaco"],
    include_dirs=[np.get_include(), "flaco"],
    library_dirs=[
        str(pathlib.Path("target/release")),
    ],
    extra_compile_args=["-fopenmp", "-O3"],
    extra_link_args=["-l:libflaco.a"]
)

setup(
    name="flaco",
    version="0.2.0",
    test_suite="tests",
    tests_require=[
        "pytest",
        "docker",
        "sqlalchemy",
        "psycopg2",
        "hypothesis",
        "pandas",
    ],
    cmdclass = {"build_ext": build_ext},
    install_requires=["numpy"],
    ext_modules=cythonize(extension),
    include_dirs=[np.get_include(), "flaco"],
    zip_safe=False,
)
