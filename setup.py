import os
import pathlib
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

if os.getenv("RUNNER_OS", "").lower() == "windows":
    p = pathlib.Path(__file__).parent.joinpath("target").joinpath("release").joinpath("flaco.lib")
    assert p.is_file(), "Rust lib not built!"
    extra_link_args = [f"/link", f"LIBPATH:D:\\a\\flaco\\flaco\\target\\release\\flaco.lib"]  # .lib on MSVC .a on MinGW
    extra_compile_args = []
else:
    extra_link_args = ["-l:libflaco.a"]
    extra_compile_args = ["-fopenmp", "-O3"]

extension = Extension(
    name="*",
    sources=[str(pathlib.Path("flaco/*.pyx"))],
    libraries=["flaco"],
    include_dirs=[np.get_include(), "flaco"],
    library_dirs=[str(pathlib.Path("target/release"))],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c"
)

dev_requirements = [
    "pytest",
    "docker",
    "sqlalchemy",
    "psycopg2",
    "hypothesis",
    "pandas",
]

setup(
    name="flaco",
    version="0.1.0.post2",
    test_suite="tests",
    tests_require=dev_requirements,
    extras_require={"dev": dev_requirements},
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy"],
    ext_modules=cythonize(extension),
    include_dirs=[np.get_include(), "flaco"],
    zip_safe=False,
)
