import os
import pathlib
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


flaco_lib_dir = pathlib.Path(__file__).parent.joinpath("target").joinpath("release")

libraries = ["flaco"]
if os.getenv("RUNNER_OS", "").lower() == "windows":
    libraries.extend(["ntdll", "ws2_32", "bcrypt", "advapi32"])
    flaco_lib_file = flaco_lib_dir.joinpath("flaco.lib")
    extra_link_args = []
    extra_compile_args = [
        "/link",
        "/SUBSYSTEM:WINDOWS",
    ]
else:
    flaco_lib_file = flaco_lib_dir.joinpath("libflaco.a")
    extra_link_args = ["-l:libflaco.a"]
    extra_compile_args = ["-fopenmp", "-O3"]

assert (
    flaco_lib_file.is_file()
), "flaco lib not build; run 'cargo build --release' first."


extension = Extension(
    name="*",
    sources=[str(pathlib.Path("flaco/*.pyx"))],
    libraries=libraries,
    include_dirs=[np.get_include()],
    library_dirs=[str(flaco_lib_dir)],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c",
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
    install_requires=["numpy>=1.0.0"],
    ext_modules=cythonize(extension),
    include_dirs=[np.get_include(), "flaco"],
    zip_safe=False,
)
