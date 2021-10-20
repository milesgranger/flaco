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

assert flaco_lib_file.is_file(), "flaco lib not built; run 'cargo build --release'"


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
    "Cython",
    "wheel",
    "pytest-runner",
    "pytest-benchmark~=3.0",
    "pytest",
    "sqlalchemy",
    "psycopg2-binary<2.9.0",
    "memory-profiler==0.58.0",
    "hypothesis",
    "pandas",
]

setup(
    name="flaco",
    version="0.3.0",
    author="Miles Granger",
    author_email="miles59923@gmail.com",
    description="Fast and Efficient PostgreSQL data into numpy/pandas",
    license="MIT",
    keywords="pandas postgres postgresql numpy rust python",
    url="https://github.com/milesgranger/flaco",
    long_description=pathlib.Path("README.md").read_text(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: The Unlicense (Unlicense)",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Rust",
        "Programming Language :: Cython",
    ],
    test_suite="tests",
    tests_require=dev_requirements,
    extras_require={"dev": dev_requirements},
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy>=1.0.0"],
    ext_modules=cythonize(extension),
    include_dirs=[np.get_include(), "flaco"],
    zip_safe=False,
)
