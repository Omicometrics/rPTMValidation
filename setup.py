from distutils.core import setup, Extension
import os

from Cython.Build import cythonize
import numpy as np

PACKAGE_DIR = "rPTMDetermine"

c_rptmdetermine = Extension(
    "crPTMDetermine",
    sources=[
        os.path.join(PACKAGE_DIR, "converters.cpp"),
        os.path.join(PACKAGE_DIR, "annotate.cpp"),
        os.path.join(PACKAGE_DIR, "crPTMDetermine.cpp"),
    ],
    extra_compile_args=['-std=c++11']
)

setup(
    name="rPTMDetermine",
    version="1.0.0",
    packages=[
        "rPTMDetermine",
        "rPTMDetermine.readers",
    ],
    ext_modules=cythonize(
        [
            c_rptmdetermine,
            os.path.join(PACKAGE_DIR, "binomial.pyx"),
            os.path.join(PACKAGE_DIR, "ionscore.pyx"),
        ],
        include_path=[
            PACKAGE_DIR
        ],
        compiler_directives={
            "language_level" : "3",
        }
    ),
    include_dirs=[np.get_include()],
    package_data={
        "rPTMDetermine": [
            "binomial.pxd",
            "EnzymeRules.json",
        ],
        "rPTMDetermine.readers": [
            "unimod.xml",
        ]
    },
    include_package_data=True,
    language="c++"
)
