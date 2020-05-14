from distutils.core import setup, Extension
import os

from Cython.Build import cythonize
import numpy as np

PACKAGE_DIR = "rPTMDetermine"

c_rptmdetermine = Extension(
    "crPTMDetermine",
    sources=[
        os.path.join(PACKAGE_DIR, "crPTMDetermine", "converters.cpp"),
        os.path.join(PACKAGE_DIR, "crPTMDetermine", "annotate.cpp"),
        os.path.join(PACKAGE_DIR, "crPTMDetermine", "crPTMDetermine.cpp"),
    ],
    extra_compile_args=['-std=c++11']
)

setup(
    name="rPTMDetermine",
    version="2.0.0a",
    packages=[
        "rPTMDetermine",
        "rPTMDetermine.config",
        "rPTMDetermine.crPTMDetermine",
        "rPTMDetermine.machinelearning",
        "rPTMDetermine.plotting",
        "rPTMDetermine.readers",
        "rPTMDetermine.spectra_readers"
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
            "language_level": "3",
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
    scripts=[
        "bin/rptmdetermine_validate.py",
        "bin/rptmdetermine_retrieve.py"
    ],
    include_package_data=True,
    language="c++"
)
