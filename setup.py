from distutils.core import setup
import os

from Cython.Build import cythonize
import numpy as np

setup(
    name="rPTMDetermine",
    version="0.1",
    packages=[
        "rPTMDetermine",
        "rPTMDetermine.readers",
    ],
    ext_modules=cythonize(
        [
            os.path.join("rPTMDetermine", "annotate.pyx"),
            os.path.join("rPTMDetermine", "binomial.pyx"),
            os.path.join("rPTMDetermine", "ionscore.pyx"),
        ],
        include_path=[
            "rPTMDetermine"
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
    include_package_data=True
)