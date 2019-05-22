from distutils.core import setup

from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["rPTMDetermine/binomial.pyx", "rPTMDetermine/ionscore.pyx"]),
    include_dirs=[np.get_include()]
)