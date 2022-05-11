from distutils.core import setup
from Cython.Build import cythonize

setup(
    # has to indicate each module one by one
    ext_modules=cythonize("compass/scheduler.py"), 
)