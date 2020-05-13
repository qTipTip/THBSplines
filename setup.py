import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='THBSplines',
    version='',
    packages=['THBSplines', 'THBSplines.src', 'THBSplines.src_c'],
    url='',
    license='',
    author='Ivar Stangeby',
    author_email='',
    description='',
    ext_modules=cythonize('**/*.pyx', annotate=True, build_dir="THBSplines/src_c/build"),
    include_dirs = [numpy.get_include()]
)

