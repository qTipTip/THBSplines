import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

ext_modules = [
    Extension("BSpline", ["THBSplines/src_c/BSpline.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name='THBSplines',
    version='',
    packages=['THBSplines'],
    url='',
    license='',
    author='Ivar Stangeby',
    author_email='',
    description='',
    zip_safe=False,
    ext_modules=cythonize(ext_modules, annotate=True, install_requires=['matplotlib'])
)
