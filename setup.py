from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="kraken.ConfPruneIdx",
        sources=["kraken/ConfPruneIdx.pyx"],
        language="c",
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level=3)
)
