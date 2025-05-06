from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension('kraken.ConfPruneIdx', ['kraken/ConfPruneIdx.pyx'])
]

setup(
    name='kraken',
    version='0.1',
    packages=['kraken'],
    ext_modules=cythonize(extensions, language_level=3),
    zip_safe=False,
    entry_points={
        'console_scripts': ['extract_dft_files.py=kraken.cli.extract_dft_files:main',
                            'return_dft_files.py=kraken.cli.return_dft_files:main',
                            'run_kraken_dft.py=kraken.cli.run_kraken_dft:main',
                            'run_kraken_conf_search.py=kraken.cli.run_kraken_conf_search:main',
                            'example_conf_search_submission_script.py=kraken.cli.example_conf_search_submission_script:main',
                            'example_dft_submission_script.py=kraken.cli.example_dft_submission_script:main',
        ]
}
)
