#!/usr/bin/env python

# Authors: Santi Villalba <sdvillal@gmail.com>
# Licence: BSD 3 clause

from setuptools import setup, find_packages

setup(
    name='abbert2',
    version='1.0.0dev0',
    packages=find_packages(),
    url='',
    license='BSD 3 clause',
    author='Santi Villalba',
    author_email='sdvillal@gmail.com',
    description='Antibody-specific language models',

    # at the moment deps are managed solely via conda
    install_requires=[],
    tests_require=[],
    extras_require={},

    entry_points={
        'console_scripts': [
            'oas = abbert2.oas.cli:main'
        ]
    },


    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    platforms=['Any'],
)
