#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def calculate_version():
    initpy = open('pmlb/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='pmlb',
    version=package_version,
    author='Randal S. Olson',
    author_email='rso@randalolson.com',
    packages=find_packages(),
    url='https://github.com/EpistasisLab/penn-ml-benchmarks',
    license='License :: OSI Approved :: MIT License',
    description=('A Python wrapper for the Penn Machine Learning Benchmark data repository.'),
    long_description='''
A Python wrapper for the Penn Machine Learning Benchmark data repository.

Contact
=============
If you have any questions or comments about the Penn Machine Learning Benchmark, please feel free to contact us via e-mail: rso@randalolson.com

This project is hosted at https://github.com/EpistasisLab/penn-ml-benchmarks
''',
    zip_safe=True,
    install_requires=['pandas'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Utilities'
    ],
    keywords=['data mining', 'benchmark', 'machine learning', 'data analysis', 'data sets', 'data science', 'wrapper'],
)
