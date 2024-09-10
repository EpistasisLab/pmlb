#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def calculate_version():
    initpy = open('pmlb/_version.py').read().split('\n')
    version = list(filter(
        lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='pmlb',
    version=package_version,
    author='Randal S. Olson, William La Cava, Trang Le, Weixuan Fu',
    author_email=('rso@randalolson.com, lacava@upenn.edu, '
            'ttle@pennmedicine.upenn.edu, weixuanf@pennmedicine.upenn.edu'),
    packages=find_packages(),
    package_data={'pmlb': ['*.tsv']},
    include_package_data=True,
    url='https://github.com/EpistasisLab/pmlb',
    license='License :: OSI Approved :: MIT License',
    description=('A Python wrapper for the Penn Machine Learning Benchmark '
        'data repository.'),
    long_description='''
A Python wrapper for the Penn Machine Learning Benchmark data repository.

Contact
=============
If you have any questions or comments about the Penn Machine Learning Benchmark, 
please feel free to contact us via e-mail: ttle@pennmedicine.upenn.edu

This project is hosted at https://github.com/EpistasisLab/pmlb
''',
    zip_safe=True,
    install_requires=['pandas>=1.0.5',
                    'requests>=2.24.0',
                    'pyyaml>=5.3.1',
                    'scikit-learn>=0.19.0'
                    ],
    extras_require={
        'dev': ['nose', 'numpy', 'scipy', 'tabulate', 'parameterized',
        'matplotlib', 'seaborn', 'ydata-profiling'],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Utilities'
    ],
    keywords=['data mining', 'benchmark', 'machine learning', 'data analysis', 'data sets', 'data science', 'wrapper'],
)
