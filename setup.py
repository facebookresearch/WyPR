# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='wypr',
    version='0.0.0',
    description='weakly-supervised point could recognition framework',
    author='Jason Ren',
    author_email='jasonren@fb.com',
    url='https://github.com/fairinternal/WyPR',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)