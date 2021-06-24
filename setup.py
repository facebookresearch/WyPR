# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='wypr',
    version='0.1.0',
    description='weakly-supervised point could recognition framework',
    author='JR',
    author_email='renzhzh@gmail.com',
    url='https://github.com/facebookresearch/WyPR',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)