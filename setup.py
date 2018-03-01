#!/usr/bin/env python3

import os

from setuptools import setup, find_packages

root_dir = os.path.dirname(__file__)

setup(
    name="nvme_sampler",
    version="0.0.2",
    author="Paweł Wiejacha",
    author_email="pawel.wiejacha@rtbhouse.com",
    description="Library for sampling batches of rows from a binary dataset file.",
    packages=find_packages(exclude=["build"]),
    setup_requires=["cffi>=1.0.0"],
    install_requires=["cffi>=1.0.0"],
    cffi_modules=[
        os.path.join(root_dir, "build.py:ffi")
    ]
)
