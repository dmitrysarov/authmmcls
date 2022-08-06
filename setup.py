import codecs
import os
import re

from setuptools import find_packages, setup


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


def get_install_requires():
    requirements_txt = parse_requirements("requirements.txt")
    return requirements_txt


setup(
    name="authmmcls",
    author="dmitrii.shaulskii",
    description="inference model for classification",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=get_install_requires(),
    python_requires=">=3.8.0",
)
