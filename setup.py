import codecs
import os
import re

from setuptools import find_packages, setup

def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *args)


def get_contents(*args):
    """Get the contents of a file relative to the source distribution directory."""
    with codecs.open(get_absolute_path(*args), "r", "UTF-8") as handle:
        return handle.read()


def get_version(*args):
    """Extract the version number from a Python module."""
    contents = get_contents(*args)
    metadata = dict(re.findall("__([a-z]+)__ = ['\"]([^'\"]+)", contents))
    return metadata["version"]


def get_install_requires():
    install_requires = ["numpy", "torch"]
    requirements_txt = parse_requirements("requirements.txt")
    icheck_pretracker_found = False
    return install_requires


setup(
    name="authmmcls",
    version=get_version("authmmcls", "__init__.py"),
    author="dmitrii.shaulskii",
    description="inference model for classification",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=get_install_requires(),
    python_requires=">=3.8.0",
)
