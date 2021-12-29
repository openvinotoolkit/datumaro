
# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from distutils.util import strtobool
import os
import os.path as osp
import re

import setuptools

# Snyk scan integration
here = None


def find_version(project_dir=None):
    if not project_dir:
        project_dir = osp.dirname(osp.abspath(__file__))

    file_path = osp.join(project_dir, 'datumaro', 'version.py')

    with open(file_path, 'r') as version_file:
        version_text = version_file.read()

    # PEP440:
    # https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
    pep_regex = r'([1-9]\d*!)?(0|[1-9]\d*)(\.(0|[1-9]\d*))*((a|b|rc)(0|[1-9]\d*))?(\.post(0|[1-9]\d*))?(\.dev(0|[1-9]\d*))?'
    version_regex = r'VERSION\s*=\s*.(' + pep_regex + ').'
    match = re.match(version_regex, version_text)
    if not match:
        raise RuntimeError("Failed to find version string in '%s'" % file_path)

    version = version_text[match.start(1) : match.end(1)]
    return version

CORE_REQUIREMENTS_FILE = 'requirements-core.txt'
DEFAULT_REQUIREMENTS_FILE = 'requirements-default.txt'

def parse_requirements(filename=CORE_REQUIREMENTS_FILE):
    with open(filename) as fh:
        return fh.readlines()

CORE_REQUIREMENTS = parse_requirements(CORE_REQUIREMENTS_FILE)
if strtobool(os.getenv('DATUMARO_HEADLESS', '0').lower()):
    CORE_REQUIREMENTS.append('opencv-python-headless')
else:
    CORE_REQUIREMENTS.append('opencv-python')

DEFAULT_REQUIREMENTS = parse_requirements(DEFAULT_REQUIREMENTS_FILE)

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="datumaro",
    version=find_version(here),
    author="Intel",
    author_email="maxim.zhiltsov@intel.com",
    description="Dataset Management Framework (Datumaro)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openvinotoolkit/datumaro",
    packages=setuptools.find_packages(include=['datumaro*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        'tf': ['tensorflow'],
        'tfds': ['tensorflow-datasets'],
        'tf-gpu': ['tensorflow-gpu'],
        'default': DEFAULT_REQUIREMENTS,
    },
    entry_points={
        'console_scripts': [
            'datum=datumaro.cli.__main__:main',
        ],
    },
    include_package_data=True,
)
