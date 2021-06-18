
# Copyright (C) 2019-2020 Intel Corporation
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

def get_requirements():
    requirements = [
        'attrs>=19.3.0',
        'defusedxml',
        'GitPython',
        'lxml',
        'matplotlib',
        'numpy>=1.17.3',
        'Pillow',

        # Avoid 2.0.2 Linux binary distribution because of
        # a conflict in numpy versions with TensorFlow:
        # - TF is compiled with numpy 1.19 ABI
        # - pycocotools is compiled with numpy 1.20 ABI
        # Using a previous version allows to force package rebuilding.
        #
        # https://github.com/openvinotoolkit/datumaro/issues/253
        'pycocotools!=2.0.2; platform_system != "Windows"',
        'pycocotools-windows; platform_system == "Windows"',

        'PyYAML',
        'scikit-image',
        'tensorboardX',
    ]
    if strtobool(os.getenv('DATUMARO_HEADLESS', '0').lower()):
        requirements.append('opencv-python-headless')
    else:
        requirements.append('opencv-python')

    return requirements

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.dist.Distribution().fetch_build_eggs([
    'Cython>=0.27.3' # required for pycocotools and others, if need to compile
])

setuptools.setup(
    name="datumaro",
    version=find_version(here),
    author="Intel",
    author_email="maxim.zhiltsov@intel.com",
    description="Dataset Management Framework (Datumaro)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openvinotoolkit/datumaro",
    packages=setuptools.find_packages(exclude=['tests*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=get_requirements(),
    extras_require={
        'tf': ['tensorflow'],
        'tf-gpu': ['tensorflow-gpu'],
    },
    entry_points={
        'console_scripts': [
            'datum=datumaro.cli.__main__:main',
        ],
    },
)
