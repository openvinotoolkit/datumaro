---
title: 'Installation'
linkTitle: 'Installation'
description: ''
weight: 1
---

### Dependencies

- Python (3.6+)
- Optional: OpenVINO, TensorFlow, PyTorch, MxNet, Caffe, Accuracy Checker

### Installation steps

Optionally, set up a virtual environment:

``` bash
python -m pip install virtualenv
python -m virtualenv venv
. venv/bin/activate
```

Install:
``` bash
# From PyPI:
pip install datumaro[default]

# From the GitHub repository:
pip install 'git+https://github.com/openvinotoolkit/datumaro[default]'
```

Read more about choosing between `datumaro` and `datumaro[default]`
[here](#core-install).

#### Plugins

Datumaro has many plugins, which are responsible for dataset formats,
model launchers and other optional components. If a plugin has dependencies,
they can require additional installation. You can find the list of all the
plugin dependencies in the [plugins](/docs/user-manual/extending) section.

#### Customizing installation

- <a id="core-install"></a>Datumaro has the following installation options:
  - `pip install datumaro` - for core library functionality
  - `pip install datumaro[default]` - for normal CLI experience

  In restricted installation environments, where some dependencies are
  not available, or if you need only the core library functionality,
  you can install Datumaro without extra plugins.

  In some cases, installing just the core library may be not enough,
  because there can be limited options of installing graphical libraries
  in the system (various Docker environments, servers etc). You can select
  between using `opencv-python` and `opencv-python-headless` by setting the
  `DATUMARO_HEADLESS` environment variable to `0` or `1` before installing
  the package. It requires installation from sources (using `--no-binary`):
  ```bash
  DATUMARO_HEADLESS=1 pip install datumaro --no-binary=datumaro
  ```
  This option can't be covered by extras due to Python packaging system
  limitations.

- Although Datumaro excludes `pycocotools` of version 2.0.2 in
  requirements, it works with this version perfectly fine. The
  reason for such requirement is binary incompatibility of the `numpy`
  dependency in the `TensorFlow` and `pycocotools` binary packages,
  and the current workaround forces this package to be build from sources
  on most platforms
  (see [#253](https://github.com/openvinotoolkit/datumaro/issues/253)).
  If you need to use 2.0.2, make sure it is linked with the same version
  of `numpy` as `TensorFlow` by reinstalling the package:
  ``` bash
  pip uninstall pycocotools
  pip install pycocotools --no-binary=pycocotools
  ```

- When installing directly from the repository, you can change the
  installation branch with `...@<branch_name>`. Also use `--force-reinstall`
  parameter in this case. It can be useful for testing of unreleased
  versions from GitHub pull requests.
