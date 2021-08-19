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
pip install datumaro

# From the GitHub repository:
pip install 'git+https://github.com/openvinotoolkit/datumaro'
```

> You can change the installation branch with `...@<branch_name>`
> Also use `--force-reinstall` parameter in this case.
