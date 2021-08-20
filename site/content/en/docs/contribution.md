---
title: 'Contributing Guide'
linkTitle: 'Contributing'
description: ''
weight: 50
---

## Related sections:

- [Design document](/docs/design/)
- [Developer manual](/docs/developer-manual/)

## Installation

### Prerequisites

- Python (3.6+)

``` bash
git clone https://github.com/openvinotoolkit/datumaro
```

Optionally, install a virtual environment (recommended):

``` bash
python -m pip install virtualenv
python -m virtualenv venv
. venv/bin/activate
```

Then install all dependencies:

``` bash
while read -r p; do pip install $p; done < requirements.txt
```

If you're working inside of a CVAT environment:
``` bash
. .env/bin/activate
while read -r p; do pip install $p; done < datumaro/requirements.txt
```

Install Datumaro:
``` bash
pip install -e /path/to/the/cloned/repo/
```

**Optional dependencies**

These components are only required for plugins and not installed by default:

- OpenVINO
- Accuracy Checker
- TensorFlow
- PyTorch
- MxNet
- Caffe

## Usage

``` bash
datum --help
python -m datumaro --help
python datumaro/ --help
python datum.py --help
```

``` python
import datumaro
```

## Code style

Try to be readable and consistent with the existing codebase.

The project mostly follows PEP8 with little differences.
Continuation lines have a standard indentation step by default,
or any other, if it improves readability. For long conditionals use 2 steps.
No trailing whitespaces, 80 characters per line.

Example:

```python
def do_important_work(parameter1, parameter2, parameter3,
        option1=None, option2=None, option3=None) -> str:
    """
    Optional description. Mandatory for API.
    Use comments for implementation specific information, use docstrings
    to give information to user / developer.

    Returns: status (str) - Possible values: 'done', 'failed'
    """

    ... do stuff ...

    # Use +1 level of indentation for continuation lines
    variable_with_a_long_but_meaningful_name = \
        function_with_a_long_but_meaningful_name(arg1, arg2, arg3,
            kwarg1=value_with_a_long_name, kwarg2=value_with_a_long_name)

    # long conditions, loops, with etc. also use +1 level of indentation
    if condition1 and long_condition2 or \
            not condition3 and condition4 and condition5 or \
            condition6 and condition7:

        ... do other stuff ...

    elif other_conditions:

        ... some other things ...

    # in some cases special formatting can improve code readability
    specific_case_formatting = np.array([
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
    ], dtype=np.int32)

    return status
```

## Environment

The recommended editor is VS Code with the Python language plugin.

## Testing <a id="testing"></a>

It is expected that all Datumaro functionality is covered and checked by
unit tests. Tests are placed in `tests/` directory.
Currently, we use [`pytest`](https://docs.pytest.org/) for testing, but we
also compatible with `unittest`.

To run tests use:

``` bash
pytest -v
# or
python -m pytest -v
```

If you're working inside of a CVAT environment, you can also use:

``` bash
python manage.py test datumaro/
```


### Test cases <a id="Test_case_description"></a>

### Test marking <a id="Test_marking"></a>

For better integration with CI and requirements tracking,
we use special annotations for tests.

A test needs to marked with a requirement it is related to. To mark a test, use:

```python
from unittest import TestCase
from .requirements import Requirements, mark_requirement

class MyTests(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_my_requirement(self):
        ... do stuff ...
```

Such marking will apply markings from the requirement specified.
They can be overridden for a specific test:

```python
import pytest

    @pytest.mark.proirity_low
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_my_requirement(self):
        ... do stuff ...
```

#### Requirements <a id="Requirements"></a>

Requirements and other links need to be added to [`tests/requirements.py`](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/requirements.py):

```python
DATUM_244 = "Add Snyk integration"
DATUM_BUG_219 = "Return format is not uniform"
```

```python
# Fully defined in GitHub issues:
@pytest.mark.reqids(Requirements.DATUM_244, Requirements.DATUM_333)

# And defined any other way:
@pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
```


##### Available annotations for tests and requirements

Markings are defined in [`tests/conftest.py`](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/conftest.py).

**A list of requirements and bugs**
```python
@pytest.mark.requids(Requirements.DATUM_123)
@pytest.mark.bugs(Requirements.DATUM_BUG_456)
```

**A priority**
```python
@pytest.mark.priority_low
@pytest.mark.priority_medium
@pytest.mark.priority_high
```

**Component**
The marking used for indication of different system components

```python
@pytest.mark.components(DatumaroComponent.Datumaro)
```

**Skipping tests**

```python
@pytest.mark.skip(SkipMessages.NOT_IMPLEMENTED)
```

**Parametrized runs**

Parameters are used for running the same test with different parameters e.g.

```python
@pytest.mark.parametrize("numpy_array, batch_size", [
    (np.zeros([2]), 0),
    (np.zeros([2]), 1),
    (np.zeros([2]), 2),
    (np.zeros([2]), 5),
    (np.zeros([5]), 2),
])
```

### Test documentation <a id="TestDoc"></a>

Tests are documented with docs strings. Test descriptions must contain
the following: sections: `Description`, `Expected results` and `Steps`.

```python
def test_can_convert_polygons_to_mask(self):
    """
    <b>Description:</b>
    Ensure that the dataset polygon annotation can be properly converted
    into dataset segmentation mask.

    <b>Expected results:</b>
    Dataset segmentation mask converted from dataset polygon annotation
    is equal to an expected mask.

    <b>Steps:</b>
    1. Prepare dataset with polygon annotation
    2. Prepare dataset with expected mask segmentation mode
    3. Convert source dataset to target, with conversion of annotation
      from polygon to mask.
    4. Verify that resulting segmentation mask is equal to the expected mask.
    """
```
