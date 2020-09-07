## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Design](#design-and-code-structure)

## Installation

### Prerequisites

- Python (3.5+)

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

## Testing

It is expected that all Datumaro functionality is covered and checked by
unit tests. Tests are placed in `tests/` directory.

To run tests use:

``` bash
python -m unittest discover -s tests
```

If you're working inside of a CVAT environment, you can also use:

``` bash
python manage.py test datumaro/
```

## Design and code structure

- [Design document](docs/design.md)
- [Developer guide](docs/developer_guide.md)

## Code style

Try to be readable and consistent with the existing codebase.
The project mostly follows PEP8 with little differences.
Continuation lines have a standard indentation step by default,
or any other, if it improves readability. For long conditionals use 2 steps.
No trailing whitespaces, 80 characters per line.

## Environment

The recommended editor is VS Code with the Python plugin.