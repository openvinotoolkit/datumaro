---
title: 'MNIST'
linkTitle: 'MNIST'
description: ''
weight: 7
---

## Format specification

MNIST format specification is available [here](http://yann.lecun.com/exdb/mnist/).
Fashion MNIST format specification is available [here](https://github.com/zalandoresearch/fashion-mnist).
MNIST in CSV  format specification is available [here](https://pjreddie.com/projects/mnist-in-csv/).

The dataset has few data formats available. Datumaro supports the
binary (Python pickle) format and the CSV variant. Each data format is covered
by a separate Datumaro format.

Supported formats:
- Binary (Python pickle) - `mnist`
- CSV - `mnist_csv`

Supported annotation types:
- `Label`

The format only supports single channel 28 x 28 images.

## Load MNIST dataset

The MNIST dataset is available for free download:

- [train-images-idx3-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz):
  training set images
- [train-labels-idx1-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz):
  training set labels
- [t10k-images-idx3-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz):
  test set images
- [t10k-labels-idx1-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz):
  test set labels

The Fashion MNIST dataset is available for free download:

- [train-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz):
  training set images
- [train-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz):
  training set labels
- [t10k-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz):
  test set images
- [t10k-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz):
  test set labels

The MNIST in CSV dataset is available for free download:

- [mnist_train.csv](https://pjreddie.com/media/files/mnist_train.csv)
- [mnist_test.csv](https://pjreddie.com/media/files/mnist_test.csv)

There are two ways to create Datumaro project and add MNIST dataset to it:

``` bash
datum import --format mnist --input-path <path/to/dataset>
# or
datum create
datum add path -f mnist <path/to/dataset>
```

There are two ways to create Datumaro project and add MNIST in CSV dataset
to it:

``` bash
datum import --format mnist_csv --input-path <path/to/dataset>
# or
datum create
datum add path -f mnist_csv <path/to/dataset>
```

It is possible to specify project name and project directory run
`datum create --help` for more information.

MNIST dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── labels.txt # list of non-digit labels (optional)
    ├── t10k-images-idx3-ubyte.gz
    ├── t10k-labels-idx1-ubyte.gz
    ├── train-images-idx3-ubyte.gz
    └── train-labels-idx1-ubyte.gz
```

MNIST in CSV dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── labels.txt # list of non-digit labels (optional)
    ├── mnist_test.csv
    └── mnist_train.csv
```

If the dataset needs non-digit labels, you need to add the `labels.txt`
to the dataset folder. For example, `labels.txt` for Fashion MNIST the
following contents:

<!--lint disable fenced-code-flag-->
```
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
```

## Export to other formats

Datumaro can convert MNIST dataset into any other format [Datumaro supports](/docs/user-manual/supported-formats/).
To get the expected result, convert the dataset to formats
that support the classification task (e.g. CIFAR-10/100, ImageNet, PascalVOC,
etc.) There are few ways to convert MNIST dataset to other dataset format:

``` bash
datum project import -f mnist -i <path/to/mnist>
datum export -f imagenet -o <path/to/output/dir>
# or
datum convert -if mnist -i <path/to/mnist> -f imagenet -o <path/to/output/dir>
```

These commands also work for MNIST in CSV if you use `mnist_csv` instead of `mnist`.

## Export to MNIST

There are few ways to convert dataset to MNIST format:

``` bash
# export dataset into MNIST format from existing project
datum export -p <path/to/project> -f mnist -o <path/to/export/dir> \
    -- --save-images
# converting to MNIST format from other format
datum convert -if imagenet -i <path/to/imagenet/dataset> \
    -f mnist -o <path/to/export/dir> -- --save-images
```

Extra options for export to MNIST format:

- `--save-images` allow to export dataset with saving images
(by default `False`);
- `--image-ext <IMAGE_EXT>` allow to specify image extension
for exporting dataset (by default `.png`).

These commands also work for MNIST in CSV if you use `mnist_csv` instead of `mnist`.

## Examples

Datumaro supports filtering, transformation, merging etc. for all formats
and for the MNIST format in particular. Follow [user manual](/docs/user-manual/)
to get more information about these operations.

There are few examples of using Datumaro operations to solve
particular problems with MNIST dataset:

### Example 1. How to create custom MNIST-like dataset

```python
from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem

dataset = Dataset.from_iterable([
    DatasetItem(id=0, image=np.ones((28, 28)),
        annotations=[Label(2)]
    ),
    DatasetItem(id=1, image=np.ones((28, 28)),
        annotations=[Label(7)]
    )
], categories=[str(label) for label in range(10)])

dataset.export('./dataset', format='mnist')
```

### Example 2. How to filter and convert MNIST dataset to ImageNet

Convert MNIST dataset to ImageNet format, keep only images with `3` class
presented:

``` bash
# Download MNIST dataset:
# https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
# https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
datum convert --input-format mnist --input-path <path/to/mnist> \
              --output-format imagenet \
              --filter '/item[annotation/label="3"]'
```

Examples of using this format from the code can be found in
the [binary format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_mnist_format.py) and [csv format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_mnist_csv_format.py)
