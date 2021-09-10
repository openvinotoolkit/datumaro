---
title: 'CIFAR'
linkTitle: 'CIFAR'
description: ''
weight: 1
---

## Format specification

CIFAR format specification is available [here](https://www.cs.toronto.edu/~kriz/cifar.html).

Supported annotation types:
- `Label`

Datumaro supports Python version CIFAR-10/100.
The difference between CIFAR-10 and CIFAR-100 is how labels are stored
in the meta files (`batches.meta` or `meta`) and in the annotation files.
The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image
comes with a "fine" label (the class to which it belongs) and a "coarse" label
(the superclass to which it belongs). In CIFAR-10 there are no superclasses.

CIFAR formats contains 32 x 32 images. As an extension, Datumaro supports
reading and writing of arbitrary-sized images.

## Load CIFAR dataset

The CIFAR dataset is available for free download:

- [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz):
  CIFAR-10 python version
- [cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz):
  CIFAR-100 python version

A Datumaro project with a CIFAR source can be created the following way:

``` bash
datum create
datum add --format cifar <path/to/dataset>
```

It is possible to specify project name and project directory run
`datum create --help` for more information.

CIFAR-10 dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── batches.meta
    ├── <subset_name1>
    ├── <subset_name2>
    └── ...
```

CIFAR-100 dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── meta
    ├── <subset_name1>
    ├── <subset_name2>
    └── ...
```

Dataset files use [Pickle](https://docs.python.org/3/library/pickle.html)
data format.

Meta files:

<!--lint disable fenced-code-flag-->
```
CIFAR-10:
    num_cases_per_batch: 1000
    label_names: list of strings (['airplane', 'automobile', 'bird', ...])
    num_vis: 3072

CIFAR-100:
    fine_label_names: list of strings (['apple', 'aquarium_fish', ...])
    coarse_label_names: list of strings (['aquatic_mammals', 'fish', ...])
```

Annotation files:

<!--lint disable fenced-code-flag-->
```
Common:
    'batch_label': 'training batch 1 of <N>'
    'data': numpy.ndarray of uint8, layout N x C x H x W
    'filenames': list of strings

    If images have non-default size (32x32) (Datumaro extension):
        'image_sizes': list of (H, W) tuples

CIFAR-10:
    'labels': list of strings

CIFAR-100:
    'fine_labels': list of integers
    'coarse_labels': list of integers
```

## Export to other formats

Datumaro can convert CIFAR dataset into any other format [Datumaro supports](/docs/user-manual/supported-formats).
To get the expected result, convert the dataset to formats
that support the classification task (e.g. MNIST, ImageNet, PascalVOC,
etc.) There are few ways to convert CIFAR dataset to other dataset format:

``` bash
datum create
datum add -f cifar <path/to/cifar>
datum export -f imagenet -o <path/to/output/dir>
# or
datum convert -if cifar -i <path/to/cifar> -f imagenet -o <path/to/output/dir>
```

## Export to CIFAR

There are few ways to convert dataset to CIFAR format:

``` bash
# export dataset into CIFAR format from existing project
datum export -p <path/to/project> -f cifar -o <path/to/export/dir> \
    -- --save-images
# converting to CIFAR format from other format
datum convert -if imagenet -i <path/to/imagenet/dataset> \
    -f cifar -o <path/to/export/dir> -- --save-images
```

Extra options for export to CIFAR format:

- `--save-images` allow to export dataset with saving images
(by default `False`);
- `--image-ext <IMAGE_EXT>` allow to specify image extension
for exporting dataset (by default `.png`).

The format (CIFAR-10 or CIFAR-100) in which the dataset will be
exported depends on the presence of superclasses in the `LabelCategories`.

## Examples

Datumaro supports filtering, transformation, merging etc. for all formats
and for the CIFAR format in particular. Follow [user manual](/docs/user-manual)
to get more information about these operations.

There are few examples of using Datumaro operations to solve
particular problems with CIFAR dataset:

### Example 1. How to create custom CIFAR-like dataset

```python
from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem

dataset = Dataset.from_iterable([
    DatasetItem(id=0, image=np.ones((32, 32, 3)),
        annotations=[Label(3)]
    ),
    DatasetItem(id=1, image=np.ones((32, 32, 3)),
        annotations=[Label(8)]
    )
], categories=['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck'])

dataset.export('./dataset', format='cifar')
```

### Example 2. How to filter and convert CIFAR dataset to ImageNet

Convert CIFAR dataset to ImageNet format, keep only images with `dog` class
presented:

``` bash
# Download CIFAR-10 dataset:
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
datum convert --input-format cifar --input-path <path/to/cifar> \
              --output-format imagenet \
              --filter '/item[annotation/label="dog"]'
```

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_cifar_format.py)
