# CIFAR user manual

## Contents

- [Format specification](#format-specification)
- [Load CIFAR dataset](#load-CIFAR-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export to CIFAR](#export-to-CIFAR)
- [Particular use cases](#particular-use-cases)

## Format specification

CIFAR format specification available [here](https://www.cs.toronto.edu/~kriz/cifar.html).

CIFAR dataset format supports `Labels` annotations.

Datumaro supports Python version CIFAR-10/100.

## Load CIFAR dataset

The CIFAR dataset is available for free download:

- [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz):
  CIFAR-10 python version
- [cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz):
  CIFAR-100 python version

There are two ways to create Datumaro project and add CIFAR dataset to it:

``` bash
datum import --format cifar --input-path <path/to/dataset>
# or
datum create
datum add path -f cifar <path/to/dataset>
```

It is possible to specify project name and project directory run
`datum create --help` for more information.

CIFAR-10 dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    └── test_batch
```

CIFAR-100 dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── meta
    ├── test
    └── train
```

CIFAR format only supports 32 x 32 images.

The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image
comes with a "fine" label (the class to which it belongs) and a "coarse" label
(the superclass to which it belongs)

The difference between CIFAR-10 and CIFAR-100 is how labels are stored
in the meta file (batches.meta or meta) and in the annotation file (train,
data_batch_1, test_batch, etc.).
<!--lint disable fenced-code-flag-->
```
meta file:
CIFAR-10: num_cases_per_batch: 1000
          label_names: ['airplane', 'automobile', 'bird', ...]
          num_vis: 3072
CIFAR-100: fine_label_names: ['apple', 'aquarium_fish', 'baby', ...]
           coarse_label_names: ['aquatic_mammals', 'fish', 'flowers', ...]

annotation file:
'batch_label': 'training batch 1 of 5'
'data': ndarray
'filenames': list
CIFAR-10: 'labels': list
CIFAR-100: 'fine_labels': list
           'coarse_labels': list
```

## Export to other formats

Datumaro can convert CIFAR dataset into any other format [Datumaro supports](../user_manual.md#supported-formats).
To get the expected result, the dataset needs to be converted to formats
that support the classification task (e.g. MNIST, ImageNet, PascalVOC,
etc.) There are few ways to convert CIFAR dataset to other dataset format:

``` bash
datum project import -f cifar -i <path/to/cifar>
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

## Particular use cases

Datumaro supports filtering, transformation, merging etc. for all formats
and for the CIFAR format in particular. Follow [user manual](../user_manual.md)
to get more information about these operations.

There are few examples of using Datumaro operations to solve
particular problems with CIFAR dataset:

### Example 1. How to create custom CIFAR-like dataset

```python
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import Label, DatasetItem

dataset = Dataset.from_iterable([
    DatasetItem(id=0, image=np.ones((32, 32, 3)),
        annotations=[Label(3)]
    ),
    DatasetItem(id=1, image=np.ones((32, 32, 3)),
        annotations=[Label(8)]
    )
], categories=[['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']])

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

More examples of working with CIFAR dataset from code can be found in
[tests_cifar](../../tests/test_cifar_format.py)
