---
title: 'BraTS Numpy'
linkTitle: 'BraTS Numpy'
description: ''
weight: 1
---

## Format specification

The original BraTS dataset is available
[here](https://www.med.upenn.edu/sbia/brats2018/data.html).

Supported annotation types:
- `Mask`
- `Cuboid3d`

## Import BraTS Numpy dataset

A Datumaro project with a BraTS Numpy source can be created in the following way:

```bash
datum create
datum import --format brats_numpy <path/to/dataset>
```

It is also possible to import the dataset using Python API:

```python
from datumaro.components.dataset import Dataset

brats_dataset = Dataset.import_from('<path/to/dataset>', 'brats_numpy')
```

BraTS Numpy dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── <img1>_data_cropped.npy
├── <img1>_label_cropped.npy
├── <img2>_data_cropped.npy
├── <img2>_label_cropped.npy
├── ...
├── labels
├── val_brain_bbox.p
└── val_ids.p
```

## Export to other formats

Datumaro can convert a BraTS Numpy dataset into any other format [Datumaro supports](/docs/user-manual/supported_formats/).
To get the expected result, convert the dataset to a format
that supports segmentation masks.

There are several ways to convert a BraTS Numpy dataset to other dataset
formats using CLI:

```bash
datum create
datum import -f brats_numpy <path/to/dataset>
datum export -f voc -o <output/dir> -- --save-images
```
or
``` bash
datum convert -if brats_numpy -i <path/to/dataset> \
    -f voc -o <output/dir> -- --save-images
```

Or, using Python API:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'brats_numpy')
dataset.export('save_dir', 'voc')
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_brats_numpy_format.py)
