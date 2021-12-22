---
title: 'MPII Human Pose Dataset'
linkTitle: 'MPII Human Pose Dataset'
description: ''
weight: 1
---

## Format specification

The original MPII Human Pose Dataset is available
[here](http://human-pose.mpi-inf.mpg.de).

Supported annotation types:
- `Bbox`
- `Points`

Supported attributes:
- `center` (a list with two coordinates of the center point
  of the object)
- `scale` (float)

## Import MPII Human Pose Dataset

The original MPII Human Pose Dataset has `MATLAB` annotation files.
Datumaro does not support these files. Instead, `JSON` and `NUMPY`
files are supported.

A Datumaro project with a MPII Human Pose Dataset source can be
created in the following way:

```bash
datum create
datum import --format mpii <path/to/dataset>
```

It is also possible to import the dataset using Python API:

```python
from datumaro.components.dataset import Dataset

mpii_dataset = Dataset.import_from('<path/to/dataset>', 'mpii')
```

MPII Human Pose Dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── jnt_visible.npy # optional
├── mpii_annotations.json
├── mpii_headboxes.npy # optional
├── mpii_pos_gt.npy # optional
├── 000000001.jpg
├── 000000002.jpg
├── 000000003.jpg
└── ...
```

## Export to other formats

Datumaro can convert an MPII Human Pose Dataset into any other format
[Datumaro supports](/docs/user-manual/supported_formats/).
To get the expected result, convert the dataset to a format
that supports bounding boxes or points.

There are several ways to convert a MPII Human Pose Dataset
to other dataset formats using CLI:

```bash
datum create
datum import -f mpii <path/to/dataset>
datum export -f voc -o ./save_dir -- --save-images
# or
datum convert -if mpii -i <path/to/dataset> \
    -f voc -o <output/dir> -- --save-images
```

Or, using Python API:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'mpii')
dataset.export('save_dir', 'voc')
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_mpii_format.py)
