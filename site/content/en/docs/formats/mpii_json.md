---
title: 'MPII Human Pose Dataset (JSON)'
linkTitle: 'MPII Human Pose Dataset (JSON)'
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

## Import MPII Human Pose Dataset (JSON)

A Datumaro project with an MPII Human Pose Dataset (JSON) source can be
created in the following way:

```bash
datum create
datum import --format mpii_json <path/to/dataset>
```

It is also possible to import the dataset using Python API:

```python
from datumaro import Dataset

mpii_dataset = Dataset.import_from('<path/to/dataset>', 'mpii_json')
```

_Links to API documentation:_
- [Dataset.import_from][]

MPII Human Pose Dataset (JSON) directory should have the following structure:

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

Datumaro can convert an MPII Human Pose Dataset (JSON) into
any other format [Datumaro supports](/docs/user-manual/supported_formats/).
To get the expected result, convert the dataset to a format
that supports bounding boxes or points.

There are several ways to convert an MPII Human Pose Dataset (JSON)
to other dataset formats using CLI:

```bash
datum create
datum import -f mpii_json <path/to/dataset>
datum export -f voc -o ./save_dir -- --save-images
```
or
``` bash
datum convert -if mpii_json -i <path/to/dataset> \
    -f voc -o <output/dir> -- --save-images
```

Or, using Python API:

```python
from datumaro import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'mpii_json')
dataset.export('save_dir', 'voc')
```

_Links to API documentation:_
- [Dataset.import_from][]

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_mpii_json_format.py)

[Dataset.import_from]: /api/api/components/components/datumaro.components.dataset.html#datumaro.components.dataset.Dataset.import_from
