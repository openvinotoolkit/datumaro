---
title: 'VoTT JSON'
linkTitle: 'VoTT JSON'
description: ''
weight: 1
---

## Format specification

[VoTT](https://github.com/microsoft/VoTT) (Visual Object Tagging Tool) is
an open source annotation tool released by Microsoft.
[VoTT JSON](https://roboflow.com/formats/vott-json) is the format used by VoTT
when the user exports a project and selects "VoTT JSON" as the export format.

Supported annotation types:
- `Bbox`

## Import VoTT dataset

A Datumaro project with a VoTT JSON source can be created in the following way:

```bash
datum create
datum import --format vott_json <path/to/dataset>
```

It is also possible to import the dataset using Python API:

```python
from datumaro.components.dataset import Dataset

vott_json_dataset = Dataset.import_from('<path/to/dataset>', 'vott_json')
```

VoTT JSON dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── dataset_meta.json # a list of custom labels (optional)
├── img0001.jpg
├── img0002.jpg
├── img0003.jpg
├── img0004.jpg
├── ...
├── test-export.json
├── train-export.json
└── ...
```

To add custom classes, you can use [`dataset_meta.json`](/docs/user-manual/supported_formats/#dataset-meta-file).

## Export to other formats

Datumaro can convert a VoTT JSON dataset into any other format [Datumaro supports](/docs/user-manual/supported_formats/).
To get the expected result, convert the dataset to a format
that supports bounding boxes.

There are several ways to convert a VoTT JSON dataset to other dataset
formats using CLI:

```bash
datum create
datum import -f vott_json <path/to/dataset>
datum export -f voc -o ./save_dir -- --save-images
# or
datum convert -if vott_json -i <path/to/dataset> \
    -f voc -o <output/dir> -- --save-images
```

Or, using Python API:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'vott_json')
dataset.export('save_dir', 'voc')
```

## Examples

Examples of using this format from the code can be found in
[VoTT JSON tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_vott_json_format.py).
