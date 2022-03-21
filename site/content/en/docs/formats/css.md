---
title: 'Common Semantic Segmentation'
linkTitle: 'Common Semantic Segmentation'
description: ''
weight: 2
---

## Format specification

Supported annotation types:
- `Masks`

## Import Common Semantic Segmentation dataset

A Datumaro project with a CSS source can be created in the following way:

``` bash
datum create
datum import --format common_segmentation <path/to/dataset>
```

Extra options for adding a source in the COCO format:
- `--image-prefix IMAGE_PREFIX` allow to import dataset with custom image prefix
  (by default '')
- `--mask-prefix MASK_PREFIX` allow to import dataset with custom mask prefix
  (by default '')

CSS dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── dataset_meta.json # a list of labels
    ├── images/
    │   ├── <img1>.png
    │   ├── <img2>.png
    │   └── ...
    └── masks/
        ├── <img1>.png
        ├── <img2>.png
        └── ...
```

To add classes, you should use [`dataset_meta.json`](/docs/user-manual/supported_formats/#dataset-meta-file).

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

## Export to other formats

Datumaro can convert a CSS dataset into any other format [Datumaro supports](/docs/user-manual/supported_formats/).
To get the expected result, convert the dataset to formats
that support the segmentation task (e.g. PascalVOC, CamVID, CItyscapes, etc.)

There are several ways to convert a CSS dataset to other dataset
formats using CLI:

``` bash
datum create
datum import -f common_segmentation <path/to/dataset>
datum export -f voc -o <output/dir>
```
or
``` bash
datum convert -if common_segmentation -i <path/to/dataset> \
    -f cityscapes -o <output/dir> -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'common_segmentation')
dataset.export('save_dir', 'camvid', save_media=True)
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_common_segmentation_format.py)
