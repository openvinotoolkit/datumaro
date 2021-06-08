# KITTI user manual

## Contents

- [Format specification](#format-specification)
- [Load KITTI dataset](#load-KITTI-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export to KITTI](#export-to-KITTI)
- [Particular use cases](#particular-use-cases)

## Format specification

- Original KITTI dataset format support the following types of annotations:
    - `Bounding boxes` (for [object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark));
    - `Masks` (for [segmentation](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) task).

- Supported attributes:
    - `truncated`: indicates that the bounding box specified for the object does
    not correspond to the full extent of the object;
    - `occluded`: indicates that a significant portion of the object within the
    bounding box is occluded by another object.

KITTI segmentations format specification available in `README.md` [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_semantics.zip).

KITTI object detection format specification available in `README.md` [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip).

##  Load KITTI dataset

The KITTI left color images for object detection are available [here](http://www.cvlibs.net/download.php?file=data_object_image_2.zip).

The KITTI object detection labels are available [here](http://www.cvlibs.net/download.php?file=data_object_label_2.zip).

The KITTI segmentations dataset is available [here](http://www.cvlibs.net/download.php?file=data_semantics.zip).

There are two ways to create Datumaro project and add KITTI dataset to it:

``` bash
datum import --format kitti --input-path <path/to/dataset>
# or
datum create
datum add path -f kitti <path/to/dataset>
```

It is possible to specify project name and project directory run
`datum create --help` for more information.

KITTI segmentation dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── testing/
    │   └── image_2/
    │       ├── <name_1>.png
    │       ├── <name_2>.png
    │       └── ...
    ├── training/
    │   ├── image_2/ # left color camera images
    │   │   ├── <name_1>.png
    │   │   ├── <name_2>.png
    │   │   └── ...
    │   ├── label_2/ # left color camera label files
    │   │   ├── <name_1>.txt
    │   │   ├── <name_2>.txt
    │   │   └── ...
    │   ├── instance/ # instance segmentation masks
    │   │   ├── <name_1>.png
    │   │   ├── <name_2>.png
    │   │   └── ...
    │   ├── semantic/ # semantic segmentation masks (labels are encoded by its id)
    │   │   ├── <name_1>.png
    │   │   ├── <name_2>.png
    │   │   └── ...
    │   └── semantic_rgb/ # semantic segmentation masks (labels are encoded by its color)
    │       ├── <name_1>.png
    │       ├── <name_2>.png
    │       └── ...
```

You can import dataset for specific tasks
of KITTI dataset instead of the whole dataset,
for example:

``` bash
datum add path -f kitti_detection <path/to/dataset>
```

Datumaro supports the following KITTI tasks:
- Object detection (`kitti_detection`)
- Class and instance segmentation (`kitti_segmentation`)

To make sure that the selected dataset has been added to the project, you can run
`datum info`, which will display the project and dataset information.

##  Export to other formats

Datumaro can convert KITTI dataset into any other format [Datumaro supports](../user_manual.md#supported-formats).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. segmentation annotations can be
saved in `Cityscapes` format, but no as `COCO keypoints`.

There are few ways to convert KITTI dataset to other dataset format:

``` bash
datum project import -f kitti -i <path/to/kitti>
datum export -f cityscapes -o <path/to/output/dir>
# or
datum convert -if kitti -i <path/to/kitti> -f cityscapes -o <path/to/output/dir>
```

Some formats provide extra options for conversion.
These options are passed after double dash (`--`) in the command line.
To get information about them, run

`datum export -f <FORMAT> -- -h`

##  Export to KITTI

There are few ways to convert dataset to KITTI format:

``` bash
# export dataset into KITTI format from existing project
datum export -p <path/to/project> -f kitti -o <path/to/export/dir> \
    -- --save-images
# converting to KITTI format from other format
datum convert -if cityscapes -i <path/to/cityscapes/dataset> \
    -f kitti -o <path/to/export/dir> -- --save-images
```

Extra options for export to KITTI format:
- `--save-images` allow to export dataset with saving images
(by default `False`);
- `--image-ext IMAGE_EXT` allow to specify image extension
for exporting dataset (by default - keep original or use `.png`, if none).
- `--apply-colormap APPLY_COLORMAP` allow to use colormap for class masks
(in folder `semantic_rgb`, by default `True`);
- `--label_map` allow to define a custom colormap. Example

``` bash
# mycolormap.txt :
# 0 0 255 sky
# 255 0 0 person
#...
datum export -f kitti -- --label-map mycolormap.txt

# or you can use original kitti colomap:
datum export -f kitti -- --label-map kitti
```
- `--tasks TASKS` allow to specify tasks for export dataset,
by default Datumaro uses all tasks. Example:

```bash
datum import -o project -f kitti -i <dataset>
datum export -p project -f kitti -- --tasks detection
```
- `--allow-attributes ALLOW_ATTRIBUTES` allow export of attributes
(by default `True`).

## Particular use cases

Datumaro supports filtering, transformation, merging etc. for all formats
and for the KITTI format in particular. Follow
[user manual](../user_manual.md)
to get more information about these operations.

There are few examples of using Datumaro operations to solve
particular problems with KITTI dataset:

### Example 1. How to load an original KITTI dataset and convert to Cityscapes

```bash
datum create -o project
datum add path -p project -f kitti ./KITTI/
datum stats -p project
datum export -p final_project -o dataset -f cityscapes --overwrite -- --save-images
```

### Example 2. How to create custom KITTI-like dataset

```python
import numpy as np
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import Mask, DatasetItem

import datumaro.plugins.kitti_format as KITTI

label_map = OrderedDict()
label_map['background'] = (0, 0, 0)
label_map['label_1'] = (1, 2, 3)
label_map['label_2'] = (3, 2, 1)
categories = KITTI.make_kitti_categories(label_map)

dataset = Dataset.from_iterable([
    DatasetItem(id=1,
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 0, 0, 1, 1]]), label=1, id=0,
                        attributes={'is_crowd': False}),
                    Mask(image=np.array([[0, 1, 1, 0, 0]]), label=2, id=0,
                        attributes={'is_crowd': False}),
                ]
            ),
    ], categories=categories)

dataset.export('./dataset', format='kitti')
```

More examples of working with KITTI dataset from code can be found in
[tests](../../tests/test_kitti_format.py)
