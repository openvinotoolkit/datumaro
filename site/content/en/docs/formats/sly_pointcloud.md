---
title: 'Supervisely Point Cloud'
linkTitle: 'Supervisely Point Cloud'
description: ''
weight: 10
---

## Format specification

Point Cloud data format:
- [specification](https://docs.supervise.ly/data-organization/00_ann_format_navi).
- [example](https://drive.google.com/file/d/1BtZyffWtWNR-mk_PHNPMnGgSlAkkQpBl/view).

Supported annotation types:
- `cuboid_3d`

Supported annotation attributes:
- `track_id` (read/write, integer), responsible for `object` field
- `createdAt` (write, string),
- `updatedAt` (write, string),
- `labelerLogin` (write, string), responsible for the corresponding fields
  in the annotation file.
- arbitrary attributes

Supported image attributes:
- `description` (read/write, string),
- `createdAt` (write, string),
- `updatedAt` (write, string),
- `labelerLogin` (write, string), responsible for the corresponding fields
  in the annotation file.
- `frame` (read/write, integer). Indicates frame number of the image.
- arbitrary attributes

## Import Supervisely Point Cloud dataset

An example dataset in Supervisely Point Cloud format is available for download:

<https://drive.google.com/u/0/uc?id=1BtZyffWtWNR-mk_PHNPMnGgSlAkkQpBl&export=download>

Point Cloud dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── ds0/
    │   ├── ann/
    │   │   ├── <pcdname1.pcd.json>
    │   │   ├── <pcdname2.pcd.json>
    │   │   └── ...
    │   ├── pointcloud/
    │   │   ├── <pcdname1.pcd>
    │   │   ├── <pcdname1.pcd>
    │   │   └── ...
    │   ├── related_images/
    │   │   ├── <pcdname1_pcd>/
    │   │   |  ├── <image_name.ext.json>
    │   │   |  ├── <image_name.ext.json>
    │   │   └── ...
    ├── key_id_map.json
    └── meta.json
```

There are two ways to import Supervisely Point Cloud dataset:

```bash
datum import --format sly_pointcloud --input-path <path/to/dataset>
# or
datum create
datum add path -f sly_pointcloud <path/to/dataset>
```

To make sure that the selected dataset has been added to the project,
you can run `datum info`, which will display the project and dataset
information.

## Export to other formats

Datumaro can convert Supervisely Point Cloud dataset into any other
format [Datumaro supports](/docs/user-manual/supported-formats/).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. 3D point clouds can be saved in KITTI Raw format,
but not in COCO keypoints.

There are few ways to convert Supervisely Point Cloud dataset
to other dataset formats:

``` bash
datum import -f sly_pointcloud -i <path/to/sly_pcd/> -o proj/
datum export -f kitti_raw -o <path/to/output/dir> -p proj/
# or
datum convert -if sly_pointcloud -i <path/to/sly_pcd/> -f kitti_raw
```

Some formats provide extra options for conversion.
These options are passed after double dash (`--`) in the command line.
To get information about them, run

`datum export -f <FORMAT> -- -h`

## Export to Supervisely Point Cloud

There are few ways to convert dataset to Supervisely Point Cloud format:

``` bash
# export dataset into Supervisely Point Cloud format from existing project
datum export -p <path/to/project> -f sly_pointcloud -o <path/to/export/dir> \
    -- --save-images
# converting to Supervisely Point Cloud format from other format
datum convert -if kitti_raw -i <path/to/kitti_raw/dataset> \
    -f sly_pointcloud -o <path/to/export/dir> -- --save-images
```

Extra options for exporting in Supervisely Point Cloud format:

- `--save-images` allow to export dataset with saving images. This will
  include point clouds and related images (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original or use `.png`, if none)
- `--reindex` assigns new indices to frames and annotations.
- `--allow-undeclared-attrs` allows writing arbitrary annotation attributes.
  By default, only attributes specified in the input dataset metainfo
  will be written.

## Examples

### Example 1. Import dataset, compute statistics

```bash
datum create -o project
datum add path -p project -f sly_pointcloud ../sly_dataset/
datum stats -p project
```

### Example 2. Convert Supervisely Point Clouds to KITTI Raw

``` bash
datum convert -if sly_pointcloud -i ../sly_pcd/ \
    -f kitti_raw -o my_kitti/ -- --save-images --reindex --allow-attrs
```

### Example 3. Create a custom dataset

``` python
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import Cuboid3d, DatasetItem

dataset = Dataset.from_iterable([
    DatasetItem(id='frame_1',
        annotations=[
            Cuboid3d(id=206, label=0,
                position=[320.86, 979.18, 1.04],
                attributes={'occluded': False, 'track_id': 1, 'x': 1}),

            Cuboid3d(id=207, label=1,
                position=[318.19, 974.65, 1.29],
                attributes={'occluded': True, 'track_id': 2}),
        ],
        pcd='path/to/pcd1.pcd',
        attributes={'frame': 0, 'description': 'zzz'}
    ),

    DatasetItem(id='frm2',
        annotations=[
            Cuboid3d(id=208, label=1,
                position=[23.04, 8.75, -0.78],
                attributes={'occluded': False, 'track_id': 2})
        ],
        pcd='path/to/pcd2.pcd', related_images=['image2.png'],
        attributes={'frame': 1}
    ),
], categories=['cat', 'dog'])

dataset.export('my_dataset/', format='sly_pointcloud', save_images=True,
    allow_undeclared_attrs=True)
```

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_sly_pointcloud_format.py)
