---
title: 'Velodyne Points / KITTI Raw 3D'
linkTitle: 'Velodyne Points / KITTI Raw 3D'
description: ''
weight: 5
---

## Format specification

Velodyne Points / KITTI Raw 3D data format homepage is
available [here](http://www.cvlibs.net/datasets/kitti/raw_data.php).

Velodyne Points / KITTI Raw 3D data format specification
is available [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_raw_data.zip).

Supported annotation types:
- `Cuboid3d` (represent tracks)

Supported annotation attributes:
- `truncation` (write, string), possible values: `truncation_unset`,
  `in_image`, `truncated`, `out_image`, `behind_image` (case-independent).
- `occlusion` (write, string), possible values: `occlusion_unset`, `visible`,
  `partly`, `fully` (case-independent). This attribute has priority
  over `occluded`.
- `occluded` (read/write, boolean)
- `keyframe` (read/write, boolean). Responsible for `occlusion_kf` field.
- `track_id` (read/write, integer). Indicates the group over frames for
  annotations, represent tracks.

Supported image attributes:
- `frame` (read/write, integer). Indicates frame number of the image.

## Import KITTI Raw dataset

The velodyne points/KITTI Raw dataset is available for download
[here](http://www.cvlibs.net/datasets/kitti/raw_data.php) and
[here](https://cloud.enterprise.deepsystems.io/s/YcyfIf5zrS7NZcI/download).

KITTI Raw dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── image_00/ # optional, aligned images from different cameras
    │   └── data/
    │       ├── <name1.ext>
    │       └── <name2.ext>
    ├── image_01/
    │   └── data/
    │       ├── <name1.ext>
    │       └── <name2.ext>
    ...
    │
    ├── velodyne_points/ # optional, 3d point clouds
    │   └── data/
    │       ├── <name1.pcd>
    │       └── <name2.pcd>
    ├── tracklet_labels.xml
    └── frame_list.txt # optional, required for custom image names

```

The format does not support arbitrary image names and paths, but Datumaro
provides an option to use a special index file to allow this.

`frame_list.txt` contents:
```
12345 relative/path/to/name1/from/data
46 relative/path/to/name2/from/data
...
```

A Datumaro project with a KITTI source can be created in the following way:

```bash
datum create
datum import --format kitti_raw <path/to/dataset>
```

To make sure that the selected dataset has been added to the project,
you can run `datum project info`, which will display the project and dataset
information.

## Export to other formats

Datumaro can convert a KITTI Raw dataset into any other
format [Datumaro supports](/docs/user-manual/supported_formats/).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. 3D point clouds can be saved in Supervisely Point Clouds format,
but not in COCO keypoints.

There are several ways to convert a KITTI Raw dataset to other dataset formats:

``` bash
datum create
datum import -f kitti_raw <path/to/kitti_raw>
datum export -f sly_pointcloud -o <output/dir>
# or
datum convert -if kitti_raw -i <path/to/kitti_raw> -f sly_pointcloud
```

Or, using Python API:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'kitti_raw')
dataset.export('save_dir', 'sly_pointcloud', save_images=True)
```

## Export to KITTI Raw

There are several ways to convert a dataset to KITTI Raw format:

``` bash
# export dataset into KITTI Raw format from existing project
datum export -p <path/to/project> -f kitti_raw -o <output/dir> \
    -- --save-images
# converting to KITTI Raw format from other format
datum convert -if sly_pointcloud -i <path/to/dataset> \
    -f kitti_raw -o <output/dir> -- --save-images --reindex
```

Extra options for exporting to KITTI Raw format:
- `--save-images` allow to export dataset with saving images. This will
  include point clouds and related images (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original or use `.png`, if none)
- `--reindex` assigns new indices to frames and tracks. Allows annotations
  without `track_id` attribute (they will be exported as single-frame tracks).
- `--allow-attrs` allows writing arbitrary annotation attributes. They will
  be written in `<annotations>` section of `<poses><item>`
  (disabled by default)

## Examples

### Example 1. Import dataset, compute statistics

```bash
datum create -o project
datum import -p project -f kitti_raw ../kitti_raw/
datum stats -p project
```

### Example 2. Convert Supervisely Pointclouds to KITTI Raw

``` bash
datum convert -if sly_pointcloud -i ../sly_pcd/ \
    -f kitti_raw -o my_kitti/ -- --save-images --allow-attrs
```

### Example 3. Create a custom dataset

``` python
from datumaro.components.annotation import Cuboid3d
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem

dataset = Dataset.from_iterable([
  DatasetItem(id='some/name/qq',
    annotations=[
      Cuboid3d(position=[13.54, -9.41, 0.24], label=0,
        attributes={'occluded': False, 'track_id': 1}),

      Cuboid3d(position=[3.4, -2.11, 4.4], label=1,
        attributes={'occluded': True, 'track_id': 2})
    ],
    pcd='path/to/pcd1.pcd',
    related_images=[np.ones((10, 10)), 'path/to/image2.png', 'image3.jpg'],
    attributes={'frame': 0}
  ),
], categories=['cat', 'dog'])

dataset.export('my_dataset/', format='kitti_raw', save_images=True)
```

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_kitti_raw_format.py)
