---
title: 'YOLOv8'
linkTitle: 'YOLOv8'
description: ''
---

## Format specification

The YOLOv8 format family allows you to define the dataset root directory, the relative paths
to training/validation/testing image directories or *.txt files containing image paths,
and a dictionary of class names.

Family consists of four formats:
- [Detection](https://docs.ultralytics.com/datasets/detect/)
- [Oriented bounding Box](https://docs.ultralytics.com/datasets/obb/)
- [Segmentation](https://docs.ultralytics.com/datasets/segment/)
- [Pose](https://docs.ultralytics.com/datasets/pose/)

Supported annotation types and formats:
- `Bbox`
  - Detection (only not rotated)
  - Oriented Bounding Box,
- `Polygon`
  - Segmentation
- `Skeleton`
  - Pose

The format supports arbitrary subset names, except `classes`, `names`, `backup`, `path`, `kpt_shape`, `flip_idx`.

> Note, that by default, the YOLO framework does not expect any subset names,
  except `train` and `val`, Datumaro supports this as an extension.
  If there is no subset separation in a project, the data
  will be saved in the `train` subset.

## Import YOLOv8 dataset
To create a Datumaro project with a YOLOv8 source, use the following commands:

```bash
datum create
datum import --format yolov8 <path/to/dataset> # for Detection dataset
datum import --format yolov8_oriented_boxes <path/to/dataset> # for Oriented Bounding Box dataset
datum import --format yolov8_segmentation <path/to/dataset> # for Segmentation dataset
datum import --format yolov8_pose <path/to/dataset> # for Pose dataset
```

The YOLOv8 dataset directory should have the following structure:

```bash
└─ yolo_dataset/
   │ # a list of non-format labels (optional)  # file with list of classes
   ├── data.yaml    # file with dataset information
   ├── train.txt    # list of image paths in train subset [Optional]
   ├── val.txt    # list of image paths in valid subset [Optional]
   │
   ├── images/
   │   ├── train/  # directory with images for train subset
   │   │    ├── image1.jpg
   │   │    ├── image2.jpg
   │   │    ├── image3.jpg
   │   │    └── ...
   │   ├── valid/  # directory with images for validation subset
   │   │    ├── image11.jpg
   │   │    ├── image12.jpg
   │   │    ├── image13.jpg
   │   │    └── ...
   ├── labels/
   │   ├── train/  # directory with annotations for train subset
   │   │    ├── image1.txt
   │   │    ├── image2.txt
   │   │    ├── image3.txt
   │   │    └── ...
   │   ├── valid/  # directory with annotations for validation subset
   │   │    ├── image11.txt
   │   │    ├── image12.txt
   │   │    ├── image13.txt
   │   │    └── ...
```

`data.yaml` should have the following content:

```yaml
path:  ./ # dataset root dir
train: train.txt  # train images (relative to 'path') 4 images
val: val.txt  # val images (relative to 'path') 4 images

# YOLOv8 Pose specific field
# First number is a number of points in skeleton
# Second number defines a format of point info in an annotation txt files
kpt_shape: [17, 3]

# Classes
names:
  0: person
  1: bicycle
  2: car
  # ...
  77: teddy bear
  78: hair drier
  79: toothbrush
```
> Note, that though by default YOLOv8 framework expects `data.yaml`,
  Datumaro allows this file to have arbitrary name.

`data.yaml` can specify what images a subset contains in 3 ways:
1. subset can point to a folder, in which case all images in the folder will belong to the subset:
   ```yaml
   val: images/valid
   ```
2. subset can be described as a list of images:
   ```yaml
   val:
   - images/valid/image1.jpg
   - images/valid/image2.jpg
   ```
3. subset can point at a `.txt` file which contains a list of images:
   ```yaml
   val: val.txt
   ```
   `val.txt` should have the following structure:
   ```txt
   <path/to/image1.jpg>
   <path/to/image2.jpg>
   ...
   ```

Files in directories `labels/train/` and `labels/valid/` should
contain information about labels for images in `images/train` and `images/valid` respectively.
If there are no objects in an image, no `.txt` file is required.

Content of the `.txt` file depends on format.

For **Detection** it contains bounding boxes:
```txt
# labels/train/image1.txt:
# <label_index> <x_center> <y_center> <width> <height>
0 0.250000 0.400000 0.300000 0.400000
3 0.600000 0.400000 0.400000 0.266667
...
```

For **Oriented Bounding Box** it contains coordinates of four corners of oriented bounding box:
```txt
# labels/train/image1.txt:
# <label_index> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
0 0.146731 0.151795 0.319936 0.301795 0.186603 0.648205 0.013397 0.498205
3 0.557735 0.090192 0.357735 0.609808 0.242265 0.509808 0.442265 -0.009808
...
```

For **Segmentation** it contains coordinates of all points of polygon.
A polygon can have three or more points:
```txt
# labels/train/image1.txt:
# <label_index> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <x5> <y5> ...
0 0.146731 0.151795 0.319936 0.301795 0.186603 0.648205
3 0.557735 0.090192 0.357735 0.609808 0.242265 0.509808 0.442265 -0.009808 0.400000 0.266667
...
```

For **Pose** it contains bounding boxes and then description of points in one of two forms.
- If the second number in kpt_shape field is 2,
  then the line contains two values for every point - `x`, `y`.
- If the second number in kpt_shape field is 3,
  then the line contains three values for every point - `x`, `y`, `visibility`,
  where `visibility` can have one of three values:
  - 0: The keypoint is not visible.
  - 1: The keypoint is partially visible.
  - 2: The keypoint is fully visible.

```txt
# labels/image1.txt:
# <label_index> <x_center> <y_center> <width> <height> <x1> <y1> <visibility1> <x2> <y2> <visibility2> ...
0 0.250000 0.400000 0.300000 0.400000 0.250000 0.400000 2 0.350000 0.500000 0 ...
3 0.600000 0.400000 0.400000 0.266667 0.250000 0.400000 1 0.440000 0.550000 2 ...
...
```

All coordinates must be normalized and be in range \[0, 1\].
It can be achieved by dividing x coordinates and widths by image width,
and y coordinates and heights by image height.


## Export to other formats

Datumaro can convert a YOLOv8 dataset into any other format Datumaro supports.
To get the expected result, convert the dataset to formats
that support the same annotations as YOLOv8 format you have.

```bash
datum create
datum add -f yolov8 <path/to/yolov8/>
datum export -f coco_instances -o <output/dir>
```
or
```bash
datum convert -if yolov8 -i <path/to/dataset> -f coco_instances -o <path/to/dataset>
```

Extra options for importing YOLOv8 format:
- `--config-file` allows to specify config file name to use instead of default `data.yaml`

Alternatively, using the Python API:

```python
from datumaro.components.dataset import Dataset

data_path = 'path/to/dataset'
data_format = 'yolov8'

dataset = Dataset.import_from(data_path, data_format)
dataset.export('save_dir', 'coco_instances')
```

## Export to YOLOv8 format
Datumaro can convert an existing dataset to YOLOv8 format
if it supports annotations from source format.

Example:

```bash
datum create
datum import -f coco_instances <path/to/dataset>
datum export -f yolov8 -o <path/to/dataset>
```

Extra options for exporting to YOLOv8 format:
- `--save-media` allow to export dataset with saving media files
  (default: `False`)
- `--image-ext <IMAGE_EXT>` allow to specify image extension
  for exporting dataset (default: use original or `.jpg`, if none)
- `--add-path-prefix` allows to specify, whether to include the
  `data/` path prefix in the annotation files or not (default: `True`)
- `--config-file` allows to specify config file name to use instead of default `data.yaml`

## Examples

### Example 1. Create a custom dataset in YOLOv8 Detection format

```python
import numpy as np
import datumaro as dm

dataset = dm.Dataset.from_iterable(
    [
        dm.DatasetItem(
            id=1,
            subset="train",
            media=dm.Image(data=np.ones((8, 8, 3))),
            annotations=[
                dm.Bbox(0, 2, 4, 2, label=2),
                dm.Bbox(0, 1, 2, 3, label=4),
            ],
        ),
    ],
    categories=["label_" + str(i) for i in range(10)],
)
dataset.export('../yolov8_dataset', format='yolov8')
```

### Example 2. Create a custom dataset in YOLOv8 Oriented Bounding Box format

Orientation of bounding boxes is controlled through `rotation` attribute of `Bbox` annotation.
Its value is a counter-clockwise angle in degrees.

```python
import numpy as np
import datumaro as dm

dataset = dm.Dataset.from_iterable(
    [
        dm.DatasetItem(
            id=1,
            subset="train",
            media=dm.Image(data=np.ones((8, 8, 3))),
            annotations=[
                dm.Bbox(0, 2, 4, 2, label=2),
                dm.Bbox(0, 1, 2, 3, label=4, attributes={"rotation": 30.0}),
            ],
        ),
    ],
    categories=["label_" + str(i) for i in range(10)],
)
dataset.export('../yolov8_dataset', format='yolov8_oriented_boxes')
```

### Example 3. Create a custom dataset in YOLOv8 Segmentation format

```python
import numpy as np
import datumaro as dm

dataset = dm.Dataset.from_iterable(
    [
        dm.DatasetItem(
            id=1,
            subset="train",
            media=dm.Image(data=np.ones((8, 8, 3))),
            annotations=[
                dm.Polygon([3.0, 1.5, 6.0, 1.5, 6.0, 7.5, 4.5, 7.5, 3.75, 3.0], label=4),
            ],
        ),
    ],
    categories=["label_" + str(i) for i in range(10)],
)
dataset.export('../yolov8_dataset', format='yolov8_segmentation')
```

### Example 4. Create a custom dataset in YOLOv8 Pose format

```python
import numpy as np
import datumaro as dm

dataset = dm.Dataset.from_iterable(
    [
        dm.DatasetItem(
            id="1",
            subset="train",
            media=dm.Image(data=np.ones((5, 10, 3))),
            annotations=[
                dm.Skeleton(
                    [
                        dm.Points([1.5, 2.0], [2], label=1),
                        dm.Points([4.5, 4.0], [2], label=2),
                        dm.Points([7.5, 6.0], [1], label=3),
                    ],
                    label=0,
                ),
            ],
        ),
    ],
    categories={
        dm.AnnotationType.label: dm.LabelCategories.from_iterable([
            "skeleton_label",
            ("point_0", "skeleton_label"),
            ("point_1", "skeleton_label"),
            ("point_2", "skeleton_label"),
        ]),
        dm.AnnotationType.points: dm.PointsCategories.from_iterable([
            (0, ["point_0", "point_1", "point_2"], set())
        ]),
    },
)
dataset.export('../yolov8_dataset', format='yolov8_pose')
```
