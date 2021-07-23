# Open Images user manual

## Contents

- [Format specification](#format-specification)
- [Load Open Images dataset](#load-open-images-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export to Open Images](#export-to-open-images)
- [Particular use cases](#particular-use-cases)

## Format specification

A description of the Open Images Dataset (OID) format is available
on [its website](https://storage.googleapis.com/openimages/web/download.html).
Datumaro supports versions 4, 5 and 6.

Datumaro currently supports the following types of annotations:

- human-verified image-level labels;
- bounding boxes;
- segmentation masks.

One attribute is supported on the labels:

- `score` (read/write, float).
  The confidence level from 0 to 1.
  A score of 0 indicates that
  the image does not contain objects of the corresponding class.

The following attributes are supported on the bounding boxes:

- `score` (read/write, float).
  The confidence level from 0 to 1.
  In the original dataset this is always equal to 1,
  but custom datasets may be created with arbitrary values.
- `occluded` (read/write, boolean).
  Whether the object is occluded by another object.
- `truncated` (read/write, boolean).
  Whether the object extends beyond the boundary of the image.
- `is_group_of` (read/write, boolean).
  Whether the object represents a group of objects of the same class.
- `is_depiction` (read/write, boolean).
  Whether the object is a depiction (such as a drawing)
  rather than a real object.
- `is_inside` (read/write, boolean).
  Whether the object is seen from the inside.

The following attributes are supported on the masks:

- `box_id` (read/write, string).
  An identifier for the bounding box associated with the mask.

- `box` (read/write, dict).
  A dictionary describing the bounding box associated with the mask.
  The keys are `x`, `y`, `w` and `h`, and their values correspond
  to the analogous properties of the Datumaro `BBox` type.

- `predicted_iou` (read/write, float).
  Predicted IoU value with respect to the ground truth.

## Load Open Images dataset

The Open Images dataset is available for free download.

See the [`open-images-dataset` GitHub repository](https://github.com/cvdfoundation/open-images-dataset)
for information on how to download the images.

Datumaro also requires the image description files,
which can be downloaded from the following URLs:

- [complete set](https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv)
- [train set](https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv)
- [validation set](https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv)
- [test set](https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv)

Datumaro expects at least one of the files above to be present.

In addition, the following metadata file must be present as well:

- [class descriptions](https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv)

You can optionally download the following additional metadata file:

- [class hierarchy](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json)

Annotations can be downloaded from the following URLs:

- [train image labels](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv)
- [validation image labels](https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv)
- [test image labels](https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv)
- [train bounding boxes](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv)
- [validation bounding boxes](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv)
- [test bounding boxes](https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv)
- [train segmentation masks (metadata)](https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv)
- train segmentation masks (images):
  [0](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-0.zip)
  [1](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-1.zip)
  [2](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-2.zip)
  [3](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-3.zip)
  [4](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-4.zip)
  [5](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-5.zip)
  [6](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-6.zip)
  [7](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-7.zip)
  [8](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-8.zip)
  [9](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-9.zip)
  [a](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-a.zip)
  [b](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-b.zip)
  [c](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-c.zip)
  [d](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-d.zip)
  [e](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-e.zip)
  [f](https://storage.googleapis.com/openimages/v5/train-masks/train-masks-f.zip)
- [validation segmentation masks (metadata)](https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv)
- validation segmentation masks (images):
  [0](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-0.zip)
  [1](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-1.zip)
  [2](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-2.zip)
  [3](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-3.zip)
  [4](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-4.zip)
  [5](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-5.zip)
  [6](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-6.zip)
  [7](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-7.zip)
  [8](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-8.zip)
  [9](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-9.zip)
  [a](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-a.zip)
  [b](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-b.zip)
  [c](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-c.zip)
  [d](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-d.zip)
  [e](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-e.zip)
  [f](https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-f.zip)
- [test segmentation masks (metadata)](https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv)
- test segmentation masks (images):
  [0](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-0.zip)
  [1](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-1.zip)
  [2](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-2.zip)
  [3](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-3.zip)
  [4](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-4.zip)
  [5](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-5.zip)
  [6](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-6.zip)
  [7](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-7.zip)
  [8](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-8.zip)
  [9](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-9.zip)
  [a](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-a.zip)
  [b](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-b.zip)
  [c](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-c.zip)
  [d](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-d.zip)
  [e](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-e.zip)
  [f](https://storage.googleapis.com/openimages/v5/test-masks/test-masks-f.zip)

All annotation files are optional,
except that if the mask metadata files for a given subset are downloaded,
all corresponding images must be downloaded as well, and vice versa.

There are two ways to create Datumaro project and add OID to it:

``` bash
datum import --format open_images --input-path <path/to/dataset>
# or
datum create
datum add path -f open_images <path/to/dataset>
```

It is possible to specify project name and project directory; run
`datum create --help` for more information.

Open Images dataset directory should have the following structure:

```
└─ Dataset/
    ├── annotations/
    │   └── bbox_labels_600_hierarchy.json
    │   └── image_ids_and_rotation.csv
    │   └── oidv6-class-descriptions.csv
    │   └── *-annotations-bbox.csv
    │   └── *-annotations-human-imagelabels.csv
    │   └── *-annotations-object-segmentation.csv
    ├── images/
    |   ├── test/
    |   │   ├── <image_name1.jpg>
    |   │   ├── <image_name2.jpg>
    |   │   └── ...
    |   ├── train/
    |   │   ├── <image_name1.jpg>
    |   │   ├── <image_name2.jpg>
    |   │   └── ...
    |   └── validation/
    |       ├── <image_name1.jpg>
    |       ├── <image_name2.jpg>
    |       └── ...
    └── masks/
        ├── test/
        │   ├── <mask_name1.png>
        │   ├── <mask_name2.png>
        │   └── ...
        ├── train/
        │   ├── <mask_name1.png>
        │   ├── <mask_name2.png>
        │   └── ...
        └── validation/
            ├── <mask_name1.png>
            ├── <mask_name2.png>
            └── ...
```

The mask images must be extracted from the ZIP archives linked above.

To use per-subset image description files instead of `image_ids_and_rotation.csv`,
place them in the `annotations` subdirectory.

### Creating an image metadata file

To load bounding box and segmentation mask annotations,
Datumaro needs to know the sizes of the corresponding images.
By default, it will determine these sizes by loading each image from disk,
which requires the images to be present and makes the loading process slow.

If you want to load the aforementioned annotations on a machine where
the images are not available,
or just to speed up the dataset loading process,
you can extract the image size information in advance
and record it in an image metadata file.
This file must be placed at `annotations/images.meta`,
and must contain one line per image, with the following structure:

```
<ID> <height> <width>
```

Where `<ID>` is the file name of the image without the extension,
and `<height>` and `<width>` are the dimensions of that image.
`<ID>` may be quoted with either single or double quotes.

The image metadata file, if present, will be used to determine the image
sizes without loading the images themselves.

Here's one way to create the `images.meta` file using ImageMagick,
assuming that the images are present on the current machine:

```bash
# run this from the dataset directory
find images -name '*.jpg' -exec \
    identify -format '"%[basename]" %[height] %[width]\n' {} + \
    > annotations/images.meta
```

## Export to other formats

Datumaro can convert OID into any other format [Datumaro supports](../user_manual.md#supported-formats).
To get the expected result, the dataset needs to be converted to a format
that supports image-level labels.
There are a few ways to convert OID to other dataset format:

``` bash
datum project import -f open_images -i <path/to/open_images>
datum export -f cvat -o <path/to/output/dir>
# or
datum convert -if open_images -i <path/to/open_images> -f cvat -o <path/to/output/dir>
```

Some formats provide extra options for conversion.
These options are passed after double dash (`--`) in the command line.
To get information about them, run

`datum export -f <FORMAT> -- -h`

## Export to Open Images

There are few ways to convert an existing dataset to the Open Images format:

``` bash
# export dataset into Open Images format from existing project
datum export -p <path/to/project> -f open_images -o <path/to/export/dir> \
  -- --save_images

# convert a dataset in another format to the Open Images format
datum convert -if imagenet -i <path/to/imagenet/dataset> \
    -f open_images -o <path/to/export/dir> \
    -- --save-images
```

Extra options for export to the Open Images format:

- `--save-images` - save image files when exporting the dataset
  (by default, `False`)

- `--image-ext IMAGE_EXT` - save image files with the speficied extension
  when exporting the dataset (by default, uses the original extension
  or `.jpg` if there isn't one)

## Particular use cases

Datumaro supports filtering, transformation, merging etc. for all formats
and for the Open Images format in particular. Follow
[user manual](../user_manual.md)
to get more information about these operations.

Here are a few examples of using Datumaro operations to solve
particular problems with the Open Images dataset:

### Example 1. How to load the Open Images dataset and convert to the format used by CVAT

```bash
datum create -o project
datum add path -p project -f open_images ./open-images-dataset/
datum stats -p project
datum export -p project -o dataset -f cvat --overwrite -- --save-images
```

### Example 2. How to create a custom OID-like dataset

```python
import numpy as np
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (
    AnnotationType, Label, LabelCategories, DatasetItem,
)

dataset = Dataset.from_iterable(
    [
        DatasetItem(
            id='0000000000000001',
            image=np.ones((1, 5, 3)),
            subset='validation',
            annotations=[
                Label(0, attributes={'score': 1}),
                Label(1, attributes={'score': 0}),
            ],
        ),
    ],
    categories=['/m/0', '/m/1'],
)
dataset.export('./dataset', format='open_images')
```

More examples of working with OID from code can be found in
[tests](../../tests/test_open_images_format.py).
