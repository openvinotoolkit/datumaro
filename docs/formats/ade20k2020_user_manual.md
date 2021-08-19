# ADE20K 2020 user manual

## Contents
- [Format specification](#format-specification)
- [Load ADE20K 2020 dataset](#load-ade20k-2020-dataset)
- [Export to other formats](#export-to-other-formats)

## Format specification

- The original ADE20K 2020 dataset available
[here](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

- Also the consistency set (for checking the annotation consistency)
available [here](https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2017_05_30_consistency.zip).

- ADE20K format supports the following type of annotations:
  - `Masks`

- The following attributes are supported on the masks:
  - `occluded` (boolean): whether the object is occluded by another object.
  - other arbitrary boolean user attributes can be specified
    in the annotation file `<image_name>.json`

# Load ADE20K dataset

- There are two ways to create Datumaro project and add ADE20K to it:

```bash
datum import --format ade20k2020 --input-path <path/to/dataset>
# or
datum create
datum add path -f ade20k2020 <path/to/dataset>
```

- Also it is possible to load dataset using Python API:

```python
from datumaro.components.dataset import Dataset

ade20k_dataset = Dataset.import_from('<path/to/dataset>', 'ade20k2020')
```

ADE20K dataset directory should has the following structure:

```
dataset
├── subset1
│   ├── img1  # directory with instance masks for img1
│   |    ├── instance_001_img1.png
│   |    ├── instance_002_img1.png
│   |    ├── ...
│   ├── img1.jpg
│   ├── img1.json
│   └── img1_seg.png
│   ├── img1_parts_1.png
│   ├── img2  # directory with instance masks for img2
│   |    ├── instance_001_img2.png
│   |    ├── instance_002_img2.png
│   |    ├── ...
│   └── img2.jpg
│   └── img2.json
│   └── ...
└── subset2
│   ├── super_label_1
│   |   ├── img3  # directory with instance masks for img3
│   |   |    ├── instance_001_img3.png
│   |   |    ├── instance_002_img3.png
│   |   |    ├── ...
│   |   ├── img3.jpg
│   |   ├── img3.json
│   |   ├── img3_seg.png
│   |   ├── img3_parts_1.png
│   |   ├── ...
│   ├── img4  # directory with instance masks for img4
│   |   ├── instance_001_img4.png
│   |   ├── instance_002_img4.png
│   |   ├── ...
│   ├── img4.jpg
│   ├── img4.json
│   └── img4_seg.png
│   └── ...
```

The mask images `<image_name>_seg.png` contain information about the object
class segmentation masks and also separates each class into instances.
The channels R and G encode the objects class masks.
The channel B encodes the instance object masks.

The mask images `<image_name>_parts_N.png` contain segmentation mask for
parts of objects, where N is a number indicating the level in the part
hierarchy.

The `<image_name>` directory contains instance masks for each
object in the image, these masks represent one-channel images,
each pixel of which indicates an affinity to a specific object.

The annotation files `<image_name>.json` describing the content of each image.
See our [tests asset](../../tests/assets/ade20k2020_dataset)
for example of this file,
or check [ADE20K toolkit](https://github.com/CSAILVision/ADE20K) for it.
# Export to other formats

Datumaro can convert ADE20K into any other format [Datumaro supports](../user_manual.md#supported-formats).
To get the expected result, the dataset needs to be converted to a format
that supports segmentation masks.

There are a few ways to convert ADE20k to other dataset format using CLI:

```bash
datum import -f ade20k2020 -i <path/to/dataset>
datum export -f coco -o ./save_dir -- --save-images
# or
datum convert -if ade20k2020 -i <path/to/dataset> -f coco -o ./save_dir \
    --save-images
```

Or using Python API

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'ade20k2020')
dataset.export('save_dir', 'voc')
```
