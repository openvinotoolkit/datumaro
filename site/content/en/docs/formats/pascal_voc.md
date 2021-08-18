---
title: 'Pascal VOC'
linkTitle: 'Pascal VOC'
description: ''
weight: 9
---

## Format specification

- Pascal VOC format specification available
  [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf).

- Original Pascal VOC dataset format support the following types of annotations:
  - `Labels` (for classification tasks);
  - `Bounding boxes` (for detection, action detection and person layout tasks);
  - `Masks` (for segmentations tasks).

- Supported attributes:
  - `occluded`: indicates that a significant portion of the object within the
    bounding box is occluded by another object;
  - `truncated`: indicates that the bounding box specified for the object does
    not correspond to the full extent of the object;
  - `difficult`: indicates that the object is considered difficult to recognize;
  - action attributes (`jumping`, `reading`, `phoning` and
    [more](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/actionexamples/index.html)).

## Load Pascal VOC dataset

The Pascal VOC dataset is available for free download
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)

There are two ways to create Datumaro project and add Pascal VOC dataset to it:

``` bash
datum import --format voc --input-path <path/to/dataset>
# or
datum create
datum add path -f voc <path/to/dataset>
```

It is possible to specify project name and project directory run
`datum create --help` for more information.
Pascal VOC dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
   ├── label_map.txt # list of non-pascal labels (optional)
   ├── Annotations/
   │     ├── ann1.xml # Pascal VOC format annotation file
   │     ├── ann2.xml
   │     ├── ...
   ├── JPEGImages/
   │    ├── img1.jpg
   │    ├── img2.jpg
   │    ├── ...
   ├── SegmentationClass/ # directory with semantic segmentation masks
   │    ├── img1.png
   │    ├── img2.png
   │    ├── ...
   ├── SegmentationObject/ # directory with instance segmentation masks
   │    ├── img1.png
   │    ├── img2.png
   │    ├── ...
   ├── ImageSets/
   │    ├── Main/ # directory with list of images for detection and classification task
   │    │   ├── test.txt  # list of image names in test subset  (without extension)
   |    |   ├── train.txt # list of image names in train subset (without extension)
   |    |   ├── ...
   │    ├── Layout/ # directory with list of images for person layout task
   │    │   ├── test.txt
   |    |   ├── train.txt
   |    |   ├── ...
   │    ├── Action/ # directory with list of images for action classification task
   │    │   ├── test.txt
   |    |   ├── train.txt
   |    |   ├── ...
   │    ├── Segmentation/ # directory with list of images for segmentation task
   │    │   ├── test.txt
   |    |   ├── train.txt
   |    |   ├── ...
```

The `ImageSets` directory should contain at least one of the directories:
`Main`, `Layout`, `Action`, `Segmentation`.
These directories contain `.txt` files with a list of images in a subset,
the subset name is the same as the `.txt` file name.

In `label_map.txt` you can define custom color map and non-pascal labels,
for example:

```
# label_map [label : color_rgb : parts : actions]
helicopter:::
elephant:0:124:134:head,ear,foot:
```
It is also possible to import grayscale (1-channel) PNG masks.
For grayscale masks provide a list of labels with the number of lines
equal to the maximum color index on images. The lines must be in the
right order so that line index is equal to the color index. Lines can
have arbitrary, but different, colors. If there are gaps in the used
color indices in the annotations, they must be filled with arbitrary
dummy labels. Example:

```
car:0,128,0:: # color index 0
aeroplane:10,10,128:: # color index 1
_dummy2:2,2,2:: # filler for color index 2
_dummy3:3,3,3:: # filler for color index 3
boat:108,0,100:: # color index 3
...
_dummy198:198,198,198:: # filler for color index 198
_dummy199:199,199,199:: # filler for color index 199
the_last_label:12,28,0:: # color index 200
```

You can import dataset for specific tasks
of Pascal VOC dataset instead of the whole dataset,
for example:

``` bash
datum add path -f voc_detection <path/to/dataset/ImageSets/Main/train.txt>
```

Datumaro supports the following Pascal VOC tasks:
- Image classification (`voc_classification`)
- Object detection (`voc_detection`)
- Action classification (`voc_action`)
- Class and instance segmentation (`voc_segmentation`)
- Person layout detection (`voc_layout`)

To make sure that the selected dataset has been added to the project, you
can run `datum info`, which will display the project and dataset information.

## Export to other formats

Datumaro can convert Pascal VOC dataset into any other format
[Datumaro supports](/docs/user-manual/supported-formats).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. image classification annotations can be
saved in `ImageNet` format, but no as `COCO keypoints`.

There are few ways to convert Pascal VOC dataset to other dataset format:

``` bash
datum import -f voc -i <path/to/voc>
datum export -f coco -o <path/to/output/dir>
# or
datum convert -if voc -i <path/to/voc> -f coco -o <path/to/output/dir>

```

Some formats provide extra options for conversion.
These options are passed after double dash (`--`) in the command line.
To get information about them, run

`datum export -f <FORMAT> -- -h`

## Export to Pascal VOC

There are few ways to convert an existing dataset to Pascal VOC format:

``` bash
# export dataset into Pascal VOC format (classification) from existing project
datum export -p <path/to/project> -f voc -o <path/to/export/dir> -- --tasks classification

# converting to Pascal VOC format from other format
datum convert -if imagenet -i <path/to/imagenet/dataset> \
    -f voc -o <path/to/export/dir> \
    -- --label_map voc --save-images
```

Extra options for export to Pascal VOC format:

- `--save-images` - allow to export dataset with saving images
  (by default `False`)

- `--image-ext IMAGE_EXT` - allow to specify image extension
  for exporting dataset (by default use original or `.jpg` if none)

- `--apply-colormap APPLY_COLORMAP` - allow to use colormap for class
  and instance masks (by default `True`)

- `--allow-attributes ALLOW_ATTRIBUTES` - allow export of attributes
  (by default `True`)

- `--keep-empty KEEP_EMPTY` - write subset lists even if they are empty
  (by default: `False`)

- `--tasks TASKS` - allow to specify tasks for export dataset,
  by default Datumaro uses all tasks. Example:

```bash
datum import -o project -f voc -i ./VOC2012
datum export -p project -f voc -- --tasks detection,classification
```

- `--label_map` allow to define a custom colormap. Example

``` bash
# mycolormap.txt [label : color_rgb : parts : actions]:
# cat:0,0,255::
# person:255,0,0:head:
datum export -f voc_segmentation -- --label-map mycolormap.txt

# or you can use original voc colomap:
datum export -f voc_segmentation -- --label-map voc
```

## Particular use cases

Datumaro supports filtering, transformation, merging etc. for all formats
and for the Pascal VOC format in particular. Follow
[user manual](/docs/user-manual/)
to get more information about these operations.

There are few examples of using Datumaro operations to solve
particular problems with Pascal VOC dataset:

### Example 1. How to prepare an original dataset for training.
In this example, preparing the original dataset to train the semantic
segmentation model includes:
loading,
checking duplicate images,
setting the number of images,
splitting into subsets,
export the result to Pascal VOC format.

```bash
datum create -o project
datum add path -p project -f voc_segmentation ./VOC2012/ImageSets/Segmentation/trainval.txt
datum stats -p project # check statisctics.json -> repeated images
datum transform -p project -o ndr_project -t ndr -- -w trainval -k 2500
datum filter -p ndr_project -o trainval2500 -e '/item[subset="trainval"]'
datum transform -p trainval2500 -o final_project -t random_split -- -s train:.8 -s val:.2
datum export -p final_project -o dataset -f voc -- --label-map voc --save-images
```

### Example 2. How to create custom dataset

```python
from datumaro.components.dataset import Dataset
from datumaro.util.image import Image
from datumaro.components.extractor import Bbox, Polygon, Label, DatasetItem

dataset = Dataset.from_iterable([
    DatasetItem(id='image1', image=Image(path='image1.jpg', size=(10, 20)),
       annotations=[Label(3),
           Bbox(1.0, 1.0, 10.0, 8.0, label=0, attributes={'difficult': True, 'running': True}),
           Polygon([1, 2, 3, 2, 4, 4], label=2, attributes={'occluded': True}),
           Polygon([6, 7, 8, 8, 9, 7, 9, 6], label=2),
        ]
    ),
], categories=['person', 'sky', 'water', 'lion'])

dataset.transform('polygons_to_masks')
dataset.export('./mydataset', format='voc', label_map='my_labelmap.txt')

"""
my_labelmap.txt:
# label:color_rgb:parts:actions
person:0,0,255:hand,foot:jumping,running
sky:128,0,0::
water:0,128,0::
lion:255,128,0::
"""
```

### Example 3. Load, filter and convert from code
Load Pascal VOC dataset, and export train subset with items
which has `jumping` attribute:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('./VOC2012', format='voc')

train_dataset = dataset.get_subset('train').as_dataset()

def only_jumping(item):
    for ann in item.annotations:
        if ann.attributes.get('jumping'):
            return True
    return False

train_dataset.select(only_jumping)

train_dataset.export('./jumping_label_me', format='label_me', save_images=True)
```

### Example 4. Get information about items in Pascal VOC 2012 dataset for segmentation task:

```python
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import AnnotationType

dataset = Dataset.import_from('./VOC2012', format='voc')

def has_mask(item):
    for ann in item.annotations:
        if ann.type == AnnotationType.mask:
            return True
    return False

dataset.select(has_mask)

print("Pascal VOC 2012 has %s images for segmentation task:" % len(dataset))
for subset_name, subset in dataset.subsets().items():
    for item in subset:
        print(item.id, subset_name, end=";")
```

After executing this code, we can see that there are 5826 images
in Pascal VOC 2012 has for segmentation task and this result is the same as the
[official documentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/dbstats.html)

More examples of working with Pascal VOC dataset from code can be found in
[tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_voc_format.py)
