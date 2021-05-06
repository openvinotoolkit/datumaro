# Pascal VOC user manual

## Contents
- [Format specification](#format-specification)
- [Load Pascal VOC dataset](#load-pascal-voc-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export to Pascal VOC](#export-to-pascal-VOC)
- [Datumaro functionality](#datumaro-functionality)
- [Using from code](#using-from-code)

## Format specification

- Pascal VOC format specification available
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf).

- Original Pascal VOC dataset format support the followoing types of annotaions:
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
These directories contain `.txt` files
with a list of images in a subset, the subset name is the same as the `.txt` file name .

It is also possible to import specific tasks
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

To make sure that the selected dataset has been added to the project, you can run
`datum info`, which will display the project and dataset information.

## Export to other formats

Datumaro can convert Pascal VOC dataset into any other format
[Datumaro supports](../docs/user_manual.md#supported-formats).

Such conversion will only be successful if the output
format can represent the type of dataset you want to convert,
e.g. image classification annotations can be
saved in `ImageNet` format, but no as `COCO keypoints`.

There are few ways to convert Pascal VOC dataset to other dataset format:

``` bash
datum project import -f voc -i <path/to/voc>
datum export -f coco -o <path/to/output/dir>
# or
datum convert -if voc -i <path/to/voc> -f coco -o <path/to/output/dir>
```

Also it is possible using filters for converting, check
[user manual](../docs/user_manual.md#filter-project)
for information about filtering:

``` bash
datum convert -if voc -i <path/to/voc> \
    -f yolo -o <path/to/output/dir> \
    --filter-mode FILTER_MODE \
    --filter '<xpath filter expression>'
```

Some formats provide extra options for conversion.
To get information about them, run
`datum export -f <FORMAT> -- -h`
These options are passed after double dash (`--`) in the command line.
For example, the `voc_segmentation` format has an extra argument
`--label_map`, which provides an option to load a custom color map for
segmentation masks:

``` bash
datum export -f voc_segmentation -- --label-map mycolormap.txt

# mycolormap.txt [label : color_rgb : parts : actions]:
# cat:0,0,255::
# person:255,0,0:head:
```

Also you can use original voc format of label map, for example:

``` bash
datum export -f voc_layout -- --label-map voc
```

## Export to Pascal VOC

There are few ways to convert dataset to Pascal VOC format:

``` bash
# export dataset into Pascal VOC format (classification) from existing project
datum export -p <path/to/project> -f voc -o <path/to/export/dir> -- --tasks classification

# converting to Pascal VOC format from other format
datum convert -if imagenet -i <path/to/imagenet/dataset> \
    -f voc -o <path/to/export/dir> \
    -- --label_map voc --save-images
```

Argument `--tasks` allow to specify tasks for export dataset,
by default Datumaro uses all tasks.
Argument   `--label_map` allow to define user label map, for example

``` bash
# mycolormap.txt [label : color_rgb : parts : actions]:
# cat:0,0,255::
# person:255,0,0:head:
datum export -f voc_segmentation -- --label-map mycolormap.txt
```

Or you can use original Pascal VOC label map:

``` bash
datum export -f voc_layout -- --label-map voc
```

## Datumaro functionality

Datumaro supports filtering, transformation, merging etc. for all formats
and for the Pascal VOC format in particular. Follow
[user manual](../docs/user_manual.md)
to get more information about these operations.

There are few examples of using Datumaro operations to solve
particular problems:

### Example 1. Preparing Pascal VOC dataset for converting to Market-1501 dataset format.
Market-1501 dataset only has a person class, marked with a bounding box.
And to perform the conversion we could filter the Pascal VOC dataset.
With Datumaro we can do it like this

``` bash
# create Datumaro project with Pascal VOC dataset
datum import -o myproject -f voc -i <path/to/voc/dataset>

# convert labeled shapes into bboxes
datum transform -p myproject -t shapes_to_boxes

# keep only person class items
datum filter -p myproject-shapes_to_boxes \
    --mode items+annotations \
    -e '/item/annotation[label="person"]' \
    -o tmp_project

# delete other labels from dataset
datum transform -p tmp_project -o final_project \
    -t remap_labels -- -l person:person --default delete
```

To make sure that the converting was succesful we can check the output project:

```bash
cd <path/to/final/project>
datum info
```

### Example 2. Get difference between datasets
When multiple datasets are used for research, it can be useful to find out how the
datasets differ from each other, to see information about this difference, you
can run `datum diff`. For example calculate difference between Pascal VOC 2007
and Pascal VOC 2012 trainval subsets:

```bash
datum import -o ./project2007 -f voc -i <path/to/voc/2007>
datum import -o ./project2012 -f voc -i <path/to/voc/2012>
datum filter -p ./project2007 -e '/item[subset="trainval"]' -o ../trainval_voc2007
datum filter -p ./project2012 -e '/item[subset="trainval"]' -o ../trainval_voc2012
datum diff -p ../trainval_voc2012 ../trainval_voc2007

Datasets have different lengths: 17125 vs 5011
Unmatched items in the first dataset: {('2012_002332', 'trainval'), ...}
Unmatched items in the second dataset: {('001580', 'trainval'), ...}
```

This result matches with the official description of datasets
[Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/dbstats.html) and
[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/dbstats.html)

## Using from code

There are few examples of working with Pascal VOC dataset from code.
Some examples are also available in the
[tests](../tests/test_voc_format.py).

### Example 1
Load Pascal VOC dataset, and export train subset with items
which has `jumping` attribute:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('./VOC2012', 'voc')

train_dataset = dataset.get_subset('train').as_dataset()

def only_jumping(item):
    for ann in item.annotations:
        if ann.attributes.get('jumping'):
            return True
    return False

train_dataset.select(only_jumping)

train_dataset.export('./jumping_label_me', 'label_me', save_images=True)
```

### Example 2
Get information about items in Pascal VOC 2012 dataset for segmentation task:

```python
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import AnnotationType

dataset = Dataset.import_from('./VOC2012', 'voc')

def has_mask(item):
    for ann in item.annotations:
        if ann.type in {AnnotationType.polygon, AnnotationType.mask}:
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