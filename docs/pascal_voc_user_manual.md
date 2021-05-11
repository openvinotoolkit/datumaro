# Pascal VOC user manual

## Contents
- [Format specification](#format-specification)
- [Load Pascal VOC dataset](#load-pascal-voc-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export to Pascal VOC](#export-to-pascal-VOC)
- [Particular use cases](#particular-use-cases)

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
datum import -f voc -i <path/to/voc>
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

## Particular use cases

Datumaro supports filtering, transformation, merging etc. for all formats
and for the Pascal VOC format in particular. Follow
[user manual](../docs/user_manual.md)
to get more information about these operations.

There are few examples of using Datumaro operations to solve
particular problems with Pascal VOC dataset:

### Example 1. How to prepare an original dataset for training.
In this example, preparing the original dataset to train the semantic segmentation model includes:
loading,
checking duplicate images,
setting the number of images,
replacing masks of instances with one mask,
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