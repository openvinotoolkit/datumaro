# Working with Pascal VOC dataset from CLI

## Contents
- [Load Pascal VOC dataset](#load-pascal-voc-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export to Pascal VOC](#export-to-pascal-vOC)
- [Datumaro functionality](#datumaro-functionality)

## Load Pascal VOC dataset
The Pascal VOC dataset is available for free download
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)

There are two ways to create Datumaro project and add Pascal VOC dataset to it:

``` bash
datum import --format voc --input-paath <path/to/dataset>
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

- Example 1. Preparing Pascal VOC dataset for converting to Market-1501 dataset format.

``` bash
# create Datumaro project with Pascal VOC dataset
datum import -n myproject -f voc -i <path/to/voc/dataset>

# convert labeled shapes into bboxes
datum transform -t shapes_to_boxes
cd myproject-shapes_to_boxes

# keep only person class items
datum filter --mode items+annotations \
    -e '/item/annotation[label="person"]' \
    -o <path/to/output/project>

# delete other labels from dataset
datum transform -t remap_labels -o ./final_project -- -l person:person --default delete
```

To make sure that the converting was succesful we can check the output project:

```bash
cd <path/to/final/project>
datum info
```

- Example 2. Pascal VOC 2007 use about 900MB disk space, you can store half as much if keep
only store the test subset, and with Datumaro split the test subset into test
and training subsets:

```bash
# create Datumaro project with Pascal VOC 2007 (only test subset) dataset
datum import -o ./VOC2007 --name project

# split the test subset into test and training subsets
datum transform -t random_split

# or you can specify ratio for subsets (by default test:.67 train:.33)
datum transform -t random_split -- -s train:.5 -s test:.5
```

- Example 3. If you don`t need a variety of classes in Pascal VOC dataset,
with Datumaro you can rename the classes for your task and
thus assign the same name to different labels:

```bash
# create Datumaro project with Pascal VOC dataset
datum import -o ./VOC2007 --name project

# rename the labels
datum transform -t remap_labels -- -l car:vehicle -l aeroplane:vehicle \
    -l bicycle:vehicle -l boat:vehicle -l bus:vehicle -l car:vehicle \
    -l train:vehicle -l motorbike:vehicle -l bottle:indoor -l chair:indoor \
    -l diningtable:indoor -l pottedplant:indoor -l sofa:indoor \
    -l tvmonitor:indoor -l bird:animal -l cat:animal -l dog:animal \
    -l horse:animal -l sheep:animal -l cow:animal -l person:person \
    --default delete
```

- Example 4. When choosing a dataset for research, it is often useful to find out how the
datasets differ from each other, to see information about this difference, you
can run `datum diff`. For example calculate difference between Pascal VOC 2007
and Pascal VOC 2012 trainval subsets:

```bash
datum import -p ./project2007 -f voc <path/to/voc/2007>
datum import -p ./project2012 -f voc <path/to/voc/2012>
datum filer -p ./project2007 -e '/item[subset="trainval]' -o ../trainval_voc2007
datum filer -p ./project2012 -e '/item[subset="trainval]' -o ../trainval_voc2012
datum diff -p ../trainval_voc2012 ../trainval_voc2007

Datasets have different lengths: 17125 vs 5011
Unmatched items in the first dataset: {('2012_002332', 'trainval'), ...}
Unmatched items in the second dataset: {('001580', 'trainval'), ...}
```
This result mathces with the official description of datasets
[Pascal VOC 2007](#http://host.robots.ox.ac.uk/pascal/VOC/voc2007/dbstats.html) and
[Pascal VOC 2012](#http://host.robots.ox.ac.uk/pascal/VOC/voc2012/dbstats.html)