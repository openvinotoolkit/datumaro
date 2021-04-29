# Working with Pascal VOC dataset from CLI

## Contents
- [Load Pascal VOC dataset](#load-pascal-voc-dataset)
- [Export to other formats](#export-to-other-formats)
- [Import to Pascal VOC](#import-to-pascal-vOC)
- [Datumaro functionality](#datumaro-functionality)
- [Dataset statistics](#dataset-statistics)

## Load Pascal VOC dataset
The Pascal VOC dataset is available for free download [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)

There are two ways to create datumaro project and add Pascal VOC dataset to it

``` bash
datum import --format voc --input-paath <path/to/dataset>
# or
datum create
datum add path -f voc <path/to/dataset>
```

It is possible to specify project name and project directory run `datum create --help` for more information.
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

The ImageSets directory should contain at least one of the directories: Main, Layout, Action, Segmentation. These directories contain `.txt` files with a list of images in a subset, the subset name is the same as the `.txt` file name .

Also it is possible to add Pascal VOC dataset and specify task for it, for example:

``` bash
datum add path -f voc_detection <path/to/dataset/ImageSets/Main/train.txt>
```
In addition to `voc_detection`, Datumaro supports `voc_action` (for action classification task), `voc_classification`, `voc_segmentation`, `voc_layout` (for person layout task).

To make sure that the selected dataset has been added to the project, you can run `datum info`

## Export to other formats
Datumaro supports converting Pascal VOC dataset to [all dataset formats that datumaro supports](../docs/user_manual.md#supported-formats). But the converting will be successful only if the output format supports the type of dataset you want to convert.

There are few ways to convert Pascal VOC dataset to other dataset format:

``` bash
datum project import -f voc -i <path/to/voc>
datum export -f coco -o <path/to/output/dir>
# or
datum convert -if voc -i <path/to/voc> -f coco -o <path/to/output/dir>
```

Also it is possible using filters for converting, check user manual for information about filtering:

``` bash
datum convert -if voc -i <path/to/voc> \
    -f yolo -o <path/to/output/dir> \
    --filter-mode FILTER_MODE \
    --filter '<xpath filter expression>'
```

Some formats have extra arguments for converting, run `datum export -f FORMAT -- -h` for more information. For example voc_segmentation format has extra argument `--label_map` which get opportunity to load user color map for segmentation mask:

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

## Import to Pascal VOC
There are few ways to convert dataset to Pascal VOC format:

``` bash
# export dataset into Pascal VOC format (classification) from existing project
datum export -p <path/to/project> -f voc -o <path/to/export/dir> -- --tasks classification

# converting to Pascal VOC format from other format
datum convert -if imagenet -i <path/to/imagenet/dataset> \
    -f voc -o <path/to/export/dir> \
    -- --label_map voc --save-images
```

Argument `--tasks` allow to specify tasks for export dataset, by default datumaro uses all tasks.
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
Datumaro supports filtering, transformation, merging and etc. for all formats and for the Pascal VOC format in particular. Follow [user manual](../docs/user_manual.md) to get more information about these operations.
Some examples of using datumaro operations for Pascal VOC dataset format:

``` bash
# import and compare 2009 and 2012 Pascal VOC datasets
datum import -f voc <path/to/voc2009> -o ./2009
datum import -f voc <path/to/voc2012> -o ./2012
cd 2009 && datum diff ../2012
...
Datasets have different lengths: 15674 vs 34314
Unmatched items in the second dataset: {('2011_002719', 'trainval'), ... }

# extract dataset with only car and bus class items from train subset
datum filter --mode items+annotations \
    -e '/item/annotation[label="car" or label="bus" or label="train"]' \
    -o <path/to/extract/dir>

# extract dataset has only train and val subsets
datum filter -e '/item[subset="train" or subset="val"]'

# rename item like "2008_000008" to "voc2008_000008"
datum transform -t rename -- -e '|(\d+_\d+)|voc\1|'

# reduce number of items in train subset, run datum transform -t ndr -- -h for more information
datum transform -t ndr -- -w train -k 200

# remap labels car, bus, bicycle into road_obj,
# label plane, bird into sky_obj
# and delete others labels
datum transform -t remap_labels \
    -- -l car:road_obj -l bus:road_obj -l bicycle:road_obj \
    -l plane:sky_obj -l bird:sky_obj --default delete
```

## Dataset statistics
Datumaro can calculate dataset statistics, the command `datum stats`  creating `statistics.json` file which contain detailed information of the project dataset. With `datum stats` you can check the success of the operations performed on the dataset, 
Example:
<details>

```
datum stats

# statisctics.json:
...
"annotations by type": {

"bbox": {
"count": 3
},
"caption": {
"count": 0
},
"label": {
"count": 11
},
"mask": {
"count": 1
},
"points": {
"count": 0
},
"polygon": {
"count": 0
},
"polyline": {
"count": 0
}
},
...

# perform operations:
datum transform -t boxes_to_masks -o <path/to/output/dir>

# check changes
cd <path/to/output/dir>
datum stats

# now we see that there are no boxes in the dataset, but there are 4 masks
# statistics.json
...
"annotations by type": {
"bbox": {
"count": 0
},
"caption": {
"count": 0
},
"label": {
"count": 11
},
"mask": {
"count": 4
},
"points": {
"count": 0
},
"polygon": {
"count": 0
},
"polyline": {
"count": 0
}
}
...
```