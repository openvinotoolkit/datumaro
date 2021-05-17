# COCO user manual

## Contents

- [Format specification](#format-specification)
- [Load COCO dataset](#load-COCO-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export to COCO](#export-to-COCO)
- [Particular use cases](#particular-use-cases)

## Format specification

COCO format specification available [here](https://cocodataset.org/#format-data).

COCO dataset format supports `captions`, `image_info`, `instances`, `panoptic`,
`person_keypoints`, `stuff` annotation tasks
and, as Datumaro extension, `label` (like `instances` with only `category_id`)

##  Load COCO dataset

The COCO dataset is available for free download:

Images:
- [train images](http://images.cocodataset.org/zips/train2017.zip)
- [val images](http://images.cocodataset.org/zips/val2017.zip)
- [test images](http://images.cocodataset.org/zips/test2017.zip)
- [unlabeled images](http://images.cocodataset.org/zips/unlabeled2017.zip)

Annotations:
- [captions](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [image_info](http://images.cocodataset.org/annotations/image_info_test2017.zip)
- [instances](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)
- [person_keypoints](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [stuff](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)

There are two ways to create Datumaro project and add COCO dataset to it:

``` bash
datum import --format coco --input-path <path/to/dataset>
# or
datum create
datum add path -f coco <path/to/dataset>
```

It is possible to specify project name and project directory run
`datum create --help` for more information.

COCO dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── images/
    │   ├── train<year>
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   ├── val<year>
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    ├── annotations/
    │   └── <tasks>_train<year>.json
    │   └── <tasks>_test<year>.json
```

For `panoptic` COCO dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── images/
    │   ├── train<year>
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   ├── val<year>
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    ├── annotations/
    │   ├── panoptic_train<year>
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   ├── panoptic_train<year>.json
    │   ├── panoptic_val<year>
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   └── panoptic_val<year>.json
```

You can import dataset for specific tasks
of COCO dataset instead of the whole dataset,
for example:

``` bash
datum import --format coco_stuff --input-path <path/to/stuff.json>
```

Datumaro supports the following COCO tasks:
- [Image Captioning](https://cocodataset.org/#captions-2015) (`coco_caption`)
- [Object Detection](https://cocodataset.org/#detection-2020) (`coco_instances`, `coco_label`)
- [Panoptic Segmentation](https://cocodataset.org/#panoptic-2020) (`coco_panoptic`)
- [Keypoint Detection](https://cocodataset.org/#keypoints-2020) (`coco_person_keypoints`)
- [Stuff Segmentation](https://cocodataset.org/#stuff-2019) (`coco_stuff`)

To make sure that the selected dataset has been added to the project, you can run
`datum info`, which will display the project and dataset information.

##  Export to other formats

Datumaro can convert COCO dataset into any other format [Datumaro supports](../user_manual.md#supported-formats).
To get the expected result, the dataset needs to be converted to formats
that support the specified task (e.g. for panoptic segmentation - VOC, CamVID)
There are few ways to convert COCO dataset to other dataset format:

``` bash
datum project import -f coco -i <path/to/coco>
datum export -f voc -o <path/to/output/dir>
# or
datum convert -if coco -i <path/to/coco> -f voc -o <path/to/output/dir>
```

Some formats provide extra options for conversion.
These options are passed after double dash (`--`) in the command line.
To get information about them, run

`datum export -f <FORMAT> -- -h`

##  Export to COCO

There are few ways to convert dataset to COCO format:

``` bash
# export dataset into COCO format from existing project
datum export -p <path/to/project> -f coco -o <path/to/export/dir> \
    -- --save-images
# converting to COCO format from other format
datum convert -if voc -i <path/to/voc/dataset> \
    -f coco -o <path/to/export/dir> -- --save-images
```

Extra options for export to COCO format:
- `--save-images` allow to export dataset with saving images
(by default `False`);
- `--image-ext IMAGE_EXT` allow to specify image extension
for exporting dataset (by default `.png`);
- `--segmentation-mode MODE` allow to specify save mode for instance segmentation:
    - 'guess': guess the mode for each instance (using 'is_crowd' attribute as hint)
    - 'polygons': save polygons( merge and convert masks, prefer polygons)
    - 'mask': save masks (merge and convert polygons, prefer masks)
(by default `guess`);
- `--crop-covered` allow torop covered segments so that background objects
segmentation was more accurate (by default `False`);
- `--allow-attributes ALLOW_ATTRIBUTES` allow export of attributes
(by default `True`);
- `--reindex REINDEX` allow to assign new indices to images and annotations,
useful to avoid merge conflicts (by default `False`);
- `--merge-images` allow to save all images into a single directory
(by default `False`);
- `--tasks TASKS` allow to specify tasks for export dataset,
by default Datumaro uses all tasks. Example:

```bash
datum import -o project -f coco -i <dataset>
datum export -p project -f coco -- --tasks instances,stuff
```

## Particular use cases

Datumaro supports filtering, transformation, merging etc. for all formats
and for the COCO format in particular. Follow
[user manual](../user_manual.md)
to get more information about these operations.