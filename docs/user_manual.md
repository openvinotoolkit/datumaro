# User manual

## Contents

- [Installation](#installation)
- [How to use Datumaro](#how-to-use-datumaro)
- [Supported dataset formats and annotations](#dataset-formats)
- [Supported media formats](#media-formats)
- [Glossary](#glossary)
- [Command-line workflow](#command-line-workflow)
  - [Project layout](#project-layout)
  - [Examples](#cli-examples)
- [Command reference](#command-reference)
  - [Convert](#convert)
  - [Create](#create)
  - [Add](#source-add)
  - [Remove](#source-add)
  - [Filter](#filter)
  - [Merge](#merge)
  - [Export](#export)
  - [Diff](#diff)
  - [Info](#info)
  - [Stats](#stats)
  - [Validate](#validate)
  - [Transform](#transform)
  - [Run model inference explanation (explain)](#explain)
  - [Commit](#commit)
  - [Checkout](#checkout)
  - [Status](#status)
  - [Log](#log)
  - Models:
    - [Add](#model-add)
    - [Run](#run-model)
- [Extending](#extending)
  - [Builtin plugins](#builtin-plugins)
- [Links](#links)

## Installation

### Dependencies

- Python (3.6+)
- Optional: OpenVINO, TensforFlow, PyTorch, MxNet, Caffe, Accuracy Checker

### Installation steps

Optionally, set up a virtual environment:

``` bash
python -m pip install virtualenv
python -m virtualenv venv
. venv/bin/activate
```

Install:
``` bash
# From PyPI:
pip install datumaro

# From the GitHub repository:
pip install 'git+https://github.com/openvinotoolkit/datumaro'
```

**Plugins**

Datumaro has many plugins, which are responsible for dataset formats,
model launchers and other optional components. If a plugin has dependencies,
they can require additional installation. You can find the list of all the
plugin dependencies in the [plugins](#extending) section.

**Customizing installation parameters**

- In some cases, there can be limited use for UI elements outside CLI,
  or limited options of installing graphical libraries in the system
  (various Docker environments, servers etc). You can select bewtween using
  `opencv` and `opencv-headless` by setting the `DATUMARO_HEADLESS`
  environment variable to `0` or `1` before installing the package.
  It requires building from source:
  `DATUMARO_HEADLESS=1 pip install datumaro --no-binary=datumaro`

- Although Datumaro has `pycocotools==2.0.1` in requirements, it works with
  2.0.2 perfectly fine. The reason for such requirement is binary
  incompatibility of the `numpy` dependency in the `TensorFlow` and
  `pycocotools` binary packages
  (see [#253](https://github.com/openvinotoolkit/datumaro/issues/253))

- You can change the installation branch with `...@<branch_name>`.
  Also use `--force-reinstall` parameter in this case.
  It can be useful for testing of unreleased versions from GitHub Pull Requests

## How to use Datumaro

As a standalone tool or a Python module:

``` bash
datum --help

python -m datumaro --help
python datumaro/ --help
python datum.py --help
```

As a Python library:

``` python
from datumaro.components.project import Project
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import Label, Bbox, DatasetItem
...
dataset = Dataset.import_from(path, format)
...
```

## Supported Formats <a id="dataset-formats">

List of supported formats:
- MS COCO
  (`image_info`, `instances`, `person_keypoints`, `captions`, `labels`,`panoptic`, `stuff`)
  - [Format specification](http://cocodataset.org/#format-data)
  - [Dataset example](../tests/assets/coco_dataset)
  - `labels` are our extension - like `instances` with only `category_id`
  - [Format documentation](./formats/coco_user_manual.md)
- PASCAL VOC (`classification`, `detection`, `segmentation` (class, instances),
  `action_classification`, `person_layout`)
  - [Format specification](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html)
  - [Dataset example](../tests/assets/voc_dataset)
  - [Format documentation](./formats/pascal_voc_user_manual.md)
- YOLO (`bboxes`)
  - [Format specification](https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data)
  - [Dataset example](../tests/assets/yolo_dataset)
  - [Format documentation](./formats/yolo_user_manual.md)
- TF Detection API (`bboxes`, `masks`)
  - Format specifications: [bboxes](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md),
    [masks](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/instance_segmentation.md)
  - [Dataset example](../tests/assets/tf_detection_api_dataset)
- WIDER Face (`bboxes`)
  - [Format specification](http://shuoyang1213.me/WIDERFACE/)
  - [Dataset example](../tests/assets/wider_dataset)
- VGGFace2 (`landmarks`, `bboxes`)
  - [Format specification](https://github.com/ox-vgg/vgg_face2)
  - [Dataset example](../tests/assets/vgg_face2_dataset)
- MOT sequences
  - [Format specification](https://arxiv.org/pdf/1906.04567.pdf)
  - [Dataset example](../tests/assets/mot_dataset)
- MOTS (png)
  - [Format specification](https://www.vision.rwth-aachen.de/page/mots)
  - [Dataset example](../tests/assets/mots_dataset)
- ImageNet (`classification`, `detection`)
  - [Dataset example](../tests/assets/imagenet_dataset)
  - [Dataset example (txt for classification)](../tests/assets/imagenet_txt_dataset)
  - Detection format is the same as in PASCAL VOC
- CIFAR-10/100 (`classification` (python version))
  - [Format specification](https://www.cs.toronto.edu/~kriz/cifar.html)
  - [Dataset example](../tests/assets/cifar_dataset)
  - [Format documentation](./formats/cifar_user_manual.md)
- MNIST (`classification`)
  - [Format specification](http://yann.lecun.com/exdb/mnist/)
  - [Dataset example](../tests/assets/mnist_dataset)
  - [Format documentation](./formats/mnist_user_manual.md)
- MNIST in CSV (`classification`)
  - [Format specification](https://pjreddie.com/projects/mnist-in-csv/)
  - [Dataset example](../tests/assets/mnist_csv_dataset)
  - [Format documentation](./formats/mnist_user_manual.md)
- CamVid (`segmentation`)
  - [Format specification](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
  - [Dataset example](../tests/assets/camvid_dataset)
- Cityscapes (`segmentation`)
  - [Format specification](https://www.cityscapes-dataset.com/dataset-overview/)
  - [Dataset example](../tests/assets/cityscapes_dataset)
  - [Format documentation](./formats/cityscapes_user_manual.md)
- KITTI (`segmentation`, `detection`)
  - [Format specification](http://www.cvlibs.net/datasets/kitti/index.php)
  - [Dataset example](../tests/assets/kitti_dataset)
  - [Format documentation](./formats/kitti_user_manual.md)
- KITTI 3D (`raw`/`tracklets`/`velodyne points`)
  - [Format specification](http://www.cvlibs.net/datasets/kitti/raw_data.php)
  - [Dataset example](../tests/assets/kitti_dataset/kitti_raw)
  - [Format documentation](./formats/kitti_raw_user_manual.md)
- Supervisely (`pointcloud`)
  - [Format specification](https://docs.supervise.ly/data-organization/00_ann_format_navi)
  - [Dataset example](../tests/assets/sly_pointcloud)
  - [Format documentation](./formats/sly_pointcloud_user_manual.md)
- CVAT
  - [Format specification](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md)
  - [Dataset example](../tests/assets/cvat_dataset)
- LabelMe
  - [Format specification](http://labelme.csail.mit.edu/Release3.0)
  - [Dataset example](../tests/assets/labelme_dataset)
- ICDAR13/15 (`word_recognition`, `text_localization`, `text_segmentation`)
  - [Format specification](https://rrc.cvc.uab.es/?ch=2)
  - [Dataset example](../tests/assets/icdar_dataset)
- Market-1501 (`person re-identification`)
  - [Format specification](https://www.aitribune.com/dataset/2018051063)
  - [Dataset example](../tests/assets/market1501_dataset)
- LFW (`classification`, `person re-identification`, `landmarks`)
  - [Format specification](http://vis-www.cs.umass.edu/lfw/)
  - [Dataset example](../tests/assets/lfw_dataset)

### Supported annotation types

- Labels
- Bounding boxes
- Polygons
- Polylines
- (Segmentation) Masks
- (Key-)Points
- Captions
- 3D cuboids

## Media formats

Datumaro supports the following media types:
- 2D RGB(A) images
- KITTI Point Clouds

To create an unlabelled dataset from an arbitrary directory with images use
`image_dir` and `image_zip` formats:

```bash
datum create -o <project/dir>
datum add -p <project/dir> -f image_dir <directory/path/>
```

or, if you work with Datumaro API:

- for using with a project:

  ```python
  from datumaro.components.project import Project

  project = Project.init()
  project.import_source('source1', format='image_dir', url='directory/path/')
  dataset = project.working_tree.make_dataset()
  ```

- for using as a dataset:

  ```python
  from datumaro.components.dataset import Dataset

  dataset = Dataset.import_from('directory/path/', 'image_dir')
  ```

This will search for images in the directory recursively and add
them as dataset entries with names like `<subdir1>/<subsubdir1>/<image_name1>`.
The list of formats matches the list of supported image formats in OpenCV:
```
.jpg, .jpeg, .jpe, .jp2, .png, .bmp, .dib, .tif, .tiff, .tga, .webp, .pfm,
.sr, .ras, .exr, .hdr, .pic, .pbm, .pgm, .ppm, .pxm, .pnm
```

Once there is a `Dataset` instance, it's items can be split into subsets,
renamed, filtered, joined with annotations, exported in various formats etc.

To use a video as an input, one should either create a [plugin](#extending),
which splits a video into frames, or split the video manually and import images.

## Glossary

- Basic concepts:
  - dataset - A number of media (dataset items) and associated annotations
  - project - A combination of multiple datasets, plugins, models and metadata

- Project versioning concepts:
  - data source - a link to a dataset or a copy of a dataset inside a project.
    Basically, an URL + dataset format name
  - project revision - a commit hash or a named reference from
    Git (branch, tag, HEAD~3 etc.).
  - working / head / revision tree - a project build tree and plugins at
    a specified revision
  - data source revision - a data source hash at a specific stage
  - object - a tree or a data source revision data

- Dataset path concepts:
  - source / dataset / revision / project path - a path to a dataset in a
    special format

    - (project local) **rev**ision **path**s - a way to specify the path
      to a source revision in the CLI, the syntax is:
      `<revision>:<source/target name>`, any part can be omitted.
      - Default revision is the working tree of the project
      - Default target is the compiled project

    - dataset revpath - a path to a dataset in the following format:
      `<dataset path>:<format>`
      - Format is optional. If not specified, will try to autodetect

    - full revpath - a path to a source revision in a project, the syntax is:
      `<project path>@<revision>:<target name>`, any part can be omitted.
      - Default project is the current project (`-p`/`--project` CLI arg.)
      - Default revision is the working tree of the project
      - Default target is the compiled project

- Dataset building concepts:
  - stage - a modification of a data source. A transformation,
    filter or something else.
  - build tree - a directed graph (tree) with leaf nodes at data sources
    and a single root node called "project"
  - build target - a data source or a stage
  - pipeline - a subgraph of a build target

## Command-line workflow

In Datumaro, most command-line commands operate on projects, but there are
also few commands operating on datasets directly. There are 2 basic ways
to use Datumaro from the command-line:
- Use [`convert`](#convert), [`diff`](#compare), [`merge`](#merge)
  directly on existing datasets

- Create a Datumaro project and operate on it:
  - Create an empty project with [`create`](#create)
  - Import existing datasets with [`add`](#source-add)
  - Modify the project with [`transform`](#transform) and [`filter`](#filter)
  - Create new revisions of the project, navigate over them, compare
  - Export the resulting dataset with [`export`](#export)

Basically, a project is a combination of datasets, models and environment.

A project can contain an arbitrary number of data sources. Each data source
describes a dataset in a specific format. A project acts as a manager for
the data sources and allows to manipulate them separately or as a whole, in
which case it combines dataset items from all the sources into one composite
dataset. You can manage separate sources in a project by commands in
the `datum source` command line context.

If you want to interact with models, you need to add them to project first.

A typical way to obtain Datumaro projects is to export tasks in
[CVAT](https://github.com/openvinotoolkit/cvat) UI.

### Project layout

```bash
project
├── .dvc/
├── .dvcignore
├── .git/
├── .gitignore
├── .datumaro/
│   ├── cache/ # object cache
│   │   ├── <2 leading symbols of obj hash>/
│   │   │   └── <rest symbols of obj hash>/
│   │   │       └── <object data>
│   ├── models/
│   ├── plugins/ # custom project-specific plugins
│   │   ├── plugin1/
│   │   |   ├── __init__.py
│   │   |   └── file2.py
│   │   ├── plugin2.py
│   │   └── ...
│   ├── tmp/ # temp files
│   └── tree/ # working tree metadata
│       ├── config.yml
│       └── sources/
│           ├── <source name 1>.dvc
│           ├── <source name 2>.dvc
│           └── ...
│
├── <source name 1>/
│   └── <source data>
└── <source name 2>/
    └── <source data>
```

### Examples <a id="cli-examples"></a>

Example: create a project, add dataset, modify, restore an old version

``` bash
datum create
datum add <path/to/coco/dataset> -f coco -n source1
datum commit -m "Added a dataset"
datum transform -t shapes_to_boxes
datum filter -e '/item/annotation[label="cat" or label="dog"]' -m i+a
datum commit -m "Transformed"
datum checkout HEAD~1 -- source1 # restore a previous revision
datum status # prints "modified source1"
datum checkout source1 # restore the last revision
datum export -f voc -- --save-images
```

## Command reference

> **Note**: command invocation syntax is subject to change,
> **always refer to command --help output**

### Convert datasets <a id="convert"></a>

This command allows to convert a dataset from one format to another.
The command is a usability alias for [`create`](#create),
[`add`](#source-add) and [`export`](#export) and just provides a simpler
way to obtain the same result in simple cases. A list of supported
formats can be found in the `--help` output of this command.

Usage:

``` bash
datum convert --help

datum convert \
    -i <input path> \
    -if <input format> \
    -o <output path> \
    -f <output format> \
    -- [extra parameters for output format]
```

Example: convert a VOC-like dataset to a COCO-like one:

``` bash
datum convert --input-format voc --input-path <path/to/voc/> \
              --output-format coco \
              -- --save-images
```

### Create project <a id="create"></a>

The command creates an empty project. Once a Project is created, there are
a few options to interact with it.

Usage:

``` bash
datum create --help

datum create \
    -o <project_dir>
```

Example: create an empty project `my_dataset`

``` bash
datum create -o my_dataset/
```

### Add and remove data sources <a id="source-add"></a>

A project can contain an arbitrary number of Data Sources. Each Data Source
describes a dataset in a specific format. A project acts as a manager for
the data sources and allows to manipulate them separately or as a whole, in
which case it combines dataset items from all the sources into one composite
dataset. You can manage separate sources in a project by commands in
the `datum source` command line context.

Datasets come in a wide variety of formats. Each dataset
format defines its own data structure and rules on how to
interpret the data. For example, the following data structure
is used in COCO format:
<!--lint disable fenced-code-flag-->
```
/dataset/
- /images/<id>.jpg
- /annotations/
```
<!--lint enable fenced-code-flag-->

Check [supported formats](#dataset-formats) for more info about
format specifications, supported options and other details.
The list of formats can be extened by custom plugins, check [extending tips](#extending)
for information on this topic.

Available formats are listed in the command help output.

Datumaro supports working with datasets with annotations only.

A dataset is imported by its URL. Currently, only local filesystem
paths are supported. The URL can be a file or a directory path
to a dataset. When the dataset is read, it is read as a whole.
Many formats can have multiple subsets like `train`, `val`, `test` etc. If
you want to limit reading only to a specific subset, use the `-r/--path`
parameter. It can also be useful when subset files have non-standard
placement or names.

When a dataset is imported, the following things are done:
- URL is saved in the project config
- data in copied into the project
- data is cached inside the project (use `--no-cache` to disable)

The dataset is added into the working tree of the project. A new commit
is *not* done automatically.

Usage:

``` bash
datum add --help
datum remove --help

datum add \
    <url> \
    -p <project dir> \
    -f <format>

datum remove \
    -p <project dir> \
    -n <name>
```

Example: create a project from a bunch of different annotations and images,
and generate TFrecord for TF Detection API for model training

``` bash
datum create
# 'default' is the name of the subset below
datum add <path/to/coco/instances_default.json> -f coco_instances
datum add <path/to/cvat/default.xml> -f cvat
datum add <path/to/voc> -f voc_detection -r custom_subset_dir/default.txt
datum add <path/to/datumaro/default.json> -f datumaro
datum add <path/to/images/dir> -f image_dir
datum export -f tf_detection_api
```

### Filter project <a id="filter"></a>

This command allows to create a sub-dataset from a dataset. The new dataset
includes only items satisfying some condition. [XPath](https://devhints.io/xpath)
is used as a query format.

There are several filtering modes available (`-m/--mode` parameter).
Supported modes:
- `i`, `items`
- `a`, `annotations`
- `i+a`, `a+i`, `items+annotations`, `annotations+items`

When filtering annotations, use the `items+annotations`
mode to point that annotation-less dataset items should be
removed. To select an annotation, write an XPath that
returns `annotation` elements (see examples).

Item representations are available with `--dry-run` parameter:

``` xml
<item>
  <id>290768</id>
  <subset>minival2014</subset>
  <image>
    <width>612</width>
    <height>612</height>
    <depth>3</depth>
  </image>
  <annotation>
    <id>80154</id>
    <type>bbox</type>
    <label_id>39</label_id>
    <x>264.59</x>
    <y>150.25</y>
    <w>11.199999999999989</w>
    <h>42.31</h>
    <area>473.87199999999956</area>
  </annotation>
  <annotation>
    <id>669839</id>
    <type>bbox</type>
    <label_id>41</label_id>
    <x>163.58</x>
    <y>191.75</y>
    <w>76.98999999999998</w>
    <h>73.63</h>
    <area>5668.773699999998</area>
  </annotation>
  ...
</item>
```

Usage:

``` bash
datum filter --help

datum filter \
    -p <project dir> \
    -e '<xpath filter expression>'
```

Example: extract a dataset with only images which `width` < `height`

``` bash
datum filter \
    -p test_project \
    -e '/item[image/width < image/height]'
```

Example: extract a dataset with only images of subset `train`.
``` bash
datum project filter \
    -p test_project \
    -e '/item[subset="train"]'
```

Example: extract a dataset with only large annotations of class `cat` and any
non-`persons`

``` bash
datum filter \
    -p test_project \
    --mode annotations -e '/item/annotation[(label="cat" and area > 99.5) or label!="person"]'
```

Example: extract a dataset with only occluded annotations, remove empty images

``` bash
datum filter \
    -p test_project \
    -m i+a -e '/item/annotation[occluded="True"]'
```

### Merge datasets <a id="merge"></a>

This command merges items from 2 or more projects and checks annotations for
errors.

Spatial annotations are compared by distance and intersected, labels and
attributes are selected by voting. Merge conflicts, missing items and
annotations, other errors are saved into a `.json` file.

Usage:

``` bash
datum merge --help

datum merge <project dirs>
```

Example: merge 4 (partially-)intersecting projects,
- consider voting succeeded when there are 3+ same votes
- consider shapes intersecting when IoU >= 0.6
- check annotation groups to have `person`, `hand`, `head` and `foot`
(`?` for optional)

``` bash
datum merge project1/ project2/ project3/ project4/ \
    --quorum 3 \
    -iou 0.6 \
    --groups 'person,hand?,head,foot?'
```

### Export datasets <a id="export"></a>

This command exports a project or a source as a dataset in some format.
Check [supported formats](#dataset-formats) for more info about
format specifications, supported options and other details.
The list of formats can be extened by custom plugins, check [extending tips](#extending)
for information on this topic.

Available formats are listed in the command help output.

Usage:

``` bash
datum export --help

datum export \
    -p <project dir> \
    -o <output dir> \
    -f <format> \
    -- [additional format parameters]
```

Example: save a project as a VOC-like dataset, include images, convert
images to `PNG` from any other formats.

``` bash
datum export \
    -p test_project \
    -o test_project-export \
    -f voc \
    -- --save-images --image-ext='.png'
```

### Get project info <a id="info"></a>

This command outputs project status information.

Usage:

``` bash
datum info --help

datum info \
    -p <project dir>
```

Example:

``` bash
datum info -p /test_project

Project:
  name: test_project
  location: /test_project
Sources:
  source 'instances_minival2014':
    format: coco_instances
    url: /coco_like/annotations/instances_minival2014.json
Dataset:
  length: 5000
  categories: label
    label:
      count: 80
      labels: person, bicycle, car, motorcycle (and 76 more)
  subsets: minival2014
    subset 'minival2014':
      length: 5000
      categories: label
        label:
          count: 80
          labels: person, bicycle, car, motorcycle (and 76 more)
```

### Get project statistics <a id="stats"></a>

This command computes various project statistics, such as:
- image mean and std. dev.
- class and attribute balance
- mask pixel balance
- segment area distribution

Usage:

``` bash
datum stats --help

datum stats \
    -p <project dir>
```

Example:

<details>

``` bash
datum stats -p test_project

{
    "annotations": {
        "labels": {
            "attributes": {
                "gender": {
                    "count": 358,
                    "distribution": {
                        "female": [
                            149,
                            0.41620111731843573
                        ],
                        "male": [
                            209,
                            0.5837988826815642
                        ]
                    },
                    "values count": 2,
                    "values present": [
                        "female",
                        "male"
                    ]
                },
                "view": {
                    "count": 340,
                    "distribution": {
                        "__undefined__": [
                            4,
                            0.011764705882352941
                        ],
                        "front": [
                            54,
                            0.1588235294117647
                        ],
                        "left": [
                            14,
                            0.041176470588235294
                        ],
                        "rear": [
                            235,
                            0.6911764705882353
                        ],
                        "right": [
                            33,
                            0.09705882352941177
                        ]
                    },
                    "values count": 5,
                    "values present": [
                        "__undefined__",
                        "front",
                        "left",
                        "rear",
                        "right"
                    ]
                }
            },
            "count": 2038,
            "distribution": {
                "car": [
                    340,
                    0.16683022571148184
                ],
                "cyclist": [
                    194,
                    0.09519136408243375
                ],
                "head": [
                    354,
                    0.17369970559371933
                ],
                "ignore": [
                    100,
                    0.04906771344455348
                ],
                "left_hand": [
                    238,
                    0.11678115799803729
                ],
                "person": [
                    358,
                    0.17566241413150147
                ],
                "right_hand": [
                    77,
                    0.037782139352306184
                ],
                "road_arrows": [
                    326,
                    0.15996074582924436
                ],
                "traffic_sign": [
                    51,
                    0.025024533856722278
                ]
            }
        },
        "segments": {
            "area distribution": [
                {
                    "count": 1318,
                    "max": 11425.1,
                    "min": 0.0,
                    "percent": 0.9627465303140978
                },
                {
                    "count": 1,
                    "max": 22850.2,
                    "min": 11425.1,
                    "percent": 0.0007304601899196494
                },
                {
                    "count": 0,
                    "max": 34275.3,
                    "min": 22850.2,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 45700.4,
                    "min": 34275.3,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 57125.5,
                    "min": 45700.4,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 68550.6,
                    "min": 57125.5,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 79975.7,
                    "min": 68550.6,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 91400.8,
                    "min": 79975.7,
                    "percent": 0.0
                },
                {
                    "count": 0,
                    "max": 102825.90000000001,
                    "min": 91400.8,
                    "percent": 0.0
                },
                {
                    "count": 50,
                    "max": 114251.0,
                    "min": 102825.90000000001,
                    "percent": 0.036523009495982466
                }
            ],
            "avg. area": 5411.624543462382,
            "pixel distribution": {
                "car": [
                    13655,
                    0.0018431496518735067
                ],
                "cyclist": [
                    939005,
                    0.12674674030446592
                ],
                "head": [
                    0,
                    0.0
                ],
                "ignore": [
                    5501200,
                    0.7425510702956085
                ],
                "left_hand": [
                    0,
                    0.0
                ],
                "person": [
                    954654,
                    0.12885903974805205
                ],
                "right_hand": [
                    0,
                    0.0
                ],
                "road_arrows": [
                    0,
                    0.0
                ],
                "traffic_sign": [
                    0,
                    0.0
                ]
            }
        }
    },
    "annotations by type": {
        "bbox": {
            "count": 548
        },
        "caption": {
            "count": 0
        },
        "label": {
            "count": 0
        },
        "mask": {
            "count": 0
        },
        "points": {
            "count": 669
        },
        "polygon": {
            "count": 821
        },
        "polyline": {
            "count": 0
        }
    },
    "annotations count": 2038,
    "dataset": {
        "image mean": [
            107.06903686941979,
            79.12831698580979,
            52.95829558185416
        ],
        "image std": [
            49.40237673503467,
            43.29600731496902,
            35.47373007603151
        ],
        "images count": 100
    },
    "images count": 100,
    "subsets": {},
    "unannotated images": [
        "img00051",
        "img00052",
        "img00053",
        "img00054",
        "img00055",
    ],
    "unannotated images count": 5,
    "unique images count": 97,
    "repeating images count": 3,
    "repeating images": [
        [("img00057", "default"), ("img00058", "default")],
        [("img00059", "default"), ("img00060", "default")],
        [("img00061", "default"), ("img00062", "default")],
    ],
}
```

</details>


### Validate project annotations <a id="validate"></a>

This command inspects annotations with respect to the task type
and stores the result in JSON file.

The task types supported are `classification`, `detection`, and `segmentation`.

The validation result contains
- `annotation statistics` based on the task type
- `validation reports`, such as
  - items not having annotations
  - items having undefined annotations
  - imbalanced distribution in class/attributes
  - too small or large values
- `summary`

Usage:
- There are five configurable parameters for validation
  - `few_samples_thr` : threshold for giving a warning for minimum number of
    samples per class
  - `imbalance_ratio_thr` : threshold for giving imbalance data warning
  - `far_from_mean_thr` : threshold for giving a warning that data is far
    from mean
  - `dominance_ratio_thr` : threshold for giving a warning bounding box
    imbalance
  - `topk_bins` : ratio of bins with the highest number of data to total bins
    in the histogram

``` bash
datum validate --help

datum validate -p <project dir> -t <task_type> -- \
    -fs <few_samples_thr> \
    -ir <imbalance_ratio_thr> \
    -m <far_from_mean_thr> \
    -dr <dominance_ratio_thr> \
    -k <topk_bins>
```

Example : give warning when imbalance ratio of data with classification task
over 40

``` bash
datum validate -p prj-cls -t classification -- \
    -ir 40
```

Here is the list of validation items(a.k.a. anomaly types).

| Anomaly Type | Description | Task Type |
| ------------ | ----------- | --------- |
| MissingLabelCategories | Metadata (ex. LabelCategories) should be defined | common |
| MissingAnnotation | No annotation found for an Item | common |
| MissingAttribute  | An attribute key is missing for an Item | common |
| MultiLabelAnnotations | Item needs a single label | classification |
| UndefinedLabel     | A label not defined in the metadata is found for an item | common |
| UndefinedAttribute | An attribute not defined in the metadata is found for an item | common |
| LabelDefinedButNotFound     | A label is defined, but not found actually | common |
| AttributeDefinedButNotFound | An attribute is defined, but not found actually | common |
| OnlyOneLabel          | The dataset consists of only label | common |
| OnlyOneAttributeValue | The dataset consists of only attribute value | common |
| FewSamplesInLabel     | The number of samples in a label might be too low | common |
| FewSamplesInAttribute | The number of samples in an attribute might be too low | common |
| ImbalancedLabels    | There is an imbalance in the label distribution | common |
| ImbalancedAttribute | There is an imbalance in the attribute distribution | common |
| ImbalancedDistInLabel     | Values (ex. bbox width) are not evenly distributed for a label | detection, segmentation |
| ImbalancedDistInAttribute | Values (ex. bbox width) are not evenly distributed for an attribute | detection, segmentation |
| NegativeLength | The width or height of bounding box is negative | detection |
| InvalidValue | There's invalid (ex. inf, nan) value for bounding box info. | detection |
| FarFromLabelMean | An annotation has an too small or large value than average for a label | detection, segmentation |
| FarFromAttrMean  | An annotation has an too small or large value than average for an attribute | detection, segmentation |


Validation Result Format:

<details>

``` bash
{
    'statistics': {
        ## common statistics
        'label_distribution': {
            'defined_labels': <dict>,   # <label:str>: <count:int>
            'undefined_labels': <dict>
            # <label:str>: {
            #     'count': <int>,
            #     'items_with_undefined_label': [<item_key>, ]
            # }
        },
        'attribute_distribution': {
            'defined_attributes': <dict>,
            # <label:str>: {
            #     <attribute:str>: {
            #         'distribution': {<attr_value:str>: <count:int>, },
            #         'items_missing_attribute': [<item_key>, ]
            #     }
            # }
            'undefined_attributes': <dict>
            # <label:str>: {
            #     <attribute:str>: {
            #         'distribution': {<attr_value:str>: <count:int>, },
            #         'items_with_undefined_attr': [<item_key>, ]
            #     }
            # }
        },
        'total_ann_count': <int>,
        'items_missing_annotation': <list>, # [<item_key>, ]

        ## statistics for classification task
        'items_with_multiple_labels': <list>, # [<item_key>, ]

        ## statistics for detection task
        'items_with_invalid_value': <dict>,
        # '<item_key>': {<ann_id:int>: [ <property:str>, ], }
        # - properties: 'x', 'y', 'width', 'height',
        #               'area(wxh)', 'ratio(w/h)', 'short', 'long'
        # - 'short' is min(w,h) and 'long' is max(w,h).
        'items_with_negative_length': <dict>,
        # '<item_key>': { <ann_id:int>: { <'width'|'height'>: <value>, }, }
        'bbox_distribution_in_label': <dict>, # <label:str>: <bbox_template>
        'bbox_distribution_in_attribute': <dict>,
        # <label:str>: {<attribute:str>: { <attr_value>: <bbox_template>, }, }
        'bbox_distribution_in_dataset_item': <dict>,
        # '<item_key>': <bbox count:int>

        ## statistics for segmentation task
        'items_with_invalid_value': <dict>,
        # '<item_key>': {<ann_id:int>: [ <property:str>, ], }
        # - properties: 'area', 'width', 'height'
        'mask_distribution_in_label': <dict>, # <label:str>: <mask_template>
        'mask_distribution_in_attribute': <dict>,
        # <label:str>: {
        #     <attribute:str>: { <attr_value>: <mask_template>, }
        # }
        'mask_distribution_in_dataset_item': <dict>,
        # '<item_key>': <mask/polygon count: int>
    },
    'validation_reports': <list>, # [ <validation_error_format>, ]
    # validation_error_format = {
    #     'anomaly_type': <str>,
    #     'description': <str>,
    #     'severity': <str>, # 'warning' or 'error'
    #     'item_id': <str>,  # optional, when it is related to a DatasetItem
    #     'subset': <str>,   # optional, when it is related to a DatasetItem
    # }
    'summary': {
        'errors': <count: int>,
        'warnings': <count: int>
    }
}

```

`item_key` is defined as,
``` python
item_key = (<DatasetItem.id:str>, <DatasetItem.subset:str>)
```

`bbox_template` and `mask_template` are defined as,

``` python
bbox_template = {
    'width': <numerical_stat_template>,
    'height': <numerical_stat_template>,
    'area(wxh)': <numerical_stat_template>,
    'ratio(w/h)': <numerical_stat_template>,
    'short': <numerical_stat_template>, # short = min(w, h)
    'long': <numerical_stat_template>   # long = max(w, h)
}
mask_template = {
    'area': <numerical_stat_template>,
    'width': <numerical_stat_template>,
    'height': <numerical_stat_template>
}
```

`numerical_stat_template` is defined as,

``` python
numerical_stat_template = {
    'items_far_from_mean': <dict>,
    # {'<item_key>': {<ann_id:int>: <value:float>, }, }
    'mean': <float>,
    'stdev': <float>,
    'min': <float>,
    'max': <float>,
    'median': <float>,
    'histogram': {
        'bins': <list>,   # [<float>, ]
        'counts': <list>, # [<int>, ]
    }
}
```

</details>


### Commit <a id="commit"></a>

This command allows to fix current state of a project and create a new
project revision.

Usage:

```bash
datum commit --help

datum commit \
    -m "Commit message"
```

### Checkout <a id="checkout"></a>

This command allows to restore a specific project revision in the project
tree or to restore states of specific sources.

Usage:

```bash
datum checkout --help

datum checkout <commit_hash>
datum checkout -- <source_name1> <source_name2> ...
datum checkout <commit_hash> -- <source_name1> <source_name2> ...
```

### Status <a id="status"></a>

This command prints the summary of the changes between the working tree
of a project and its HEAD revision.

Usage:

```bash
datum status --help

datum status
```

### Log <a id="log"></a>

This command prints the history of the current project revision.

Usage:

```bash
datum log --help

datum log
```

### Register model <a id="model-add"></a>

Supported models:
- OpenVINO
- Custom models via custom `launchers`

Usage:

``` bash
datum model add --help
```

Example: register an OpenVINO model

A model consists of a graph description and weights. There is also a script
used to convert model outputs to internal data structures.

``` bash
datum create
datum model add \
    -n <model_name> -l open_vino -- \
    -d <path_to_xml> -w <path_to_bin> -i <path_to_interpretation_script>
```

Interpretation script for an OpenVINO detection model (`convert.py`):
You can find OpenVINO model interpreter samples in
`datumaro/plugins/openvino/samples` ([instruction](datumaro/plugins/openvino/README.md)).

``` python
from datumaro.components.extractor import *

max_det = 10
conf_thresh = 0.1

def process_outputs(inputs, outputs):
    # inputs = model input, array or images, shape = (N, C, H, W)
    # outputs = model output, shape = (N, 1, K, 7)
    # results = conversion result, [ [ Annotation, ... ], ... ]
    results = []
    for input, output in zip(inputs, outputs):
        input_height, input_width = input.shape[:2]
        detections = output[0]
        image_results = []
        for i, det in enumerate(detections):
            label = int(det[1])
            conf = float(det[2])
            if conf <= conf_thresh:
                continue

            x = max(int(det[3] * input_width), 0)
            y = max(int(det[4] * input_height), 0)
            w = min(int(det[5] * input_width - x), input_width)
            h = min(int(det[6] * input_height - y), input_height)
            image_results.append(Bbox(x, y, w, h,
                label=label, attributes={'score': conf} ))

            results.append(image_results[:max_det])

    return results

def get_categories():
    # Optionally, provide output categories - label map etc.
    # Example:
    label_categories = LabelCategories()
    label_categories.add('person')
    label_categories.add('car')
    return { AnnotationType.label: label_categories }
```

### Run model <a id="model-run"></a>

This command applies model to dataset images and produces a new project.

Usage:

``` bash
datum model run --help

datum model run \
    -p <project dir> \
    -m <model_name> \
    -o <save_dir>
```

Example: launch inference on a dataset

``` bash
datum import <...>
datum model add mymodel <...>
datum model run -m mymodel -o inference
```

### Compare datasets <a id="diff"></a>

The command compares two datasets and saves the results in the
specified directory. The current project is considered to be
"ground truth".

``` bash
datum diff --help

datum diff <other_project_dir> -o <save_dir>
```

Example: compare a dataset with model inference

``` bash
datum import <...>
datum model add mymodel <...>
datum transform <...> -o inference
datum diff inference -o diff
```

### Explain inference <a id="explain"></a>

Runs an explainable AI algorithm for a model.

This tool is supposed to help an AI developer to debug a model and a dataset.
Basically, it executes inference and tries to find problems in the trained
model - determine decision boundaries and belief intervals for the classifier.

Currently, the only available algorithm is RISE ([article](https://arxiv.org/pdf/1806.07421.pdf)),
which runs inference and then re-runs a model multiple times on each
image to produce a heatmap of activations for each output of the
first inference. As a result, we obtain few heatmaps, which
shows, how image pixels affected the inference result. This algorithm doesn't
require any special information about the model, but it requires the model to
return all the outputs and confidences. The algorighm only supports
classification and detection models.

The following use cases available:
- RISE for classification
- RISE for object detection

Usage:

``` bash
datum explain --help

datum explain \
    -m <model_name> \
    -o <save_dir> \
    -t <target> \
    <method> \
    <method_params>
```

Example: run inference explanation on a single image with visualization

``` bash
datum create <...>
datum model add mymodel <...>
datum explain -t image.png -m mymodel \
    rise --max-samples 1000 --progressive
```

> Note: this algorithm requires the model to return
> _all_ (or a _reasonable_ amount) the outputs and confidences unfiltered,
> i.e. all the `Label` annotations for classification models and
> all the `Bbox`es for detection models.
> You can find examples of the expected model outputs in [`tests/test_RISE.py`](../tests/test_RISE.py)

For OpenVINO models the output processing script would look like this:

Classification scenario:

``` python
from datumaro.components.extractor import *
from datumaro.util.annotation_util import softmax

def process_outputs(inputs, outputs):
    # inputs = model input, array or images, shape = (N, C, H, W)
    # outputs = model output, logits, shape = (N, n_classes)
    # results = conversion result, [ [ Annotation, ... ], ... ]
    results = []
    for input, output in zip(inputs, outputs):
        input_height, input_width = input.shape[:2]
        confs = softmax(output[0])
        for label, conf in enumerate(confs):
            results.append(Label(int(label)), attributes={'score': float(conf)})

    return results
```


Object Detection scenario:

``` python
from datumaro.components.extractor import *

# return a significant number of output boxes to make multiple runs
# statistically correct and meaningful
max_det = 1000

def process_outputs(inputs, outputs):
    # inputs = model input, array or images, shape = (N, C, H, W)
    # outputs = model output, shape = (N, 1, K, 7)
    # results = conversion result, [ [ Annotation, ... ], ... ]
    results = []
    for input, output in zip(inputs, outputs):
        input_height, input_width = input.shape[:2]
        detections = output[0]
        image_results = []
        for i, det in enumerate(detections):
            label = int(det[1])
            conf = float(det[2])
            x = max(int(det[3] * input_width), 0)
            y = max(int(det[4] * input_height), 0)
            w = min(int(det[5] * input_width - x), input_width)
            h = min(int(det[6] * input_height - y), input_height)
            image_results.append(Bbox(x, y, w, h,
                label=label, attributes={'score': conf} ))

            results.append(image_results[:max_det])

    return results
```

### Transform project <a id="transform"></a>

This command allows to modify images or annotations in a project all at once.

Note that this command is designed for batch processing and if you only
need to modify few elements of a dataset, you might want to use
other approaches for better performance.

``` bash
datum transform --help

datum transform \
    -p <project_dir> \
    -t <transform_name> \
    -- [extra transform options]
```

Example: split a dataset randomly to `train` and `test` subsets, ratio is 2:1

``` bash
datum transform -t random_split -- --subset train:.67 --subset test:.33
```

Example: split a dataset in task-specific manner. The tasks supported are
classification, detection, segmentation and re-identification.

``` bash
datum transform -t split -- \
    -t classification --subset train:.5 --subset val:.2 --subset test:.3

datum transform -t split -- \
    -t detection --subset train:.5 --subset val:.2 --subset test:.3

datum transform -t split -- \
    -t segmentation --subset train:.5 --subset val:.2 --subset test:.3

datum transform -t split -- \
    -t reid --subset train:.5 --subset val:.2 --subset test:.3 \
    --query .5
```

Example: convert polygons to masks, masks to boxes etc.:

``` bash
datum transform -t boxes_to_masks
datum transform -t masks_to_polygons
datum transform -t polygons_to_masks
datum transform -t shapes_to_boxes
```

Example: remap dataset labels, `person` to `car` and `cat` to `dog`,
keep `bus`, remove others

``` bash
datum transform -t remap_labels -- \
    -l person:car -l bus:bus -l cat:dog \
    --default delete
```

Example: rename dataset items by a regular expression
- Replace `pattern` with `replacement`
- Remove `frame_` from item ids

``` bash
datum transform -t rename -- -e '|pattern|replacement|'
datum transform -t rename -- -e '|frame_(\d+)|\\1|'
```

Example: sampling dataset items as many as the number of target samples with
sampling method entered by the user, divide into `sampled` and `unsampled`
subsets
- There are five methods of sampling the m option.
  - `topk`: Return the k with high uncertainty data
  - `lowk`: Return the k with low uncertainty data
  - `randk`: Return the random k data
  - `mixk`: Return half to topk method and the rest to lowk method
  - `randtopk`: First, select 3 times the number of k randomly, and return
  the topk among them.

``` bash
datum transform -t sampler -- \
    -a entropy \
    -i train \
    -o sampled \
    -u unsampled \
    -m topk \
    -k 20
```

Example : control number of outputs to 100 after NDR
- There are two methods in NDR e option
  - `random`: sample from removed data randomly
  - `similarity`: sample from removed data with ascending
- There are two methods in NDR u option
  - `uniform`: sample data with uniform distribution
  - `inverse`: sample data with reciprocal of the number

```bash
datum transform -t ndr -- \
    -w train \
    -a gradient \
    -k 100 \
    -e random \
    -u uniform
```

## Extending

There are few ways to extend and customize Datumaro behaviour, which is
supported by plugins. Check [our contribution guide](../CONTRIBUTING.md) for
details on plugin implementation. In general, a plugin is a Python code file.
It must be put into a plugin directory:
- `<project_dir>/.datumaro/plugins` for project-specific plugins
- `<datumaro_dir>/plugins` for global plugins

### Built-in plugins <a id="builtin-plugins"></a>

Datumaro provides several builtin plugins. Plugins can have dependencies,
which need to be installed separately.

#### TensorFlow

The plugin provides support of TensorFlow Detection API format, which includes
boxes and masks.

**Dependencies**

The plugin depends on TensorFlow, which can be installed with `pip`:

```bash
pip install tensorflow
# or
pip install tensorflow-gpu
# or
pip install datumaro[tf]
# or
pip install datumaro[tf-gpu]
```

#### Accuracy Checker

This plugin allows to use [Accuracy Checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker)
to launch deep learning models from various frameworks
(Caffe, MxNet, PyTorch, OpenVINO, ...) through Accuracy Checker's API.

**Dependencies**

The plugin depends on Accuracy Checker, which can be installed with `pip`:

```bash
pip install 'git+https://github.com/openvinotoolkit/open_model_zoo.git#subdirectory=tools/accuracy_checker'
```

To execute models with deep learning frameworks, they need to be installed too.

#### OpenVINO™

This plugin provides support for model inference with [OpenVINO™](https://01.org/openvinotoolkit).

**Dependencies**

The plugin depends on the OpenVINO™ Toolkit, which can be installed by
following [these instructions](https://docs.openvinotoolkit.org/latest/index.html#packaging_and_deployment)

### Dataset Formats

Dataset reading is supported by Extractors and Importers.
An Extractor produces a list of dataset items corresponding
to the dataset. An Importer creates a project from the data source location.
It is possible to add custom Extractors and Importers. To do this, you need
to put an Extractor and Importer implementation scripts to a plugin directory.

Dataset writing is supported by Converters.
A Converter produces a dataset of a specific format from dataset items.
It is possible to add custom Converters. To do this, you need to put a Converter
implementation script to a plugin directory.

### Dataset Conversions ("Transforms")

A Transform is a function for altering a dataset and producing a new one.
It can update dataset items, annotations, classes, and other properties.
A list of available transforms for dataset conversions can be extended by
adding a Transform implementation script into a plugin directory.

### Model launchers

A list of available launchers for model execution can be extended by adding
a Launcher implementation script into a plugin directory.

## Links
- [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- [How to convert model to OpenVINO format](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)
- [Model conversion script example](https://github.com/opencv/cvat/blob/3e09503ba6c6daa6469a6c4d275a5a8b168dfa2c/components/tf_annotation/install.sh#L23)
