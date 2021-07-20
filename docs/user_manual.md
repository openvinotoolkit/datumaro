# User manual

## Contents

- [Installation](#installation)
- [Interfaces](#interfaces)
- [Supported dataset formats and annotations](#supported-formats)
- [Supported data formats](#data-formats)
- [Command line workflow](#command-line-workflow)
  - [Project structure](#project-structure)
- [Command reference](#command-reference)
  - [Convert datasets](#convert-datasets)
  - [Create project](#create-project)
  - [Add and remove data](#add-and-remove-data)
  - [Import project](#import-project)
  - [Filter project](#filter-project)
  - [Update project (merge)](#update-project)
  - [Merge projects](#merge-projects)
  - [Export project](#export-project)
  - [Compare projects](#compare-projects)
  - [Obtaining project info](#get-project-info)
  - [Obtaining project statistics](#get-project-statistics)
  - [Validate project annotations](#validate-project-annotations)
  - [Register model](#register-model)
  - [Run inference](#run-model)
  - [Run inference explanation](#explain-inference)
  - [Transform project](#transform-project)
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

> You can change the installation branch with `...@<branch_name>`
> Also use `--force-reinstall` parameter in this case.

## Interfaces

As a standalone tool:

``` bash
datum --help
```

As a python module:
> The directory containing Datumaro should be in the `PYTHONPATH`
> environment variable or `cvat/datumaro/` should be the current directory.

``` bash
python -m datumaro --help
python datumaro/ --help
python datum.py --help
```

As a python library:

``` python
import datumaro
```

## Supported Formats

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

List of supported annotation types:
- Labels
- Bounding boxes
- Polygons
- Polylines
- (Segmentation) Masks
- (Key-)Points
- Captions

## Data formats

Datumaro only works with 2d RGB(A) images.

To create an unlabelled dataset from an arbitrary directory with images use
`ImageDir` format:

```bash
datum create -o <project/dir>
datum add path -p <project/dir> -f image_dir <directory/path/>
```

or if you work with Datumaro API:

For using with a project:

```python
from datumaro.components.project import Project

project = Project()
project.add_source('source1', {
  'format': 'image_dir',
  'url': 'directory/path/'
})
dataset = project.make_dataset()
```

And for using as a dataset:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('directory/path/', 'image_dir')
```

This will search for images in the directory recursively and add
them as dataset entries with names like `<subdir1>/<subsubdir1>/<image_name1>`.
The list of formats matches the list of supported image formats in OpenCV.
```
.jpg, .jpeg, .jpe, .jp2, .png, .bmp, .dib, .tif, .tiff, .tga, .webp, .pfm,
.sr, .ras, .exr, .hdr, .pic, .pbm, .pgm, .ppm, .pxm, .pnm
```

After addition into a project, images can be split into subsets and renamed
with transformations, filtered, joined with existing annotations etc.

To use a video as an input, one should either [create an Extractor plugin](../docs/developer_guide.md#plugins),
which splits a video into frames, or split the video manually and import images.

## Command line workflow

The key object is a project, so most CLI commands operate on projects.
However, there are few commands operating on datasets directly.
A project is a combination of a project's own dataset, a number of
external data sources and an environment.
An empty Project can be created by `project create` command,
an existing dataset can be imported with `project import` command.
A typical way to obtain projects is to export tasks in CVAT UI.

If you want to interact with models, you need to add them to project first.

### Project structure

<!--lint disable fenced-code-flag-->
```
└── project/
    ├── .datumaro/
    |   ├── config.yml
    │   ├── .git/
    │   ├── models/
    │   └── plugins/
    │       ├── plugin1/
    │       |   ├── file1.py
    │       |   └── file2.py
    │       ├── plugin2.py
    │       ├── custom_extractor1.py
    │       └── ...
    ├── dataset/
    └── sources/
        ├── source1
        └── ...
```
<!--lint enable fenced-code-flag-->

## Command reference

> **Note**: command invocation syntax is subject to change,
> **always refer to command --help output**

Available CLI commands:
![CLI design doc](images/cli_design.png)

### Convert datasets

This command allows to convert a dataset from one format into another.
In fact, this command is a combination of `project import` and `project export`
and just provides a simpler way to obtain the same result when no extra options
is needed. A list of supported formats can be found in the `--help` output of
this command.

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
              --output-format coco
```

### Import project

This command creates a Project from an existing dataset.

Supported formats are listed in the command help. Check [extending tips](#extending)
for information on extra format support.

Usage:

``` bash
datum import --help

datum import \
    -i <dataset_path> \
    -o <project_dir> \
    -f <format>
```

Example: create a project from COCO-like dataset

``` bash
datum import \
    -i /home/coco_dir \
    -o /home/project_dir \
    -f coco
```

An _MS COCO_-like dataset should have the following directory structure:

<!--lint disable fenced-code-flag-->
```
COCO/
├── annotations/
│   ├── instances_val2017.json
│   ├── instances_train2017.json
├── images/
│   ├── val2017
│   ├── train2017
```
<!--lint enable fenced-code-flag-->

Everything after the last `_` is considered a subset name in the COCO format.

### Create project

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

### Add and remove data

A Project can contain a number of external Data Sources. Each Data Source
describes a way to produce dataset items. A Project combines dataset items from
all the sources and its own dataset into one composite dataset. You can manage
project sources by commands in the `source` command line context.

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

Supported formats are listed in the command help. Check [extending tips](#extending)
for information on extra format support.

Usage:

``` bash
datum add --help
datum remove --help

datum add \
    path <path> \
    -p <project dir> \
    -f <format> \
    -n <name>

datum remove \
    -p <project dir> \
    -n <name>
```

Example: create a project from a bunch of different annotations and images,
and generate TFrecord for TF Detection API for model training

``` bash
datum create
# 'default' is the name of the subset below
datum add path <path/to/coco/instances_default.json> -f coco_instances
datum add path <path/to/cvat/default.xml> -f cvat
datum add path <path/to/voc> -f voc_detection
datum add path <path/to/datumaro/default.json> -f datumaro
datum add path <path/to/images/dir> -f image_dir
datum export -f tf_detection_api
```

### Filter project

This command allows to create a sub-Project from a Project. The new project
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

### Update project

This command updates items in a project from another one
(check [Merge Projects](#merge-projects) for complex merging).

Usage:

``` bash
datum merge --help

datum merge \
    -p <project dir> \
    -o <output dir> \
    <other project dir>
```

Example: update annotations in the `first_project` with annotations
from the `second_project` and save the result as `merged_project`

``` bash
datum merge \
    -p first_project \
    -o merged_project \
    second_project
```

### Merge projects

This command merges items from 2 or more projects and checks annotations for
errors.

Spatial annotations are compared by distance and intersected, labels and
attributes are selected by voting.
Merge conflicts, missing items and annotations, other errors are saved into a `.json` file.

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

### Export project

This command exports a Project as a dataset in some format.

Supported formats are listed in the command help. Check [extending tips](#extending)
for information on extra format support.

Usage:

``` bash
datum export --help

datum export \
    -p <project dir> \
    -o <output dir> \
    -f <format> \
    -- [additional format parameters]
```

Example: save project as VOC-like dataset, include images, convert images to `PNG`

``` bash
datum export \
    -p test_project \
    -o test_project-export \
    -f voc \
    -- --save-images --image-ext='.png'
```

### Get project info

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

### Get project statistics

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


### Validate project annotations

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

### Register model

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

### Run model

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

### Compare projects

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

### Explain inference

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
    <target> \
    <method> \
    <method_params>
```

Example: run inference explanation on a single image with visualization

``` bash
datum create <...>
datum model add mymodel <...>
datum explain image.png -m mymodel \
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

def get_categories():
    # Optionally, provide output categories - label map etc.
    # Example:
    label_categories = LabelCategories()
    label_categories.add('person')
    label_categories.add('car')
    return { AnnotationType.label: label_categories }
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


### Transform Project

This command allows to modify images or annotations in a project all at once.

``` bash
datum transform --help

datum transform \
    -p <project_dir> \
    -o <output_dir> \
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

### Built-in plugins

Datumaro provides several builtin plugins. Plugins can have dependencies,
which need to be installed separately.

#### TensorFlow

The plugin provides support of TensorFlow Detection API format, which includes
boxes and masks. It depends on TensorFlow, which can be installed with `pip`:

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
The plugin depends on Accuracy Checker, which can be installed with `pip`:

```bash
pip install 'git+https://github.com/openvinotoolkit/open_model_zoo.git#subdirectory=tools/accuracy_checker'
```

#### OpenVINO™

This plugin provides support for model inference with [OpenVINO™](https://01.org/openvinotoolkit).
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
