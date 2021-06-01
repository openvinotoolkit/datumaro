# Dataset Management Framework (Datumaro)

[![Build status](https://github.com/openvinotoolkit/datumaro/actions/workflows/health_check.yml/badge.svg?branch=develop)](https://github.com/openvinotoolkit/datumaro/actions/workflows/health_check.yml)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/759d2d873b59495aa3d3f8c51b786246)](https://app.codacy.com/gh/openvinotoolkit/datumaro?utm_source=github.com&utm_medium=referral&utm_content=openvinotoolkit/datumaro&utm_campaign=Badge_Grade_Dashboard)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/9511b691ff134e739ea6fc524f7cc760)](https://www.codacy.com/gh/openvinotoolkit/datumaro?utm_source=github.com&utm_medium=referral&utm_content=openvinotoolkit/datumaro&utm_campaign=Badge_Coverage)

A framework and CLI tool to build, transform, and analyze datasets.

<!--lint disable fenced-code-flag-->
```
VOC dataset                                  ---> Annotation tool
     +                                     /
COCO dataset -----> Datumaro ---> dataset ------> Model training
     +                                     \
CVAT annotations                             ---> Publication, statistics etc.
```
<!--lint enable fenced-code-flag-->

# Table of Contents

- [Examples](#examples)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [User manual](docs/user_manual.md)
- [Contributing](#contributing)

## Examples

[(Back to top)](#table-of-contents)

<!--lint disable list-item-indent-->
<!--lint disable list-item-bullet-indent-->

- Convert PASCAL VOC dataset to COCO format, keep only images with `cat` class presented:
  ```bash
  # Download VOC dataset:
  # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  datum convert --input-format voc --input-path <path/to/voc> \
                --output-format coco \
                --filter '/item[annotation/label="cat"]' \
                -- --reindex 1 # avoid annotation id conflicts
  ```

- Convert only non-`occluded` annotations from a [CVAT](https://github.com/opencv/cvat) project to TFrecord:
  ```bash
  # export Datumaro dataset in CVAT UI, extract somewhere, go to the project dir
  datum filter -e '/item/annotation[occluded="False"]' \
    --mode items+anno --output-dir not_occluded
  datum export --project not_occluded \
    --format tf_detection_api -- --save-images
  ```

- Annotate MS COCO dataset, extract image subset, re-annotate it in [CVAT](https://github.com/opencv/cvat), update old dataset:
  ```bash
  # Download COCO dataset http://cocodataset.org/#download
  # Put images to coco/images/ and annotations to coco/annotations/
  datum import --format coco --input-path <path/to/coco>
  datum export --filter '/image[images_I_dont_like]' --format cvat \
    --output-dir reannotation
  # import dataset and images to CVAT, re-annotate
  # export Datumaro project, extract to 'reannotation-upd'
  datum merge reannotation-upd
  datum export --format coco
  ```

- Annotate instance polygons in [CVAT](https://github.com/opencv/cvat), export as masks in COCO:
  ```bash
  datum convert --input-format cvat --input-path <path/to/cvat.xml> \
                --output-format coco -- --segmentation-mode masks
  ```

- Apply an OpenVINO detection model to some COCO-like dataset,
  then compare annotations with ground truth and visualize in TensorBoard:
  ```bash
  datum import --format coco --input-path <path/to/coco>
  # create model results interpretation script
  datum model add mymodel openvino \
    --weights model.bin --description model.xml \
    --interpretation-script parse_results.py
  datum model run --model mymodel --output-dir mymodel_inference/
  datum diff mymodel_inference/ --format tensorboard --output-dir diff
  ```

- Change colors in PASCAL VOC-like `.png` masks:
  ```bash
  datum import --format voc --input-path <path/to/voc/dataset>

  # Create a color map file with desired colors:
  #
  # label : color_rgb : parts : actions
  # cat:0,0,255::
  # dog:255,0,0::
  #
  # Save as mycolormap.txt

  datum export --format voc_segmentation -- --label-map mycolormap.txt
  # add "--apply-colormap=0" to save grayscale (indexed) masks
  # check "--help" option for more info
  # use "datum --loglevel debug" for extra conversion info
  ```

- Create a custom COCO-like dataset:
  ```python
  import numpy as np
  from datumaro.components.extractor import (DatasetItem,
    Bbox, LabelCategories, AnnotationType)
  from datumaro.components.dataset import Dataset

  dataset = Dataset(categories={
    AnnotationType.label: LabelCategories.from_iterable(['cat', 'dog'])
  })
  dataset.put(DatasetItem(id=0, image=np.ones((5, 5, 3)), annotations=[
    Bbox(1, 2, 3, 4, label=0),
  ]))
  dataset.export('test_dataset', 'coco')
  ```

<!--lint enable list-item-bullet-indent-->
<!--lint enable list-item-indent-->

## Features

[(Back to top)](#table-of-contents)

- Dataset reading, writing, conversion in any direction. [Supported formats](docs/user_manual.md#supported-formats):
  - [COCO](http://cocodataset.org/#format-data) (`image_info`, `instances`, `person_keypoints`, `captions`, `labels`, `panoptic`, `stuff`)
  - [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html) (`classification`, `detection`, `segmentation`, `action_classification`, `person_layout`)
  - [YOLO](https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data) (`bboxes`)
  - [TF Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md) (`bboxes`, `masks`)
  - [WIDER Face](http://shuoyang1213.me/WIDERFACE/) (`bboxes`)
  - [VGGFace2](https://github.com/ox-vgg/vgg_face2) (`landmarks`, `bboxes`)
  - [MOT sequences](https://arxiv.org/pdf/1906.04567.pdf)
  - [MOTS PNG](https://www.vision.rwth-aachen.de/page/mots)
  - [ImageNet](http://image-net.org/)
  - [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) (`classification`)
  - [MNIST](http://yann.lecun.com/exdb/mnist/) (`classification`)
  - [MNIST in CSV](https://pjreddie.com/projects/mnist-in-csv/) (`classification`)
  - [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
  - [Cityscapes](https://www.cityscapes-dataset.com/)
  - [CVAT](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md)
  - [LabelMe](http://labelme.csail.mit.edu/Release3.0)
  - [ICDAR13/15](https://rrc.cvc.uab.es/?ch=2) (`word_recognition`, `text_localization`, `text_segmentation`)
  - [Market-1501](https://www.aitribune.com/dataset/2018051063) (`person re-identification`)
  - [LFW](http://vis-www.cs.umass.edu/lfw/) (`classification`, `person re-identification`, `landmarks`)
- Dataset building
  - Merging multiple datasets into one
  - Dataset filtering by a custom criteria:
    - remove polygons of a certain class
    - remove images without annotations of a specific class
    - remove `occluded` annotations from images
    - keep only vertically-oriented images
    - remove small area bounding boxes from annotations
  - Annotation conversions, for instance:
    - polygons to instance masks and vice-versa
    - apply a custom colormap for mask annotations
    - rename or remove dataset labels
  - Splitting a dataset into multiple subsets like `train`, `val`, and `test`:
    - random split
    - task-specific splits based on annotations,
      which keep initial label and attribute distributions
      - for classification task, based on labels
      - for detection task, based on bboxes
      - for re-identification task, based on labels,
        avoiding having same IDs in training and test splits
  - Sampling a dataset
    - analyzes inference result from the given dataset
      and selects the ‘best’ and the ‘least amount of’ samples for annotation.
    - Select the sample that best suits model training.
      - sampling with Entropy based algorithm
- Dataset quality checking
  - Simple checking for errors
  - Comparison with model inference
  - Merging and comparison of multiple datasets
  - Annotation validation based on the task type(classification, etc)
- Dataset comparison
- Dataset statistics (image mean and std, annotation statistics)
- Model integration
  - Inference (OpenVINO, Caffe, PyTorch, TensorFlow, MxNet, etc.)
  - Explainable AI ([RISE algorithm](https://arxiv.org/abs/1806.07421))

> Check [the design document](docs/design.md) for a full list of features.
> Check [the user manual](docs/user_manual.md) for usage instructions.

## Installation

[(Back to top)](#table-of-contents)

### Dependencies

- Python (3.6+)
- Optional: OpenVINO, TensforFlow, PyTorch, MxNet, Caffe, Accuracy Checker

Optionally, create a virtual environment:

``` bash
python -m pip install virtualenv
python -m virtualenv venv
. venv/bin/activate
```

Install Datumaro package:

``` bash
pip install datumaro
```

## Usage

[(Back to top)](#table-of-contents)

There are several options available:
- [A standalone command-line tool](#standalone-tool)
- [A python module](#python-module)

### Standalone tool

Datuaro as a standalone tool allows to do various dataset operations from
the command line interface:

``` bash
datum --help
python -m datumaro --help
```

### Python module

Datumaro can be used in custom scripts as a Python module. Used this way, it
allows to use its features from an existing codebase, enabling dataset
reading, exporting and iteration capabilities, simplifying integration of custom
formats and providing high performance operations:

``` python
from datumaro.components.project import Project

# load a Datumaro project
project = Project.load('directory')

# create a dataset
dataset = project.make_dataset()

# keep only annotated images
dataset.select(lambda item: len(item.annotations) != 0)

# change dataset labels
dataset.transform('remap_labels',
  {'cat': 'dog', # rename cat to dog
    'truck': 'car', # rename truck to car
    'person': '', # remove this label
  }, default='delete') # remove everything else

# iterate over dataset elements
for item in dataset:
  print(item.id, item.annotations)

# export the resulting dataset in COCO format
dataset.export('dst/dir', 'coco')
```

> Check our [developer guide](docs/developer_guide.md) for additional information.

## Contributing

[(Back to top)](#table-of-contents)

Feel free to [open an Issue](https://github.com/openvinotoolkit/datumaro/issues/new), if you
think something needs to be changed. You are welcome to participate in development,
instructions are available in our [contribution guide](CONTRIBUTING.md).
