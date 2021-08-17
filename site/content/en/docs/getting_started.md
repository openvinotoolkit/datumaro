---
title: 'Getting started'
linkTitle: 'Getting started'
description: ''
no_list: true
weight: 1
tags: ['Basics',  'Examples for python module',
    'Examples for standalone tool', 'CVAT', 'OpenVINO™', 'PASCAL VOC',
    'MS COCO', ]
---

To read about the design concept and features of Datumaro, go to the [design section](/docs/design/).

## Installation

### Dependencies

- Python (3.6+)
- Optional: OpenVINO, TensorFlow, PyTorch, MxNet, Caffe, Accuracy Checker

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

> Check our [developer guide](/docs/developer-guide/) for additional
  information.

## Examples

<!--lint disable list-item-indent-->
<!--lint disable list-item-bullet-indent-->

- Convert PASCAL VOC dataset to COCO format, keep only images with `cat` class
  presented:
  ```bash
  # Download VOC dataset:
  # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  datum convert --input-format voc --input-path <path/to/voc> \
                --output-format coco \
                --filter '/item[annotation/label="cat"]' \
                -- --reindex 1 # avoid annotation id conflicts
  ```

- Convert only non-`occluded` annotations from a
  [CVAT](https://github.com/openvinotoolkit/cvat) project to TFrecord:
  ```bash
  # export Datumaro dataset in CVAT UI, extract somewhere, go to the project dir
  datum filter -e '/item/annotation[occluded="False"]' \
    --mode items+anno --output-dir not_occluded
  datum export --project not_occluded \
    --format tf_detection_api -- --save-images
  ```

- Annotate MS COCO dataset, extract image subset, re-annotate it in
  [CVAT](https://github.com/openvinotoolkit/cvat), update old dataset:
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

- Annotate instance polygons in [CVAT](https://github.com/openvinotoolkit/cvat), export
  as masks in COCO:
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
