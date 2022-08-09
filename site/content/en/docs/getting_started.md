---
title: 'Getting started'
linkTitle: 'Getting started'
description: ''
no_list: true
weight: 1
---

To read about the design concept and features of Datumaro, go to the [design section](/docs/design/).

## Installation

### Dependencies

- Python (3.7+)
- Optional: OpenVINO, TensorFlow, PyTorch, MxNet, Caffe, Accuracy Checker, Git

Optionally, create a virtual environment:

``` bash
python -m pip install virtualenv
python -m virtualenv venv
. venv/bin/activate
```

Install Datumaro package:

``` bash
pip install datumaro[default]
```

Read full installation instructions in [the user manual](/docs/user-manual/installation).

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
import datumaro as dm

dataset = dm.Dataset.import_from('path/', 'voc')

# keep only annotated images
dataset.select(lambda item: len(item.annotations) != 0)

# change dataset labels and corresponding annotations
dataset.transform('remap_labels',
    mapping={
      'cat': 'dog', # rename cat to dog
      'truck': 'car', # rename truck to car
      'person': '', # remove this label
    },
    default='delete') # remove everything else

# iterate over the dataset elements
for item in dataset:
    print(item.id, item.annotations)

# export the resulting dataset in COCO format
dataset.export('dst/dir', 'coco', save_images=True)
```

[List of components](/api/api/datumaro.html) with the comfortable importing.

> Check our [developer manual](/api/api/developer_manual.html) for additional
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
  [CVAT](https://github.com/cvat-ai/cvat) project to TFrecord:
  ```bash
  # export Datumaro dataset in CVAT UI, extract somewhere, go to the project dir
  datum filter -e '/item/annotation[occluded="False"]' --mode items+anno
  datum export --format tf_detection_api -- --save-images
  ```

- Annotate MS COCO dataset, extract image subset, re-annotate it in
  [CVAT](https://github.com/cvat-ai/cvat), update old dataset:
  ```bash
  # Download COCO dataset http://cocodataset.org/#download
  # Put images to coco/images/ and annotations to coco/annotations/
  datum create
  datum import --format coco <path/to/coco>
  datum export --filter '/image[images_I_dont_like]' --format cvat
  # import dataset and images to CVAT, re-annotate
  # export Datumaro project, extract to 'reannotation-upd'
  datum project update reannotation-upd
  datum export --format coco
  ```

- Annotate instance polygons in
  [CVAT](https://github.com/cvat-ai/cvat), export as masks in COCO:
  ```bash
  datum convert --input-format cvat --input-path <path/to/cvat.xml> \
                --output-format coco -- --segmentation-mode masks
  ```

- Apply an OpenVINO detection model to some COCO-like dataset,
  then compare annotations with ground truth and visualize in TensorBoard:
  ```bash
  datum create
  datum import --format coco <path/to/coco>
  # create model results interpretation script
  datum model add -n mymodel openvino \
    --weights model.bin --description model.xml \
    --interpretation-script parse_results.py
  datum model run --model -n mymodel --output-dir mymodel_inference/
  datum diff mymodel_inference/ --format tensorboard --output-dir diff
  ```

- Change colors in PASCAL VOC-like `.png` masks:
  ```bash
  datum create
  datum import --format voc <path/to/voc/dataset>

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
  import datumaro as dm

  dataset = dm.Dataset([
    dm.DatasetItem(id='image1', subset='train',
      image=np.ones((5, 5, 3)),
      annotations=[
        dm.Bbox(1, 2, 3, 4, label=0),
      ]
    ),
    # ...
  ], categories=['cat', 'dog'])
  dataset.export('test_dataset/', 'coco')
  ```

<!--lint enable list-item-bullet-indent-->
<!--lint enable list-item-indent-->
