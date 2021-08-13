# Dataset Management Framework (Datumaro)

[![Build status](https://github.com/openvinotoolkit/datumaro/actions/workflows/health_check.yml/badge.svg)](https://github.com/openvinotoolkit/datumaro/actions/workflows/health_check.yml)
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

- [Getting started](https://openvinotoolkit.github.io/datumaro/getting_started)
- [Examples](https://openvinotoolkit.github.io/datumaro/examples)
- [Features](#features)
- [User manual](https://openvinotoolkit.github.io/datumaro/docs/user-manual)
- [Contributing](#contributing)

## Features

[(Back to top)](#dataset-management-framework-datumaro)

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
  - [Kitti](http://www.cvlibs.net/datasets/kitti/index.php) (`segmentation`, `detection`, `3D raw` / `velodyne points`)
  - [Supervisely](https://docs.supervise.ly/data-organization/00_ann_format_navi) (`point cloud`)
  - [CVAT](https://github.com/openvinotoolkit/cvat/blob/develop/cvat/apps/documentation/xml_format.md)
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
    - RISE for classification
    - RISE for object detection

> Check [the design document](https://openvinotoolkit.github.io/datumaro/docs/design) for a full list of features.
> Check [the user manual](https://openvinotoolkit.github.io/datumaro/docs/user-manual) for usage instructions.

## Contributing

[(Back to top)](#dataset-management-framework-datumaro)

Feel free to
[open an Issue](https://github.com/openvinotoolkit/datumaro/issues/new), if you
think something needs to be changed. You are welcome to participate in
development, instructions are available in our
[contribution guide](https://openvinotoolkit.github.io/datumaro/contribution-guide/).
