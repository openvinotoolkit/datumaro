---
title: 'Add and remove data'
linkTitle: 'Add and remove data'
description: ''
weight: 11
tags: [  'Examples for standalone tool', 'MS COCO', ]
---

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

Supported formats are listed in the command help. Check [extending tips](/docs/user-manual/extending/)
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
