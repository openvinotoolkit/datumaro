---
title: 'Import project'
linkTitle: 'Import project'
description: ''
weight: 14
---

This command creates a Project from an existing dataset.

Supported formats are listed in the command help. Check [extending tips](/docs/user-manual/extending/)
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
