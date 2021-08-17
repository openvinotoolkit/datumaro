---
title: 'Convert datasets'
linkTitle: 'Convert datasets'
description: ''
weight: 8
tags: [  'Examples for standalone tool', 'MS COCO', ]
---

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
