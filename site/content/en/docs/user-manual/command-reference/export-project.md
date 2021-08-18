---
title: 'Export project'
linkTitle: 'Export project'
description: ''
weight: 15
---

This command exports a Project as a dataset in some format.

Supported formats are listed in the command help. Check [extending tips](/docs/user-manual/extending/)
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
