---
title: 'Run inference'
linkTitle: 'Run model'
description: ''
weight: 21
tags: [ 'Models',  'Examples for standalone tool', ]
---

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
