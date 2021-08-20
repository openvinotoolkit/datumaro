---
title: 'Obtaining project info'
linkTitle: 'Get project info'
description: ''
weight: 17
---

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
