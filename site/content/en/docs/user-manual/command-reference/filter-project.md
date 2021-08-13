---
title: 'Filter project'
linkTitle: 'Filter project'
description: ''
weight: 12
tags: [  'Examples for standalone tool', ]
---

This command allows to create a sub-Project from a Project. The new project
includes only items satisfying some condition. [XPath](https://devhints.io/xpath)
is used as a query format.

There are several filtering modes available (`-m/--mode` parameter).
Supported modes:
- `i`, `items`
- `a`, `annotations`
- `i+a`, `a+i`, `items+annotations`, `annotations+items`

When filtering annotations, use the `items+annotations`
mode to point that annotation-less dataset items should be
removed. To select an annotation, write an XPath that
returns `annotation` elements (see examples).

Usage:

``` bash
datum filter --help

datum filter \
    -p <project dir> \
    -e '<xpath filter expression>'
```

Example: extract a dataset with only images which `width` < `height`

``` bash
datum filter \
    -p test_project \
    -e '/item[image/width < image/height]'
```

Example: extract a dataset with only images of subset `train`.
``` bash
datum project filter \
    -p test_project \
    -e '/item[subset="train"]'
```

Example: extract a dataset with only large annotations of class `cat` and any
non-`persons`

``` bash
datum filter \
    -p test_project \
    --mode annotations -e '/item/annotation[(label="cat" and area > 99.5) or label!="person"]'
```

Example: extract a dataset with only occluded annotations, remove empty images

``` bash
datum filter \
    -p test_project \
    -m i+a -e '/item/annotation[occluded="True"]'
```

Item representations are available with `--dry-run` parameter:

``` xml
<item>
  <id>290768</id>
  <subset>minival2014</subset>
  <image>
    <width>612</width>
    <height>612</height>
    <depth>3</depth>
  </image>
  <annotation>
    <id>80154</id>
    <type>bbox</type>
    <label_id>39</label_id>
    <x>264.59</x>
    <y>150.25</y>
    <w>11.199999999999989</w>
    <h>42.31</h>
    <area>473.87199999999956</area>
  </annotation>
  <annotation>
    <id>669839</id>
    <type>bbox</type>
    <label_id>41</label_id>
    <x>163.58</x>
    <y>191.75</y>
    <w>76.98999999999998</w>
    <h>73.63</h>
    <area>5668.773699999998</area>
  </annotation>
  ...
</item>
```
