---
title: 'Filter datasets'
linkTitle: 'Filter'
description: ''
weight: 12
---

This command allows to extract a sub-dataset from a dataset. The new dataset
includes only items satisfying some condition. The XML [XPath](https://devhints.io/xpath)
is used as a query format.

There are several filtering modes available (the `-m/--mode` parameter).
Supported modes:
- `i`, `items`
- `a`, `annotations`
- `i+a`, `a+i`, `items+annotations`, `annotations+items`

When filtering annotations, use the `items+annotations`
mode to point that annotation-less dataset items should be
removed, otherwise they will be kept in the resulting dataset.
To select an annotation, write an XPath that returns `annotation`
elements (see examples).

Item representations can be printed with the `--dry-run` parameter:

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
    <w>11.19</w>
    <h>42.31</h>
    <area>473.87</area>
  </annotation>
  <annotation>
    <id>669839</id>
    <type>bbox</type>
    <label_id>41</label_id>
    <x>163.58</x>
    <y>191.75</y>
    <w>76.98</w>
    <h>73.63</h>
    <area>5668.77</area>
  </annotation>
  ...
</item>
```

The command can only be applied to a project build target, a stage or the
combined `project` target, in which case all the targets will be affected.
A build tree stage will be added if `--stage` is enabled, and the resulting
dataset(-s) will be saved if `--apply` is enabled.

Usage:

``` bash
datum filter [-h] [-e FILTER] [-m MODE] [--dry-run] [--stage STAGE]
  [--apply APPLY] [-o DST_DIR] [--overwrite] [-p PROJECT_DIR] [target]
```

Parameters:
- `<target>` (string) - A project build target to be filtered.
  By default, all project targets are affected.
- `-e, --filter` (string) - XML XPath filter expression for dataset items
- `-m, --mode` (string) - The filtering mode. Default is the `i` mode.
- `--dry-run` - Print XML representations of the filtered dataset and exit.
- `--stage` (bool) - Include this action as a project build step.
  If true, this operation will be saved in the project
  build tree, allowing to reproduce the resulting dataset later.
  Applicable only to data source targets (i.e. not intermediate stages)
  and the `project` target. Enabled by default.
- `--apply` (bool) - Run this command immediately. If disabled, only the
  build tree stage will be written. Enabled by default.
- `-o, --output-dir` (string) - Output directory. Can be omitted for
  data source targets (i.e. not intermediate stages) and the `project` target,
  in which case the results will be saved in place in the working tree.
- `--overwrite` - Allows to overwrite existing files in the output directory,
  when it is specified and is not empty.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Example: extract a dataset with images with `width` < `height`

``` bash
datum filter \
  -p test_project \
  -e '/item[image/width < image/height]'
```

Example: extract a dataset with images of the `train` subset

``` bash
datum filter \
  -p test_project \
  -e '/item[subset="train"]'
```

Example: extract a dataset with only large annotations of the `cat` class and
any non-`persons`

``` bash
datum filter \
  -p test_project \
  --mode annotations \
  -e '/item/annotation[(label="cat" and area > 99.5) or label!="person"]'
```

Example: extract a dataset with non-occluded annotations, remove empty images.
Use data only from the "s1" source of the project.

``` bash
datum create
datum import --format voc -i <path/to/dataset1/> --name s1
datum import --format voc -i <path/to/dataset2/> --name s2
datum filter s1 \
  -m i+a -e '/item/annotation[occluded="False"]'
```
