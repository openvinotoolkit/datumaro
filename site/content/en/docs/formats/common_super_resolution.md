---
title: 'Super Resolution'
linkTitle: 'Super Resolution'
description: ''
weight: 2
---

## Format specification

Super resolution format specification is available [here](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/accuracy_checker/openvino/tools/accuracy_checker/annotation_converters/README.md#supported-converters).

Supported annotation types:
- `ImageResolution`

## Import Super Resolution dataset

A Datumaro project with a Super Resolution source can be created in the following way:

``` bash
datum create
datum import --format super_resolution <path/to/dataset>
```

Super Resolution dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── HR/
    │   ├── <img1>.png
    │   ├── <img2>.png
    │   └── ...
    ├── LR/
    │   ├── <img1>.png
    │   ├── <img2>.png
    │   └── ...
    └── upsampled/ # optional
        ├── <img1>.png
        ├── <img2>.png
        └── ...
```

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_common_super_resolution_format.py)
