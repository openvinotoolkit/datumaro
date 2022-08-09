---
title: 'Common Super Resolution'
linkTitle: 'Common Super Resolution'
description: ''
---

## Format specification

CSR format specification is available [here](https://github.com/cvat-ai/workbench/blob/master/docs/Workbench_DG/Dataset_Types.md#common-super-resolution-csr).

Supported annotation types:
- `SuperResolutionAnnotation`

Supported attributes:
- `upsampled` (`Image`): upsampled image

## Import Common Super Resolution dataset

A Datumaro project with a CSR source can be created in the following way:

``` bash
datum create
datum import --format common_super_resolution <path/to/dataset>
```

CSR dataset directory should have the following structure:

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
[the format tests](https://github.com/cvat-ai/datumaro/blob/develop/tests/test_common_super_resolution_format.py)
