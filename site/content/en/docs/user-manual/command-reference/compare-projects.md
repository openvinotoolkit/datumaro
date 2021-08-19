---
title: 'Compare projects'
linkTitle: 'Compare projects'
description: ''
weight: 16
---

The command compares two datasets and saves the results in the
specified directory. The current project is considered to be
"ground truth".

``` bash
datum diff --help

datum diff <other_project_dir> -o <save_dir>
```

Example: compare a dataset with model inference

``` bash
datum import <...>
datum model add mymodel <...>
datum transform <...> -o inference
datum diff inference -o diff
```
