---
title: 'Update project (merge)'
linkTitle: 'Update project'
description: ''
weight: 13
tags: [  'Examples for standalone tool', ]
---

This command updates items in a project from another one
(check [Merge Projects](/docs/user-manual/command-reference/merge-projects/)
for complex merging).

Usage:

``` bash
datum merge --help

datum merge \
    -p <project dir> \
    -o <output dir> \
    <other project dir>
```

Example: update annotations in the `first_project` with annotations
from the `second_project` and save the result as `merged_project`

``` bash
datum merge \
    -p first_project \
    -o merged_project \
    second_project
```
