---
title: 'Merge projects'
linkTitle: 'Merge projects'
description: ''
weight: 15
---

This command merges items from 2 or more projects and checks annotations for
errors.

Spatial annotations are compared by distance and intersected, labels and
attributes are selected by voting.
Merge conflicts, missing items and annotations, other errors are saved into a `.json` file.

Usage:

``` bash
datum merge --help

datum merge <project dirs>
```

Example: merge 4 (partially-)intersecting projects,
- consider voting succeeded when there are 3+ same votes
- consider shapes intersecting when IoU >= 0.6
- check annotation groups to have `person`, `hand`, `head` and `foot`
(`?` for optional)

``` bash
datum merge project1/ project2/ project3/ project4/ \
    --quorum 3 \
    -iou 0.6 \
    --groups 'person,hand?,head,foot?'
```
