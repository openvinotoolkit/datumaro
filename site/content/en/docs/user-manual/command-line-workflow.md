---
title: 'Command line workflow'
linkTitle: 'Command line workflow'
description: ''
weight: 5
---

The key object is a project, so most CLI commands operate on projects.
However, there are few commands operating on datasets directly.
A project is a combination of a project's own dataset, a number of
external data sources and an environment.
An empty Project can be created by `project create` command,
an existing dataset can be imported with `project import` command.
A typical way to obtain projects is to export tasks in CVAT UI.

If you want to interact with models, you need to add them to project first.

## Project structure

<!--lint disable fenced-code-flag-->
```
└── project/
    ├── .datumaro/
    |   ├── config.yml
    │   ├── .git/
    │   ├── models/
    │   └── plugins/
    │       ├── plugin1/
    │       |   ├── file1.py
    │       |   └── file2.py
    │       ├── plugin2.py
    │       ├── custom_extractor1.py
    │       └── ...
    ├── dataset/
    └── sources/
        ├── source1
        └── ...
```
<!--lint enable fenced-code-flag-->
