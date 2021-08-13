---
title: 'Plugins'
linkTitle: 'Plugins'
description: ''
weight: 8
tags: [  'Examples for python module', ]
---

Datumaro comes with a number of built-in formats and other tools,
but it also can be extended by plugins. Plugins are optional components,
which dependencies are not installed by default.
In Datumaro there are several types of plugins, which include:
- `extractor` - produces dataset items from data source
- `importer` - recognizes dataset type and creates project
- `converter` - exports dataset to a specific format
- `transformation` - modifies dataset items or other properties
- `launcher` - executes models

A plugin is a regular Python module. It must be present in a plugin directory:
- `<project_dir>/.datumaro/plugins` for project-specific plugins
- `<datumaro_dir>/plugins` for global plugins

A plugin can be used either via the `Environment` class instance,
or by regular module importing:

```python
from datumaro.components.project import Environment, Project
from datumaro.plugins.yolo_format.converter import YoloConverter

# Import a dataset
dataset = Environment().make_importer('voc')(src_dir).make_dataset()

# Load an existing project, save the dataset in some project-specific format
project = Project.load('project/dir')
project.env.converters.get('custom_format').convert(dataset, save_dir=dst_dir)

# Save the dataset in some built-in format
Environment().converters.get('yolo').convert(dataset, save_dir=dst_dir)
YoloConverter.convert(dataset, save_dir=dst_dir)
```
