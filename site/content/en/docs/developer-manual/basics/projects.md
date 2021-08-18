---
title: 'Projects'
linkTitle: 'Projects'
description: ''
weight: 3
---

Projects are intended for complex use of Datumaro. They provide means of
persistence, of extending, and CLI operation for Datasets. A project can
be converted to a Dataset with `project.make_dataset`. Project datasets
can have multiple data sources, which are merged on dataset creation. They
can have a hierarchy. Project configuration is available in `project.config`.
A dataset can be saved in `datumaro_project` format.

The `Environment` class is responsible for accessing built-in and
project-specific plugins. For a project, there is an instance of
related `Environment` in `project.env`.
