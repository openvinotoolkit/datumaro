---
title: 'Library contents'
linkTitle: 'Library contents'
description: ''
weight: 4

---

## Dataset Formats

The framework provides functions to read and write datasets in specific formats.
It is supported by `Extractor`s, `Importer`s, and `Converter`s.

Dataset reading is supported by `Extractor`s and `Importer`s:
- An `Extractor` produces a list of `DatasetItem`s corresponding to the
  dataset. Annotations are available in the `DatasetItem.annotations` list
- An `Importer` creates a project from a data source location

It is possible to add custom `Extractor`s and `Importer`s. To do this, you need
to put an `Extractor` and `Importer` implementations to a plugin directory.

Dataset writing is supported by `Converter`s.
A `Converter` produces a dataset of a specific format from dataset items.
It is possible to add custom `Converter`s. To do this, you need to put a
`Converter` implementation script to a plugin directory.


## Dataset Conversions ("Transforms")

A `Transform` is a function for altering a dataset and producing a new one.
It can update dataset items, annotations, classes, and other properties.
A list of available transforms for dataset conversions can be extended by
adding a `Transform` implementation script into a plugin directory.

## Model launchers

A list of available launchers for model execution can be extended by
adding a `Launcher` implementation script into a plugin directory.
