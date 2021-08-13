---
title: 'Basics'
linkTitle: 'Basics'
description: ''
weight: 1
tags: [ 'PASCAL VOC',  'Schemes', 'Features', 'Subsets', ]
---

The center part of the library is the `Dataset` class, which represents
a dataset and allows to iterate over its elements.
`DatasetItem`, an element of a dataset, represents a single
dataset entry with annotations - an image, video sequence, audio track etc.
It can contain only annotated data or meta information, only annotations, or
all of this.

Basic library usage and data flow:

```lang-none
Extractors -> Dataset -> Converter
                 |
             Filtration
          Transformations
             Statistics
              Merging
             Inference
          Quality Checking
             Comparison
                ...
```

1. Data is read (or produced) by one or many `Extractor`s and merged
  into a `Dataset`
1. The dataset is processed in some way
1. The dataset is saved with a `Converter`

Datumaro has a number of dataset and annotation features:
- iteration over dataset elements
- filtering of datasets and annotations by a custom criteria
- working with subsets (e.g. `train`, `val`, `test`)
- computing of dataset statistics
- comparison and merging of datasets
- various annotation operations

```python
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import Bbox, Polygon, DatasetItem

# Import and export a dataset
dataset = Dataset.import_from('src/dir', 'voc')
dataset.export('dst/dir', 'coco')

# Create a dataset, convert polygons to masks, save in PASCAL VOC format
dataset = Dataset.from_iterable([
  DatasetItem(id='image1', annotations=[
    Bbox(x=1, y=2, w=3, h=4, label=1),
    Polygon([1, 2, 3, 2, 4, 4], label=2, attributes={'occluded': True}),
  ]),
], categories=['cat', 'dog', 'person'])
dataset.transform('polygons_to_masks')
dataset.export('dst/dir', 'voc')
```
