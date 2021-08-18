---
title: 'The Dataset class'
linkTitle: 'The Dataset class'
description: ''
weight: 2
---

The `Dataset` class from the `datumaro.components.dataset` module represents
a dataset, consisting of multiple `DatasetItem`s. Annotations are
represented by members of the `datumaro.components.extractor` module,
such as `Label`, `Mask` or `Polygon`. A dataset can contain items from one or
multiple subsets (e.g. `train`, `test`, `val` etc.), the list of dataset subsets
is available at `dataset.subsets`.

Datasets typically have annotations, and these annotations can
require additional information to be interpreted correctly. For instance, it
can include class names, class hierarchy, keypoint connections,
class colors for masks, class attributes.
This information is stored in `dataset.categories`, which is a mapping from
`AnnotationType` to a corresponding `...Categories` class. Each annotation type
can have its `Categories`. Typically, there will be a `LabelCategories` object.
Annotations and other categories address dataset labels
by their indices in this object.

The main operation for a dataset is iteration over its elements.
An item corresponds to a single image, a video sequence, etc. There are also
few other operations available, such as filtration (`dataset.select`) and
transformations (`dataset.transform`). A dataset can be created from extractors
or other datasets with `Dataset.from_extractors()` and directly from items with
`Dataset.from_iterable()`. A dataset is an extractor itself. If it is created
from multiple extractors, their categories must match, and their contents
will be merged.

A dataset item is an element of a dataset. Its `id` is a name of a
corresponding image. There can be some image `attributes`,
an `image` and `annotations`.

```python
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import Bbox, Polygon, DatasetItem

# create a dataset from other datasets
dataset = Dataset.from_extractors(dataset1, dataset2)

# or directly from items
dataset = Dataset.from_iterable([
  DatasetItem(id='image1', annotations=[
    Bbox(x=1, y=2, w=3, h=4, label=1),
    Polygon([1, 2, 3, 2, 4, 4], label=2),
  ]),
], categories=['cat', 'dog', 'person'])

# keep only annotated images
dataset.select(lambda item: len(item.annotations) != 0)

# change dataset labels
dataset.transform('remap_labels',
  {'cat': 'dog', # rename cat to dog
    'truck': 'car', # rename truck to car
    'person': '', # remove this label
  }, default='delete')

# iterate over elements
for item in dataset:
  print(item.id, item.annotations)

# iterate over subsets as Datasets
for subset_name, subset in dataset.subsets().items():
  for item in subset:
    print(item.id, item.annotations)
```
