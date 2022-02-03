---
title: 'Dataset Management Framework (Datumaro) API and developer manual'
linkTitle: 'Developer Manual'
description: ''
weight: 4
---

## Basics

The central part of the library is the `Dataset` class, which represents
a dataset and allows to iterate over its elements.
`DatasetItem`, an element of a dataset, represents a single
dataset entry with annotations - an image, video sequence, audio track etc.
It can contain only annotated data or meta information, only annotations, or
all of this.

Basic library usage and data flow:

```
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

1. Data is read (or produced) by one or many `Extractor`s and
  [merged](#merging) into a `Dataset`
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
from datumaro import Bbox, Polygon, Dataset, DatasetItem

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

### The Dataset class

The `Dataset` class from the `datumaro.components.dataset` module represents
a dataset, consisting of multiple `DatasetItem`s. Annotations are
represented by members of the `datumaro.components.extractor` module,
such as `Label`, `Mask` or `Polygon`. A dataset can contain items from one or
multiple subsets (e.g. `train`, `test`, `val` etc.), the list of dataset
subsets is available in `dataset.subsets()`.

A `DatasetItem` is an element of a dataset. Its `id` is the name of the
corresponding image, video frame, or other media being annotated.
An item can have some `attributes`, associated media info and `annotations`.

Datasets typically have annotations, and these annotations can
require additional information to be interpreted correctly. For instance, it
can be class names, class hierarchy, keypoint connections,
class colors for masks, class attributes.
Such information is stored in `dataset.categories()`, which is a mapping from
`AnnotationType` to a corresponding `...Categories` class. Each annotation type
can have its `Categories`. Typically, there will be at least `LabelCategories`;
if there are instance masks, the dataset will contain `MaskCategories` etc.
The "main" type of categories is `LabelCategories` - annotations and other
categories use label indices from this object.

The main operation for a dataset is iteration over its elements
(`DatasetItem`s). An item corresponds to a single image, a video sequence,
etc. There are also many other operations available, such as filtration
(`dataset.select()`), transformation (`dataset.transform()`),
exporting (`dataset.export()`) and others. A `Dataset` is an `Iterable` and
`Extractor` by itself.

A `Dataset` can be created from scratch by its class constructor.
Categories can be set immediately or later with the
`define_categories()` method, but only once. You can create a dataset filled
with initial `DatasetItem`s with `Dataset.from_iterable()`.
If you need to create a dataset from one or many other extractors
(or datasets), it can be done with `Dataset.from_extractors()`.

If a dataset is created from multiple extractors with
`Dataset.from_extractors()`, the source datasets will be [joined](#merging),
so their categories must match. If datasets have mismatching categories,
use the more complex `IntersectMerge` class from `datumaro.components.operations`,
which will merge all the labels and remap the shifted indices in annotations.

A `Dataset` can be loaded from an existing dataset on disk with
`Dataset.import_from()` (for arbitrary formats) and
`Dataset.load()` (for the Datumaro data format).

By default, `Dataset` works lazily, which means all the operations requiring
iteration over inputs will be deferred as much as possible. If you don't want
such behavior, use the `init_cache()` method or wrap the code in
`eager_mode` (from `datumaro.components.dataset`), which will load all
the annotations into memory. The media won't be loaded unless the data
is required, because it can quickly waste all the available memory.
You can check if the dataset is cached with the `is_cache_initialized`
attribute.

Once created, a dataset can be modified in batch mode with transforms or
directly with the `put()` and `remove()` methods. `Dataset` instances
record information about changes done, which can be obtained by `get_patch()`.
The patch information is used automatically on saving and exporting to
reduce the amount of disk writes. Changes can be flushed with
`flush_changes()`.

```python
from datumaro import Bbox, Polygon, Dataset, DatasetItem

# create a dataset directly from items
dataset1 = Dataset.from_iterable([
  DatasetItem(id='image1', annotations=[
    Bbox(x=1, y=2, w=3, h=4, label=1),
    Polygon([1, 2, 3, 2, 4, 4], label=2),
  ]),
], categories=['cat', 'dog', 'person', 'truck'])

dataset2 = Dataset(categories=dataset1.categories())
dataset2.put(DatasetItem(id='image2', annotations=[
  Label(label=3),
  Bbox(x=2, y=0, w=3, h=1, label=2)
]))

# create a dataset from other datasets
dataset = Dataset.from_extractors(dataset1, dataset2)

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

#### Dataset merging <a id="merging"></a>

There are 2 methods of merging datasets in Datumaro:

- simple merging ("joining")
- complex merging

#### The simple merging ("joining")

This approach finds the corresponding `DatasetItem`s in inputs,
finds equal annotations and leaves only the unique set of annotations.
This approach requires all the inputs to have categories with the same
labels (or no labels) in the same order.

This algorithm is applied automatically in `Dataset.from_extractors()`
and when the build targets are merged in the `ProjectTree.make_dataset()`.

#### The complex merging

If datasets have mismatching categories, they can't be
merged by the simple approach, because it can lead to errors in the
resulting annotations. For complex cases Datumaro provides a more
sophisticated algorithm, which finds matching annotations by computing
distances between them. Labels and attributes are deduced by voting,
spatial annotations use the corresponding metrics like
Intersection-over-Union (IoU), OKS, PDJ and others.

The categories of the input datasets are compared, the matching ones
complement missing information in each other, the mismatching ones are
appended after next. Label indices in annotations are shifted to the
new values.

The complex algorithm is available in the `IntersectMerge` class
from `datumaro.components.operations`. It must be used explicitly.
This class also allows to check the inputs and the output dataset
for errors and problems.

### Projects

Projects are intended for complex use of Datumaro. They provide means of
persistence, versioning, high-level operations for datasets and also
allow to extend Datumaro via [plugins](#plugins). A project provides
access to build trees and revisions, data sources, models, configuration,
plugins and cache. Projects can have multiple data sources, which are
[joined](#merging) on dataset creation. Project configuration is available
in `project.config`. To add a data source into a `Project`, use
the `import_source()` method. The build tree of the current working
directory can be converted to a `Dataset` with
`project.working_tree.make_dataset()`.

The `Environment` class is responsible for accessing built-in and
project-specific plugins. For a `Project` object, there is an instance of
related `Environment` in `project.env`.

Check the [Data Model section of the User Manual](/docs/user-manual/supported_formats)
for more info about Project behavior and high-level details.

## Library contents

### Dataset Formats

The framework provides functions to read and write datasets in specific formats.
It is supported by `Extractor`s, `Importer`s, and `Converter`s.

Dataset reading is supported by `Extractor`s and `Importer`s:
- An `Extractor` produces a list of `DatasetItem`s corresponding to the
  dataset. Annotations are available in the `DatasetItem.annotations` list.
  The `SourceExtractor` class is designed for loading simple, single-subset
  datasets. It should be used by default. The `Extractor` base class should
  be used when `SourceExtractor`'s functionality is not enough.
- An `Importer` detects dataset files and generates dataset loading parameters
  for the corresponding `Extractor`s. `Importer`s are optional, they
  only extend the Extractor functionality and make them more flexible and
  simple. They are mostly used to locate dataset subsets, but they also can
  do some data compatibility checks and have other required logic.

It is possible to add custom `Extractor`s and `Importer`s. To do this, you need
to put an `Extractor` and `Importer` implementations to a plugin directory.

Dataset writing is supported by `Converter`s.
A `Converter` produces a dataset of a specific format from dataset items.
It is possible to add custom `Converter`s. To do this, you need to put a
`Converter` implementation script to a plugin directory.

### Dataset Conversions ("Transforms")

A `Transform` is a function for altering a dataset and producing a new one.
It can update dataset items, annotations, classes, and other properties.
A list of available transforms for dataset conversions can be extended by
adding a `Transform` implementation script into a plugin directory.

### Model launchers

A list of available launchers for model execution can be extended by
adding a `Launcher` implementation script into a plugin directory.

## Plugins

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
import datumaro as dm
from datumaro.plugins.yolo_format.converter import YoloConverter

# Import a dataset
dataset = dm.Dataset.import_from(src_dir, 'voc')

# Load an existing project, save the dataset in some project-specific format
project = Project('project/')
project.env.converters['custom_format'].convert(dataset, save_dir=dst_dir)

# Save the dataset in some built-in format
Environment().converters['yolo'].convert(dataset, save_dir=dst_dir)
YoloConverter.convert(dataset, save_dir=dst_dir)
```

### Writing a plugin

A plugin is a Python module with any name, which exports some symbols. Symbols,
starting with `_` are not exported by default. To export a symbol,
inherit it from one of the special classes:

```python
from datumaro import Importer, Extractor, Transform, Launcher, Converter
```

The `exports` list of the module can be used to override default behaviour:
```python
class MyComponent1: ...
class MyComponent2: ...
exports = [MyComponent2] # exports only MyComponent2
```

There is also an additional class to modify plugin appearance in command line:

```python
from datumaro import CliPlugin

class MyPlugin(Converter, CliPlugin):
  """
  Optional documentation text, which will appear in command-line help
  """

  NAME = 'optional_custom_plugin_name'

  def build_cmdline_parser(self, **kwargs):
    parser = super().build_cmdline_parser(**kwargs)
    # set up argparse.ArgumentParser instance
    # the parsed args are supposed to be used as invocation options
    return parser
```

#### Plugin example

<!--lint disable fenced-code-flag-->

```
datumaro/plugins/
- my_plugin1/file1.py
- my_plugin1/file2.py
- my_plugin2.py
```

<!--lint enable fenced-code-flag-->

`my_plugin1/file2.py` contents:

```python
from datumaro import Transform
from .file1 import something, useful

class MyTransform(Transform):
    NAME = "custom_name" # could be generated automatically

    """
    Some description. The text will be displayed in the command line output.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-q', help="Very useful parameter")
        return parser

    def __init__(self, extractor, q):
        super().__init__(extractor)
        self.q = q

    def transform_item(self, item):
        return item
```

`my_plugin2.py` contents:

```python
from datumaro import Extractor

class MyFormat: ...
class _MyFormatConverter(Converter): ...
class MyFormatExtractor(Extractor): ...

exports = [MyFormat] # explicit exports declaration
# MyFormatExtractor and _MyFormatConverter won't be exported
```

## Command-line

Basically, the interface is divided on contexts and single commands.
Contexts are semantically grouped commands, related to a single topic or target.
Single commands are handy shorter alternatives for the most used commands
and also special commands, which are hard to be put into any specific context.
[Docker](https://www.docker.com/) is an example of similar approach.

<div class="text-center large-scheme-two">

```mermaid
%%{init { 'theme':'neutral' }}%%
flowchart LR
  d(("#0009; datum #0009;")):::mainclass
  s(source):::nofillclass
  m(model):::nofillclass
  p(project):::nofillclass

  d===s
    s===id1[add]:::hideclass
    s===id2[remove]:::hideclass
    s===id3[info]:::hideclass
  d===m
    m===id4[add]:::hideclass
    m===id5[remove]:::hideclass
    m===id6[run]:::hideclass
    m===id7[info]:::hideclass
  d===p
    p===migrate:::hideclass
    p===info:::hideclass
  d====str1[create]:::filloneclass
  d====str2[add]:::filloneclass
  d====str3[remove]:::filloneclass
  d====str4[export]:::filloneclass
  d====str5[info]:::filloneclass
  d====str6[transform]:::filltwoclass
  d====str7[filter]:::filltwoclass
  d====str8[diff]:::fillthreeclass
  d====str9[merge]:::fillthreeclass
  d====str10[validate]:::fillthreeclass
  d====str11[explain]:::fillthreeclass
  d====str12[stats]:::fillthreeclass
  d====str13[commit]:::fillfourclass
  d====str14[checkout]:::fillfourclass
  d====str15[status]:::fillfourclass
  d====str16[log]:::fillfourclass

  classDef nofillclass fill-opacity:0;
  classDef hideclass fill-opacity:0,stroke-opacity:0;
  classDef filloneclass fill:#CCCCFF,stroke-opacity:0;
  classDef filltwoclass fill:#FFFF99,stroke-opacity:0;
  classDef fillthreeclass fill:#CCFFFF,stroke-opacity:0;
  classDef fillfourclass fill:#CCFFCC,stroke-opacity:0;
```

</div>

Model-View-ViewModel (MVVM) UI pattern is used.

<div class="text-center">

```mermaid
%%{init { 'theme':'neutral' }}%%
flowchart LR
    c((CLI))<--CliModel--->d((Domain))
    g((GUI))<--GuiModel--->d
    a((API))<--->d
    t((Tests))<--->d
```

</div>
