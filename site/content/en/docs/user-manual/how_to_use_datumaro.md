---
title: 'How to use Datumaro'
linkTitle: 'How to use Datumaro'
description: ''
weight: 2
---

As a standalone tool or a Python module:

``` bash
datum --help

python -m datumaro --help
python datumaro/ --help
python datum.py --help
```

As a Python library:

``` python
from datumaro.components.project import Project
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import Label, Bbox, DatasetItem
...
dataset = Dataset.import_from(path, format)
...
```

### Glossary

- Basic concepts:
  - Dataset - A collection of dataset items, which consist of media and
    associated annotations.
  - Dataset item - A basic single element of the dataset. Also known as
    "sample", "entry". In different datasets it can be an image, a video
    frame, a whole video, a 3d point cloud etc. Typically, has corresponding
    annotations.
  - (Datumaro) Project - A combination of multiple datasets, plugins,
    models and metadata.

- Project versioning concepts:
  - Data source - A link to a dataset or a copy of a dataset inside a project.
    Basically, a URL + dataset format name.
  - Project revision - A commit or a reference from Git (branch, tag,
    HEAD~3 etc.). A revision is referenced by data hash. The `HEAD`
    revision is the currently selected revision of the project.
  - Revision tree - A project build tree and plugins at
    a specified revision.
  - Working tree - The revision tree in the working directory of a project.
  - data source revision - a state of a data source at a specific stage.
    A revision is referenced by the data hash.
  - Object - The data of a revision tree or a data source revision.
    An object is referenced by the data hash.

- Dataset path concepts: <a id="revpath"></a>
  - Dataset revpath - A path to a dataset in a special format. They are
    supposed to specify paths to files, directories or data source revisions
    in a uniform way in the CLI.

    - dataset path - a path to a dataset in the following format:
      `<dataset path>:<format>`
      - `format` is optional. If not specified, will try to detect automatically

    - **rev**ision path - a path to a data source revision in a project.
      The syntax is:
      `<project path>@<revision>:<target name>`, any part can be omitted.
      - Default project is the current project (`-p`/`--project` CLI arg.)
        Local revpaths imply that the current project is used and this part
        should be omitted.
      - Default revision is the working tree of the project
      - Default build target is `project`

      If a path refers to `project` (i.e. target name is not set, or
      this target is exactly specified), the target dataset is the result of
      [joining](/docs/developer_manual/#merging) all the project data sources.
      Otherwise, if the path refers to a data source revision, the
      corresponding stage from the revision build tree will be used.

- Dataset building concepts:
  - Stage - A revision of a dataset - the original dataset or its modification
    after transformation, filtration or something else. A build tree node.
    A stage is referred by a name.
  - Build tree - A directed graph (tree) with root nodes at data sources
    and a single top node called `project`, which represents
    a [joined](/docs/developer_manual/#merging) dataset.
    Each data source has a starting `root` node, which corresponds to the
    original dataset. The internal graph nodes are stages.
  - Build target - A data source or a stage name. Data source names correspond
    to the last stages of data sources.
  - Pipeline - A subgraph of a stage, which includes all the ancestors.

- Other:
  - Transform - A transformation operation over dataset elements. Examples
    are image renaming, image flipping, image and subset renaming,
    label remapping etc. Corresponds to the [`transform` command](/docs/user-manual/command-reference/transform).

### Command-line workflow

In Datumaro, most command-line commands operate on projects, but there are
also few commands operating on datasets directly. There are 2 basic ways
to use Datumaro from the command-line:
- Use the [`convert`](/docs/user-manual/command-reference/convert), [`diff`](/docs/user-manual/command-reference/diff), [`merge`](/docs/user-manual/command-reference/merge)
  commands directly on existing datasets

- Create a Datumaro project and operate on it:
  - Create an empty project with [`create`](/docs/user-manual/command-reference/create)
  - Import existing datasets with [`add`](/docs/user-manual/command-reference/sources/#source-add)
  - Modify the project with [`transform`](/docs/user-manual/command-reference/transform) and [`filter`](/docs/user-manual/command-reference/filter)
  - Create new revisions of the project with
    [`commit`](/docs/user-manual/command-reference/commit), navigate over
    them using [`checkout`](/docs/user-manual/command-reference/checkout),
    compare with [`diff`](/docs/user-manual/command-reference/diff), compute
    statistics with [`stats`](/docs/user-manual/command-reference/stats)
  - Export the resulting dataset with [`export`](/docs/user-manual/command-reference/export)
  - Check project config with [`project info`](/docs/user-manual/command-reference/projects/#print-project-info)

Basically, a project is a combination of datasets, models and environment.

A project can contain an arbitrary number of data sources. Each data source
describes a dataset in a specific format. A project acts as a manager for
the data sources and allows to manipulate them separately or as a whole, in
which case it combines dataset items from all the sources into one composite
dataset. You can manage separate sources in a project by commands in
the `datum source` command-line context.

Note that **modifying operations** (`transform`, `filter`) **are applied
in-place** to the data sources by default.

If you want to interact with models, you need to add them to the project
first using the [`model add`](/docs/user-manual/command-reference/models/#register-model) command.

A typical way to obtain Datumaro projects is to export tasks in
[CVAT](https://github.com/openvinotoolkit/cvat) UI.

### Project data model <a id="data-model"></a>

![project model](/images/project_model.svg)

Datumaro tries to combine a "Git for datasets" and a build system like
make or CMake for datasets in a single solution. Currently, `Project`
represents a Version Control System for datasets, which is based on Git and DVC
projects. Each project `Revision` describes a build tree of a dataset
with all the related metadata. A build tree consists of a number of data
sources and transformation stages. Each data source has its own set of build
steps (stages). Datumaro supposes copying of datasets and working in-place by
default. Modifying operations are recorded in the project, so any of the
dataset revisions can be reproduced when needed. Multiple dataset versions can
be stored in different branches with the common data shared.

Let's consider an example of a build tree:
![build tree](/images/build_tree.svg)
There are 2 data sources in the example project. The resulting dataset
is obtained by simple merging (joining) the results of the input datasets.
"Source 1" and "Source 2" are the names of data sources in the project. Each
source has several stages with their own names. The first stage (called "root")
represents the original contents of a data source - the data at the
user-provided URL. The following stages represent operations, which needs to
be done with the data source to prepare the resulting dataset.

Roughly, such build tree can be created by the following commands (arguments
are omitted for simplicity):
``` bash
datum create

# describe the first source
datum add <...> -n source1
datum filter <...> source1
datum transform <...> source1
datum transform <...> source1

# describe the second source
datum add <...> -n source2
datum model add <...>
datum transform <...> source2
datum transform <...> source2
```

Now, the resulting dataset can be built with:

``` bash
datum export <...>
```

### Project layout

``` bash
project/
├── .dvc/
├── .dvcignore
├── .git/
├── .gitignore
├── .datumaro/
│   ├── cache/ # object cache
│   │   └── <2 leading symbols of obj hash>/
│   │       └── <remaining symbols of obj hash>/
│   │           └── <object data>
│   │
│   ├── models/ # project-specific models
│   │
│   ├── plugins/ # project-specific plugins
│   │   ├── plugin1/ # composite plugin, a directory
│   │   |   ├── __init__.py
│   │   |   └── file2.py
│   │   ├── plugin2.py # simple plugin, a file
│   │   └── ...
│   │
│   ├── tmp/ # temp files
│   └── tree/ # working tree metadata
│       ├── config.yml
│       └── sources/
│           ├── <source name 1>.dvc
│           ├── <source name 2>.dvc
│           └── ...
│
├── <source name 1>/ # working directory for the source 1
│   └── <source data>
└── <source name 2>/ # working directory for the source 2
    └── <source data>
```

### Use cases

Let's consider few examples describing what Datumaro does for you behind the
scene.

The first example explains how working trees, working directories and the
cache interact. Suppose, there is a dataset which we want to modify and
export in some other format. To do it with Datumaro, we need to create a
project and register the dataset as a data source:

``` bash
datum create
datum add <...> -n source1
```

The dataset will be copied to the working directory inside the project. It
will be hashed and added to the project working tree.

After the dataset is added, we want to transform it and filter out some
irrelevant samples, so we run the following commands:

``` bash
datum transform <...> source1
datum filter <...> source1
```

The commands modify the data source inside the working directory, inplace.
The operations done are recorded in the working tree, the results are hashed.

Now, we want to make a new version of the dataset and make a snapshot in the
project cache. So we `commit` the working tree:

``` bash
datum commit <...>
```

![cache interaction diagram 1](/images/behavior_diag1.svg)

At this time, the data source is copied into the project cache and a new
project revision is created. The dataset operation history is saved, so
the dataset can be reproduced even if it is removed from the cache and the
working directory.

After this, we do some other modifications to the dataset and make a new
commit. Note that the dataset is not cached, until a `commit` is done.

When the dataset is ready and all the required operations are done, we
can `export` it to the required format. We can export the resulting dataset,
or any previous stage.

``` bash
datum export <...> source1
datum export <...> source1.stage3
```

Let's extend the example. Imagine we have a project with 2 data sources.
Roughly, it corresponds to the following set of commands:

```bash
datum create
datum add <...> -n source1
datum add <...> -n source2
datum transform <...> source1 # used 3 times
datum transform <...> source2 # used 5 times
```

Then, for some reasons, the project cache was cleaned from `source1` revisions.
We also don't have anything in the project working directories - suppose,
the user removed them to save disk space.

Let's see what happens, if we call the `diff` command with 2 different
revisions now.

![cache interaction diagram 2](/images/behavior_diag2.svg)

Datumaro needs to reproduce 2 dataset revisions requested so that they could
be read and compared. Let's see how the first dataset is reproduced
step-by-step:

1. `source1.stage2` will be looked for in the project cache. It won't be
  found, since the cache was cleaned.
1. Then, Datumaro will look for previous source revisions in the cache
  and won't find any.
1. The project can be marked read-only, if we are not working with the
  "current" project (which is specified by the `-p/--project` command
  parameter). In the example, the command is `datum diff rev1:... rev2:...`,
  which means there is a project in the current directory, so the project
  we are working with is not read-only. If a command target was specified as
  `datum diff <project>@<rev>:<source>`, the project would be loaded
  as read-only. If a project is read-only, we can't do anything more to
  reproduce the dataset and can only exit with an error (3a). The reason for
  such behavior is that the dataset downloading can be quite expensive (in
  terms of time, disk space etc.). It is supposed, that such side-effects
  should be controlled manually.
1. If the project is not read-only (3b), Datumaro will try to download
  the original dataset and reproduce the resulting dataset. The data hash
  will be computed and hashes will be compared. On success, the data will be
  put into the cache.
1. The downloaded dataset will be read and the remaining operations from the
  source history will be re-applied.
1. The resulting dataset might be cached in some cases.
1. The resulting dataset is returned.

The `source2` will be looked for the same way. In our case, it will be found
in the cache and returned. Once both datasets are restored and read, they
are compared.

Consider other situation. Let's try to `export` the `source1`. Suppose
we have a clear project cache and the `source1` has a copy in the working
directory.

![cache interaction diagram 3](/images/behavior_diag3.svg)

Again, Datumaro needs to reproduce a dataset revision (stage) requested.
1. It looks for the dataset in the working directory and finds some data. If
  there is no source working directory, Datumaro will try to reproduce the
  source using the approach described above (1b).
1. The data hash is computed and compared with the one saved in the history.
  If the hashes match, the dataset is read and returned (4).
  Note: we can't use the cached hash stored in the working tree info -
  it can be outdated, so we need to compute it again.
1. Otherwise, Datumaro tries to detect the stage by the data hash.
  If the current stage is not cached, the tree is the working tree and the
  working directory is not empty, the working copy is hashed and matched
  against the source stage list. If there is a matching stage, it will be
  read and the missing stages will be added. The result might be cached in
  some cases.
  If there is no matching stage in the source history, the situation can
  be contradictory. Currently, an error is raised (3b).
1. The resulting dataset is returned.

After the requested dataset is obtained, it is exported in the requested
format.

To sum up, Datumaro tries to restore a dataset from the project cache or
reproduce it from sources. It can be done as long as the source operations
are recorded and any step data is available. Note that cache objects share
common files, so if there are only annotation differences between datasets,
or data sources contain the same images, there will only be a single copy
of the related media files. This helps to keep storage use reasonable and
avoid unnecessary data copies.

### Examples <a id="cli-examples"></a>

Example: create a project, add dataset, modify, restore an old version

``` bash
datum create
datum add <path/to/dataset> -f coco -n source1
datum commit -m "Added a dataset"
datum transform -t shapes_to_boxes
datum filter -e '/item/annotation[label="cat" or label="dog"]' -m i+a
datum commit -m "Transformed"
datum checkout HEAD~1 -- source1 # restore a previous revision
datum status # prints "modified source1"
datum checkout source1 # restore the last revision
datum export -f voc -- --save-images
```
