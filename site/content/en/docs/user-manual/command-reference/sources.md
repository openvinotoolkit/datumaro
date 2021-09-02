---
title: 'Sources'
linkTitle: 'Sources'
description: ''
weight: 26
---

### Import Datasets <a id="source-add"></a>

A project can contain an arbitrary number of Data Sources. Each Data Source
describes a dataset in a specific format. A project acts as a manager for
the data sources and allows to manipulate them separately or as a whole, in
which case it combines dataset items from all the sources into one composite
dataset. You can manage separate sources in a project by commands in
the `datum source` command line context.

Existing datasets can be added to a Datumaro project with the `add` command.

Datasets come in a wide variety of formats. Each dataset
format defines its own data structure and rules on how to
interpret the data. For example, the following data structure
is used in COCO format:
<!--lint disable fenced-code-flag-->
```
/dataset/
- /images/<id>.jpg
- /annotations/
```
<!--lint enable fenced-code-flag-->

Dataset format readers can provide some additional import options. To pass
such options, use the `--` separator after the main command arguments.
The usage information can be printed with `datum add -f <format> -- --help`.

Check [supported formats](/docs/user-manual/supported_formats) for more info about
format specifications, supported options and other details.
The list of formats can be extended by custom plugins, check [extending tips](/docs/user-manual/extending)
for information on this topic.

The list of currently available formats are listed in the command help output.

Datumaro supports working with complete datasets, having both image data and
annotations, or with annotations only. It can be used to prepare
images and annotations independently of each other, or to process the
lightweight annotations without downloading the whole dataset.

A dataset is imported by its URL. Currently, only local filesystem
paths are supported. The URL can be a file or a directory path
to a dataset. When the dataset is read, it is read as a whole.
However, many formats can have multiple subsets like `train`, `val`, `test`
etc. If you want to limit reading only to a specific subset, use
the `-r/--path` parameter. It can also be useful when subset files have
non-standard placement or names.

When a dataset is imported, the following things are done:
- URL is saved in the project config
- data in copied into the project
- data is cached inside the project (use `--no-cache` to disable)

Each data source has a name assigned, which can be used in other commands. To
set a specific name, use the `-n/--name` parameter.

The dataset is added into the working tree of the project. A new commit
is _not_ done automatically.

Usage:

``` bash
datum add [-h] [-n NAME] -f FORMAT [-r PATH] [--no-check] [--no-cache]
  [-p PROJECT_DIR] url [-- EXTRA_FORMAT_ARGS]
```

Parameters:
- `<url>` (string) - A file of directory path to the dataset.
- `-f, --format` (string) - Dataset format
- `-r, --path` (string) - A path relative to the source URL the data source.
  Useful to specify a path to a subset, subtask, or a specific file in URL.
- `--no-check` - Don't try to read the source after importing
- `--no-cache` - Don't put a copy into the project cache
- `-n`, `--name` (string) - Name of the new source (default: generate
  automatically)
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `-- <extra format args>` - Additional arguments for the format reader
  (use `-- -h` for help). Must be specified after the main command arguments.

Example: create a project from images and annotations in different formats,
export as TFrecord for TF Detection API for model training

``` bash
datum create
# 'default' is the name of the subset below
datum add <path/to/coco/instances_default.json> -f coco_instances
datum add <path/to/cvat/default.xml> -f cvat
datum add <path/to/voc> -f voc_detection -r custom_subset_dir/default.txt
datum add <path/to/datumaro/default.json> -f datumaro
datum add <path/to/images/dir> -f image_dir
datum export -f tf_detection_api -- --save-images
```

### Remove Datasets <a id="source-remove"></a>

To remove a data source from a project, use the `remove` command.

Usage:

``` bash
datum remove [-h] [--force] [--keep-data] [-p PROJECT_DIR] name [name ...]
```

Parameters:
- `<name>` (string) - The name of the source to be removed (repeatable)
- `-f, --force` - Do not fail and stop on errors during removal
- `--keep-data` - Do not remove source data from the working directory, remove
  only project metainfo.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Example:

``` bash
datum create
datum add path/to/dataset/ -f voc -n src1
datum remove src1
```
