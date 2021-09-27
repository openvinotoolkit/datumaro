---
title: 'Patch Datasets'
linkTitle: 'Patch'
description: ''
weight: 13
---

Updates items of the first dataset with items from the second one.

By default, datasets are updated in-place. The `-o/--output-dir`
option can be used to specify another output directory. When
updating inplace, use the `--overwrite` parameter along with the
`--save-images` export option (inplace updates fail by default
to prevent data loss).

Unlike the regular project [data source joining](/docs/developer_manual/#merging),
the datasets are not required to have the same labels. The labels from
the "patch" dataset are projected onto the labels of the patched dataset,
so only the annotations with the matching labels are used, i.e.
all the annotations having unknown labels are ignored. Currently,
this command doesn't allow to update the label information in the
patched dataset.

The command supports passing extra exporting options for the output
dataset. The extra options should be passed after the main arguments
and after the `--` separator. Particularly, this is useful to include
images in the output dataset with `--save-images`.

This command can be applied to the current project targets or
arbitrary datasets outside a project. Note that if the destination
is read-only (e.g. if it is a project, stage or a cache entry),
the output directory must be provided.

Usage:
```
datum patch [-h] [-o DST_DIR] [--overwrite] [-p PROJECT_DIR]
  target patch
  [-- EXPORT_ARGS]
```

\<revpath\> - either [a dataset path or a revision path](/docs/user-manual/how_to_use_datumaro/#revpath).

The current project (`-p/--project`) is also used as a context for
plugins, so it can be useful for dataset paths having custom formats.
When not specified, the current project's working tree is used.

Parameters:
- `<target dataset>` (string) - Target [dataset revpath](/docs/user-manual/how_to_use_datumaro/#revpath)
- `<patch dataset>` (string) - Patch [dataset revpath](/docs/user-manual/how_to_use_datumaro/#revpath)
- `-o, --output-dir` (string) - Output directory. By default, saves in-place
- `--overwrite` - Allows to overwrite existing files in the output directory,
  when it is not empty.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `-- <export args>` - Additional arguments for the format writer
  (use `-- -h` for help). Must be specified after the main command arguments.

Examples:
- Update a VOC-like dataset with a COCO-like annotations:
``` bash
datum patch --overwrite dataset1/:voc dataset2/:coco -- --save-images
```

- Generate a patched dataset, based on a project:
``` bash
datum patch -o patched_proj1/ proj1/ proj2/
```

- Update the "source1" the current project with a dataset:
``` bash
datum patch -p proj/ --overwrite source1 path/to/dataset2:coco
```

- Generate a patched source from a previous revision and a dataset:
``` bash
datum patch -o new_src2/ HEAD~2:source-2 path/to/dataset2:yolo
```

- Update a dataset in a custom format, described in a project plugin:
``` bash
datum patch -p proj/ --overwrite dataset/:my_format dataset2/:coco
```
