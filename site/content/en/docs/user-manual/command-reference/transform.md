---
title: 'Transform Dataset'
linkTitle: 'Transform'
description: ''
weight: 18
---

Often datasets need to be modified during preparation for model training and
experimenting. In trivial cases it can be done manually - e.g. image renaming
or label renaming. However, in more complex cases even simple modifications
can require too much efforts, distracting the user from the real work.
Datumaro provides the `datum transform` command to help in such cases.

This command allows to modify dataset images or annotations all at once.

> This command is designed for batch dataset processing, so if you only
> need to modify few elements of a dataset, you might want to use
> other approaches for better performance. A possible solution can be
> a simple script, which uses [Datumaro API](/docs/developer_manual/).

The command can be applied to a dataset or a project build target,
a stage or the combined `project` target, in which case all the project
targets will be affected. A build tree stage will be recorded
if `--stage` is enabled, and the resulting dataset(-s) will be
saved if `--apply` is enabled.

By default, datasets are updated in-place. The `-o/--output-dir`
option can be used to specify another output directory. When
updating in-place, use the `--overwrite` parameter (in-place
updates fail by default to prevent data loss), unless a project
target is modified.

The current project (`-p/--project`) is also used as a context for
plugins, so it can be useful for dataset paths having custom formats.
When not specified, the current project's working tree is used.

Usage:

``` bash
datum transform [-h] -t TRANSFORM [-o DST_DIR] [--overwrite]
  [-p PROJECT_DIR] [--stage STAGE] [--apply APPLY] [target] [-- EXTRA_ARGS]
```

Parameters:
- `<target>` (string) - Target
  [dataset revpath](/docs/user-manual/how_to_use_datumaro/#revpath).
  By default, transforms all targets of the current project.
- `-t, --transform` (string) - Transform method name
- `--stage` (bool) - Include this action as a project build step.
  If true, this operation will be saved in the project
  build tree, allowing to reproduce the resulting dataset later.
  Applicable only to main project targets (i.e. data sources
  and the `project` target, but not intermediate stages). Enabled by default.
- `--apply` (bool) - Run this command immediately. If disabled, only the
  build tree stage will be written. Enabled by default.
- `-o, --output-dir` (string) - Output directory. Can be omitted for
  main project targets (i.e. data sources and the `project` target, but not
  intermediate stages) and dataset targets. If not specified, the results
  will be saved inplace.
- `--overwrite` - Allows to overwrite existing files in the output directory,
  when it is specified and is not empty.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `<extra args>` - The list of extra transformation parameters. Should be
  passed after the `--` separator after the main command arguments. See
  transform descriptions for info about extra parameters. Use the `--help`
  option to print parameter info.

Examples:

- Split a VOC-like dataset randomly:
``` bash
datum transform -t random_split --overwrite path/to/dataset:voc
```

- Rename images in a project data source by a regex from `frame_XXX` to `XXX`:
``` bash
datum create <...>
datum import <...> -n source-1
datum transform -t rename source-1 -- -e '|frame_(\d+)|\\1|'
```

#### Built-in transforms <a id="builtin-transforms"></a>

Basic dataset item manipulations:
- `rename` - Renames dataset items by regular expression
- `id_from_image_name` - Renames dataset items to their image filenames
- `reindex` - Renames dataset items with numbers
- `ndr` - Removes duplicated images from dataset
- `sampler` - Runs inference and leaves only the most representative images
- `resize` - Resizes images and annotations in the dataset

Subset manipulations:
- `random_split` - Splits dataset into subsets randomly
- `split` - Splits dataset into subsets for classification, detection,
  segmentation or re-identification
- `map_subsets` - Renames and removes subsets

Annotation manipulations:
- `remap_labels` - Renames, adds or removes labels in dataset
- `project_labels` - Sets dataset labels to the requested sequence
- `shapes_to_boxes` - Replaces spatial annotations with bounding boxes
- `boxes_to_masks` - Converts bounding boxes to instance masks
- `polygons_to_masks` - Converts polygons to instance masks
- `masks_to_polygons` - Converts instance masks to polygons
- `anns_to_labels` - Replaces annotations having labels with label annotations
- `merge_instance_segments` - Merges grouped spatial annotations into a mask
- `crop_covered_segments` - Removes occluded segments of covered masks
- `bbox_value_decrement` - Subtracts 1 from bbox coordinates

Examples:

- Split a dataset randomly to `train` and `test` subsets, ratio is 2:1
``` bash
datum transform -t random_split -- --subset train:.67 --subset test:.33
```

- Split a dataset for a specific task. The tasks supported are
classification, detection, segmentation and re-identification.

``` bash
datum transform -t split -- \
  -t classification --subset train:.5 --subset val:.2 --subset test:.3

datum transform -t split -- \
  -t detection --subset train:.5 --subset val:.2 --subset test:.3

datum transform -t split -- \
  -t segmentation --subset train:.5 --subset val:.2 --subset test:.3

datum transform -t split -- \
  -t reid --subset train:.5 --subset val:.2 --subset test:.3 \
  --query .5
```

- Convert spatial annotations between each other

``` bash
datum transform -t boxes_to_masks
datum transform -t masks_to_polygons
datum transform -t polygons_to_masks
datum transform -t shapes_to_boxes
```

- Set dataset labels to {`person`, `cat`, `dog`}, remove others, add missing.
  Original labels (can be any): `cat`, `dog`, `elephant`, `human`
  New labels: `person` (added), `cat` (kept), `dog` (kept)

``` bash
datum transform -t project_labels -- -l person -l cat -l dog
```

- Remap dataset labels, `person` to `car` and `cat` to `dog`,
keep `bus`, remove others

``` bash
datum transform -t remap_labels -- \
  -l person:car -l bus:bus -l cat:dog \
  --default delete
```

- Rename dataset items by a regular expression
  - Replace `pattern` with `replacement`
  - Remove `frame_` from item ids

``` bash
datum transform -t rename -- -e '|pattern|replacement|'
datum transform -t rename -- -e '|frame_(\d+)|\\1|'
```

- Create a dataset from K the most hard items for a model. The dataset will
be split into the `sampled` and `unsampled` subsets, based on the model
confidence, which is stored in the `scores` annotation attribute.

There are five methods of sampling (the `-m/--method` option):
- `topk` - Return the k with high uncertainty data
- `lowk` - Return the k with low uncertainty data
- `randk` - Return the random k data
- `mixk` - Return half to topk method and the rest to lowk method
- `randtopk` - First, select 3 times the number of k randomly, and return
  the topk among them.

``` bash
datum transform -t sampler -- \
  -a entropy \
  -i train \
  -o sampled \
  -u unsampled \
  -m topk \
  -k 20
```

- Remove duplicated images from a dataset. Keep at most N resulting images.
  - Available sampling options (the `-e` parameter):
    - `random` - sample from removed data randomly
    - `similarity` - sample from removed data with ascending
  - Available sampling methods (the `-u` parameter):
    - `uniform` - sample data with uniform distribution
    - `inverse` - sample data with reciprocal of the number

``` bash
datum transform -t ndr -- \
  -w train \
  -a gradient \
  -k 100 \
  -e random \
  -u uniform
```

- Resize dataset images and annotations. Supports upscaling, downscaling
and mixed variants.

```
datum transform -t resize -- -dw 256 -dh 256
