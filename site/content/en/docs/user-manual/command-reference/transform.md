---
title: 'Transform Dataset'
linkTitle: 'transform'
description: ''
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
datum transform -t rename source-1 -- -e '|^frame_||'
```

#### Built-in transforms <a id="builtin-transforms"></a>

Basic dataset item manipulations:
- [`rename`](#rename-transform) - Renames dataset items by regular expression
- [`id_from_image_name`](#id_from_image_name-transform) - Renames dataset
  items to their image filenames
- [`reindex`](#reindex-transform) - Renames dataset items with numbers
- [`ndr`](#ndr-transform) - Removes duplicated images from dataset
- [`relevancy_sampler`](#relevancy_sampler-transform) - Leaves only the most
  important images
  (requires model inference results)
- [`random_sampler`](#random_sampler-transform) - Leaves no more than k items
  from the dataset randomly
- [`label_random_sampler`](#label_random_sampler-transform) - Leaves at least
  k images with annotations per class
- [`resize`](#resize-transform) - Resizes images and annotations in the dataset
- [`delete_image`](#delete_image-transform) - Deletes images with annotation errors
- [`delete_annotation`](#delete_annotation-transform) - Deletes annotations with 
  annotation errors
- [`delete_attribute`](#delete_attribute-transform) - Deletes attriutes with 
  annotation errors

Subset manipulations:
- [`random_split`](#random_split-transform) - Splits dataset into subsets
  randomly
- [`split`](#split-transform) - Splits dataset into subsets for classification,
  detection, segmentation or re-identification
- [`map_subsets`](#map_subsets-transform) - Renames and removes subsets

Annotation manipulations:
- [`remap_labels`](#remap_labels-transform) - Renames, adds or removes
  labels in dataset
- [`project_labels`](#project_labels-transform) - Sets dataset labels to
  the requested sequence
- [`shapes_to_boxes`](#shapes_to_boxes-transform) - Replaces spatial
  annotations with bounding boxes
- [`boxes_to_masks`](#boxes_to_masks-transform) - Converts bounding boxes
  to instance masks
- [`polygons_to_masks`](#polygons_to_masks-transform) - Converts polygons
  to instance masks
- [`masks_to_polygons`](#masks_to_polygons-transform) - Converts instance
  masks to polygons
- [`anns_to_labels`](#anns_to_labels-transform) - Replaces annotations having
  labels with label annotations
- [`merge_instance_segments`](#merge_instance_segments-transform) - Merges
  grouped spatial annotations into a mask
- [`crop_covered_segments`](#crop_covered_segments-transform) - Removes
  occluded segments of covered masks
- [`bbox_value_decrement`](#bbox_value_decrement-transform) - Subtracts
  1 from bbox coordinates

##### `rename` <a id="rename-transform"></a>

Renames items in the dataset. Supports regular expressions.
The first character in the expression is a delimiter for
the pattern and replacement parts. Replacement part can also
contain `str.format` replacement fields with the `item`
(of type `DatasetItem`) object available.

Usage:
``` bash
rename [-h] [-e REGEX]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-e`, `--regex` (string) - Regex for renaming in the form
  `<sep><search><sep><replacement><sep>`

Examples:
Replace 'pattern' with 'replacement':
```bash
datum transform -t rename -- -e '|pattern|replacement|'
```

Remove the `frame_` prefix from item ids:
```bash
datum transform -t rename -- -e '|^frame_|\1|'
```

Collect images from subdirectories into the base image directory using regex:
```bash
datum transform -t rename -- -e '|^((.+[/\\])*)?(.+)$|\2|'
```

Add subset prefix to images:
```bash
datum transform -t rename -- -e '|(.*)|{item.subset}_\1|'
```

##### `id_from_image_name` <a id="id_from_image_name-transform"></a>

Renames items in the dataset using image file name (without extension).

Usage:
```bash
id_from_image_name [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

##### `reindex` <a id="reindex-transform"></a>

Replaces dataset item IDs with sequential indices.

Usage:
```bash
reindex [-h] [-s START]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-s`, `--start` (int) - Start value for item ids (default: 1)

##### `ndr` <a id="ndr-transform"></a>

Removes near-duplicated images in subset.

Remove duplicated images from a dataset. Keep at most `-k/--num_cut`
resulting images.

Available oversampling policies (the `-e` parameter):
- `random` - sample from removed data randomly
- `similarity` - sample from removed data with ascending similarity score

Available undersampling policies (the `-u` parameter):
- `uniform` - sample data with uniform distribution
- `inverse` - sample data with reciprocal of the number of number of
  items with the same similarity

Usage:
```bash
ndr [-h] [-w WORKING_SUBSET] [-d DUPLICATED_SUBSET] [-a {gradient}]
  [-k NUM_CUT] [-e {random,similarity}] [-u {uniform,inverse}] [-s SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-w`, `--working_subset` (str) - Name of the subset to operate
  (default: `None`)
- `-d`, `--duplicated_subset` (str) - Name of the subset for the removed
  data after NDR runs (default: duplicated)
- `-a`, `--algorithm` (one of: `gradient`) - Name of the algorithm to
  use (default: `gradient`)
- `-k`, `--num_cut` (int) - Maximum output dataset size
- `-e`, `--over_sample` (one of: `random`, `similarity`) - The policy to use
  when `num_cut` is bigger than result length (default: `random`)
- `-u`, `--under_sample` (one of: `uniform`, `inverse`) - The policy to use
  when `num_cut` is smaller than result length (default: `uniform`)
- `-s`, `--seed` (int) - Random seed

Example: apply NDR, return no more than 100 images
``` bash
datum transform -t ndr -- \
  --working_subset train
  --algorithm gradient
  --num_cut 100
  --over_sample random
  --under_sample uniform
```

##### `relevancy_sampler` <a id="relevancy_sampler-transform"></a>

Sampler that analyzes model inference results on the dataset
and picks the most relevant samples for training.

Creates a dataset from the `-k/--count` hardest items for a model.
The whole dataset or a single subset will be split into the `sampled`
and `unsampled` subsets based on the model confidence. The dataset
**must** contain model confidence values in the `scores` attributes
of annotations.

There are five methods of sampling (the `-m/--method` option):
- `topk` - Return the k items with the highest uncertainty data
- `lowk` - Return the k items with the lowest uncertainty data
- `randk` - Return random k items
- `mixk` - Return a half using topk, and the other half using lowk method
- `randtopk` - Select 3*k items randomly, and return the topk among them

Notes:
- Each image's inference result must contain the probability for
  all classes (in the `scores` attribute).
- Requesting a sample larger than the number of all images will return
  all images.

Usage:
```bash
relevancy_sampler [-h] -k COUNT [-a {entropy}] [-i INPUT_SUBSET]
  [-o SAMPLED_SUBSET] [-u UNSAMPLED_SUBSET]
  [-m {topk,lowk,randk,mixk,randtopk}] [-d OUTPUT_FILE]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-k`, `--count` (int) - Number of items to sample
- `-a`, `--algorithm` (one of: `entropy`) - Sampling
  algorithm (default: `entropy`)
- `-i`, `--input_subset` (str) - Subset name to select sample
  from (default: `None`)
- `-o`, `--sampled_subset` (str) - Subset name to put sampled data
  to (default: `sample`)
- `-u`, `--unsampled_subset` (str) - Subset name to put the
  rest data to (default: `unsampled`)
- `-m`, `--sampling_method` (one of: `topk`, `lowk`, `randk`, `mixk`,
  `randtopk`) - Sampling method (default: `topk`)
- `-d`, `--output_file` (path) - A `.csv` file path to dump sampling results

Examples:
Select the most relevant data subset of 20 images
based on model certainty, put the result into `sample` subset
and put all the rest into `unsampled` subset, use `train` subset
as input. The dataset **must** contain model confidence values in the `scores`
attributes of annotations.
```bash
datum transform -t relevancy_sampler -- \
  --algorithm entropy \
  --subset_name train \
  --sample_name sample \
  --unsampled_name unsampled \
  --sampling_method topk -k 20
```

##### `random_sampler` <a id="random_sampler-transform"></a>

Sampler that keeps no more than required number of items in the dataset.

Notes:
- Items are selected uniformly (tries to keep original item distribution
  by subsets)
- Requesting a sample larger than the number of all images will return
  all images

Usage:
```bash
random_sampler [-h] -k COUNT [-s SUBSET] [--seed SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-k`, `--count` (int) - Maximum number of items to sample
- `-s`, `--subset` (str) - Limit changes to this subset
  (default: affect all dataset)
- `--seed` (int) - Initial value for random number generator

Examples:
Select subset of 20 images randomly
```bash
datum transform -t random_sampler -- -k 20
```

Select subset of 20 images, modify only `train` subset
```bash
datum transform -t random_sampler -- -k 20 -s train
```

##### `random_label_sampler` <a id="random_label_sampler-transform"></a>

Sampler that keeps at least the required number of annotations of
each class in the dataset for each subset separately.

Consider using the "stats" command to get class distribution in the dataset.

Notes:
- Items can contain annotations of several selected classes
  (e.g. 3 bounding boxes per image). The number of annotations in the
  resulting dataset varies between `max(class counts)` and `sum(class counts)`
- If the input dataset does not has enough class annotations, the result
  will contain only what is available
- Items are selected uniformly
- For reasons above, the resulting class distribution in the dataset may
  not be the same as requested
- The resulting dataset will only keep annotations for classes with
  specified `count` > 0

Usage:
```bash
label_random_sampler [-h] -k COUNT [-l LABEL_COUNTS] [--seed SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-k`, `--count` (int) - Minimum number of annotations of each class
- `-l`, `--label` (str; repeatable) - Minimum number of annotations of
  a specific class. Overrides the `-k/--count` setting for the class.
  The format is `<label_name>:<count>`
- `--seed` (int) - Initial value for random number generator

Examples:
Select a dataset with at least 10 images of each class:
``` bash
datum transform -t label_random_sampler -- -k 10
```

Select a dataset with at least 20 `cat` images, 5 `dog`, 0 `car` and 10 of each
unmentioned class:
``` bash
datum transform -t label_random_sampler -- \
  -l cat:20 \ # keep 20 images with cats
  -l dog:5 \ # keep 5 images with dogs
  -l car:0 \ # remove car annotations
  -k 10 # for remaining classes
```

##### `resize` <a id="resize-transform"></a>

Resizes images and annotations in the dataset to the specified size.
Supports upscaling, downscaling and mixed variants.

Usage:
```bash
resize [-h] [-dw WIDTH] [-dh HEIGHT]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-dw`, `--width` (int) - Destination image width
- `-dh`, `--height` (int) - Destination image height

Examples:
Resize all images to 256x256 size
```
datum transform -t resize -- -dw 256 -dh 256
```

##### `delete_image` <a id="delete_image-transform"></a>

Deletes images with annotation errors in the dataset

Usage:
```bash
delete_image [-h] [-i IDs]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-i`, `--ids` (str) - Datasetitem ids to run trasform

Examples:
Delete an image, which has '2010_001705' as id.
```
datum transform -t delete_image -- -i '2010_001705'
```

##### `delete_annotation` <a id="delete_annotation-transform"></a>

Deletes annotations with annotation errors in the dataset

Usage:
```bash
delete_image [-h] [-i IDs]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-i`, `--ids` (str) - Datasetitem ids to run trasform

Examples:
Delete annotations, which has '2010_001705' as id.
```
datum transform -t delete_annotation -- -i '2010_001705'
```

##### `delete_attribute` <a id="delete_attribute-transform"></a>

Deletes attributes with annotation errors in the dataset

Usage:
```bash
delete_image [-h] [-i IDs]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-i`, `--ids` (str) - Datasetitem ids to run trasform

Examples:
Delete attributes, which has '2010_001705' as id.
```
datum transform -t delete_attribute -- -i '2010_001705'
```

##### `random_split` <a id="random_split-transform"></a>

Joins all subsets into one and splits the result into few parts.
It is expected that item ids are unique and subset ratios sum up to 1.

Usage:
```bash
random_split [-h] [-s SPLITS] [--seed SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-s`, `--subset` (str, repeatable) - Subsets in the form: '<subset>:<ratio>'
  (repeatable, default: {`train`: 0.67, `test`: 0.33})
- `--seed` (int) - Random seed

Example:
Split a dataset randomly to `train` and `test` subsets, ratio is 2:1
``` bash
datum transform -t random_split -- --subset train:.67 --subset test:.33
```

##### `split` <a id="split-transform"></a>

Splits a dataset for model training, using task information:

- classification splits
Splits dataset into subsets (train/val/test) in class-wise manner.
Splits dataset images in the specified ratio, keeping the initial
class distribution.

- detection & segmentation splits
Each image can have multiple object annotations - bbox, mask, polygon.
Since an image shouldn't be included in multiple subsets at the same time,
and image annotations shouldn't be split, in general, dataset annotations are
unlikely to be split exactly in the specified ratio.
This split tries to split dataset images as close as possible to the specified
ratio, keeping the initial class distribution.

- reidentification splits
In this task, the test set should consist of images of unseen people or
objects during the training phase.
This function splits a dataset in the following way:
1. Splits the dataset into `train + val` and `test` sets
  based on person or object ID.
2. Splits `test` set into `test-gallery` and `test-query` sets
  in class-wise manner.
3. Splits the `train + val` set into `train` and `val` sets
  in the same way.
The final subsets would be `train`, `val`, `test-gallery` and `test-query`.

Notes:
- Each image is expected to have only one `Annotation`. Unlabeled or
  multi-labeled images will be split into subsets randomly.
- If Labels also have attributes, also splits by attribute values.
- If there is not enough images in some class or attributes group,
  the split ratio can't be guaranteed.

In reidentification task,
- Object ID can be described by Label, or by attribute (`--attr` parameter)
- The splits of the test set are controlled by `--query` parameter
  Gallery ratio would be `1.0 - query`.

Usage:
```bash
split [-h] [-t {classification,detection,segmentation,reid}]
  [-s SPLITS] [--query QUERY] [--attr ATTR_FOR_ID] [--seed SEED]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-t`, `--task` (one of: `classification`, `detection`, `segmentation`,
  `reid`) - Dataset task (default: `classification`)
- `-s`, `--subset` (str; repeatable) - Subsets in the form: '<subset>:<ratio>'
  (default: {`train`: 0.5, `val`: 0.2, `test`: 0.3})
- `--query` (float) - Query ratio in the test set (default: 0.5)
- `--attr` (str) - Attribute name representing the ID (default: use label)
- `--seed`(int) - Random seed

Example:
```
datum transform -t split -- -t classification \
  --subset train:.5 --subset val:.2 --subset test:.3

datum transform -t split -- -t detection \
  --subset train:.5 --subset val:.2 --subset test:.3

datum transform -t split -- -t segmentation \
  --subset train:.5 --subset val:.2 --subset test:.3

datum transform -t split -- -t reid \
  --subset train:.5 --subset val:.2 --subset test:.3 --query .5
```

Example: use `person_id` attribute for splitting
```bash
datum transform -t split -- -t detection --attr person_id
```

##### `map_subsets` <a id="map_subsets-transform"></a>

Renames subsets in the dataset.

Usage:
```bash
map_subsets [-h] [-s MAPPING]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-s`, `--subset` (str; repeatable) - Subset mapping of the form: `src:dst`

##### `remap_labels` <a id="remap_labels-transform"></a>

Changes labels in the dataset.

A label can be:
- renamed (and joined with existing) -
  when `--label <old_name>:<new_name>` is specified
- deleted - when `--label <name>:` is specified, or default action is `delete`
  and the label is not mentioned in the list. When a label
  is deleted, all the associated annotations are removed
- kept unchanged - when `--label <name>:<name>` is specified,
  or default action is `keep` and the label is not mentioned in the list
Annotations with no label are managed by the default action policy.

Usage:
```bash
remap_labels [-h] [-l MAPPING] [--default {keep,delete}]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-l`, `--label` (str; repeatable) - Label in the form of: `<src>:<dst>`
- `--default` (one of: `keep`, `delete`) - Action for unspecified labels
  (default: `keep`)

Examples:
Remove the `person` label (and corresponding annotations):
```bash
datum transform -t remap_labels -- -l person: --default keep
```

Rename `person` to `pedestrian` and `human` to `pedestrian`, join annotations
that had different classes under the same class id for `pedestrian`,
don't touch other classes:
```bash
datum transform -t remap_labels -- \
  -l person:pedestrian -l human:pedestrian --default keep
```

Rename `person` to `car` and `cat` to `dog`, keep `bus`, remove others:
```bash
datum transform -t remap_labels -- \
  -l person:car -l bus:bus -l cat:dog --default delete
```

##### `project_labels` <a id="project_labels-transform"></a>

Changes the order of labels in the dataset from the existing
to the desired one, removes unknown labels and adds new labels.
Updates or removes the corresponding annotations.

Labels are matched by names (case dependent). Parent labels are
only kept if they are present in the resulting set of labels.
If new labels are added, and the dataset has mask colors defined,
new labels will obtain generated colors.

Useful for merging similar datasets, whose labels need to be aligned.

Usage:
```bash
project_labels [-h] [-l DST_LABELS]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `-l`, `--label` (str; repeatable) - Label name (ordered)

Examples:
Set dataset labels to \[`person`, `cat`, `dog`\], remove others, add missing.
Original labels (for example): `cat`, `dog`, `elephant`, `human`.
New labels: `person` (added), `cat` (kept), `dog` (kept).
``` bash
datum transform -t project_labels -- -l person -l cat -l dog
```

##### `shapes_to_boxes` <a id="shapes_to_boxes-transform"></a>

Converts spatial annotations (masks, polygons, polylines, points)
to enclosing bounding boxes.

Usage:
```bash
shapes_to_boxes [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

Example:
Convert spatial annotations between each other
``` bash
datum transform -t boxes_to_masks
datum transform -t masks_to_polygons
datum transform -t polygons_to_masks
datum transform -t shapes_to_boxes
```

##### `boxes_to_masks` <a id="boxes_to_masks-transform"></a>

Converts bounding boxes to masks.

Usage:
```bash
boxes_to_masks [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

##### `polygons_to_masks` <a id="polygons_to_masks-transform"></a>

Converts polygons to masks.

Usage:
```bash
polygons_to_masks [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

##### `masks_to_polygons` <a id="masks_to_polygons-transform"></a>

Converts masks to polygons.

Usage:
```bash
masks_to_polygons [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

##### `anns_to_labels` <a id="anns_to_labels-transform"></a>

Collects all labels from annotations (of all types) and transforms
them into a set of annotations of type `Label`

Usage:
```bash
anns_to_labels [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

##### `merge_instance_segments` <a id="merge_instance_segments-transform"></a>

Replaces instance masks and, optionally, polygons with a single mask.
A group of annotations with the same group id is considered an "instance".
The largest annotation in the group is considered the group "head", so the
resulting mask takes properties from that annotation.

Usage:
```bash
merge_instance_segments [-h] [--include-polygons]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
- `--include-polygons` (flag) - Include polygons

##### `crop_covered_segments` <a id="crop_covered_segments-transform"></a>

Sorts polygons and masks ("segments") according to `z_order`,
crops covered areas of underlying segments. If a segment is split
into several independent parts by the segments above, produces
the corresponding number of separate annotations joined into a group.

Usage:
```bash
crop_covered_segments [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit

##### `bbox_value_decrement` <a id="bbox_value_decrement-transform"></a>

Subtracts one from the coordinates of bounding boxes

Usage:
```bash
bbox_values_decrement [-h]
```

Optional arguments:
- `-h`, `--help` (flag) - Show this help message and exit
