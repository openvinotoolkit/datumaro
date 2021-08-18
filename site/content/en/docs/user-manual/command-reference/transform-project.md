---
title: 'Transform Project'
linkTitle: 'Transform Project'
description: ''
weight: 23
---

This command allows to modify images or annotations in a project all at once.

``` bash
datum transform --help

datum transform \
    -p <project_dir> \
    -o <output_dir> \
    -t <transform_name> \
    -- [extra transform options]
```

Example: split a dataset randomly to `train` and `test` subsets, ratio is 2:1

``` bash
datum transform -t random_split -- --subset train:.67 --subset test:.33
```

Example: split a dataset in task-specific manner. The tasks supported are
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

Example: convert polygons to masks, masks to boxes etc.:

``` bash
datum transform -t boxes_to_masks
datum transform -t masks_to_polygons
datum transform -t polygons_to_masks
datum transform -t shapes_to_boxes
```

Example: remap dataset labels, `person` to `car` and `cat` to `dog`,
keep `bus`, remove others

``` bash
datum transform -t remap_labels -- \
    -l person:car -l bus:bus -l cat:dog \
    --default delete
```

Example: rename dataset items by a regular expression
- Replace `pattern` with `replacement`
- Remove `frame_` from item ids

``` bash
datum transform -t rename -- -e '|pattern|replacement|'
datum transform -t rename -- -e '|frame_(\d+)|\\1|'
```

Example: sampling dataset items as many as the number of target samples with
sampling method entered by the user, divide into `sampled` and `unsampled`
subsets
- There are five methods of sampling the m option.
  - `topk`: Return the k with high uncertainty data
  - `lowk`: Return the k with low uncertainty data
  - `randk`: Return the random k data
  - `mixk`: Return half to topk method and the rest to lowk method
  - `randtopk`: First, select 3 times the number of k randomly, and return
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

Example : control number of outputs to 100 after NDR
- There are two methods in NDR e option
  - `random`: sample from removed data randomly
  - `similarity`: sample from removed data with ascending
- There are two methods in NDR u option
  - `uniform`: sample data with uniform distribution
  - `inverse`: sample data with reciprocal of the number

```bash
datum transform -t ndr -- \
    -w train \
    -a gradient \
    -k 100 \
    -e random \
    -u uniform
```
