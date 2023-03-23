# merge

## Merge Datasets

Consider the following task: there is a set of images (the original dataset)
we want to annotate. Suppose we did this manually and/or automated it
using models, and now we have few sets of annotations for the same images.
We want to merge them and produce a single set of high-precision annotations.

Another use case: there are few datasets with different sets of images
and labels, which we need to combine in a single dataset. If the labels
were the same, we could just join the datasets. But in this case we need
to merge labels and adjust the annotations in the resulting dataset.

In Datumaro, it can be done with the `merge` command. This command merges 2
or more datasets and checks annotations for errors.

> **Note** In simple cases, when dataset images do not intersect and new
> labels are not added, the recommended way of merging is using
> [the `patch` command](/docs/user-manual/command-reference/patch/).
> It will offer better performance and provide the same results.

Datasets are merged by items, and item annotations are merged by finding the
unique ones across datasets. Annotations are matched between matching dataset
items by distance. Spatial annotations are compared by the applicable distance
measure (IoU, OKS, PDJ etc.), labels and annotation attributes are selected
by voting. Each set of matching annotations produces a single annotation in
the resulting dataset. The `score` (a number in the range \[0; 1\]) attribute
indicates the agreement between different sources in the produced annotation.
The working time of the function can be estimated as
`O( (summary dataset length) * (dataset count) ^ 2 * (item annotations) ^ 2 )`

This command also allows to merge datasets with different, or partially
overlapping sets of labels (which is impossible by simple joining).

During the process, some merge conflicts can appear. For example,
it can be mismatching dataset images having the same ids, label voting
can be unsuccessful if quorum is not reached (the `--quorum` parameter),
bboxes may be too close (the `-iou` parameter) etc. Found merge
conflicts, missing items or annotations, and other errors are saved into
an output `.json` file.

In Datumaro, annotations can be grouped. It can be useful to represent
different parts of a single object - for example, it can be different parts
of a human body, parts of a vehicle etc. This command allows to check
annotation groups for completeness with the `-g/--groups` option. If used,
this parameter must specify a list of labels for annotations that must be
in the same group. It can be particularly useful to check if separate
keypoints are grouped and all the necessary object components in the same
group.

This command has multiple forms:
``` bash
1) datum merge <revpath>
2) datum merge <revpath> <revpath> ...
```

\<revpath\> - either [a dataset path or a revision path](/docs/user-manual/how_to_use_datumaro/#revpath).

1 - Merges the current project's main target ("project")
  in the working tree with the specified dataset.

2 - Merges the specified datasets.
  Note that the current project is not included in the list of merged
  sources automatically.

The command supports passing extra exporting options for the output
dataset. The format can be specified with the `-f/--format` option.
Extra options should be passed after the main arguments
and after the `--` separator. Particularly, this is useful to include
images in the output dataset with `--save-images`.

Usage:
``` bash
datum merge [-h] [-iou IOU_THRESH] [-oconf OUTPUT_CONF_THRESH]
  [--quorum QUORUM] [-g GROUPS] [-o DST_DIR] [--overwrite]
  [-p PROJECT_DIR] [-f FORMAT]
  target [target ...] [-- EXTRA_FORMAT_ARGS]
```

Parameters:
- `<target>` (string) - Target [dataset revpaths](/docs/user-manual/how_to_use_datumaro/#revpath)
  (repeatable)
- `-iou`, `--iou-thresh` (number) - IoU matching threshold for spatial
  annotations (both maximum inter-cluster and pairwise). Default is 0.25.
- `--quorum` (number) - Minimum count of votes for a label or attribute
  to be counted. Default is 0.
- `-g, --groups` (string) - A comma-separated list of label names in
  annotation groups to check. The `?` postfix can be added to a label to
  make it optional in the group (repeatable)
- `-oconf`, `--output-conf-thresh` (number) - Confidence threshold for output
  annotations to be included in the resulting dataset. Default is 0.
- `-o, --output-dir` (string) - Output directory. By default, a new directory
  is created in the current directory.
- `--overwrite` - Allows to overwrite existing files in the output directory,
  when it is specified and is not empty.
- `-f, --format` (string) - Output format. The default format is `datumaro`.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `-- <extra format args>` - Additional arguments for the format writer
  (use `-- -h` for help). Must be specified after the main command arguments.

Examples:

Merge 4 (partially-)intersecting projects,
- consider voting successful when there are no less than 3 same votes
- consider shapes intersecting when IoU >= 0.6
- check annotation groups to have `person`, `hand`, `head` and `foot`
(`?` is used for optional parts)

``` bash
datum merge project1/ project2/ project3/ project4/ \
  --quorum 3 \
  -iou 0.6 \
  --groups 'person,hand?,head,foot?'
```

Merge images and annotations from 2 datasets in COCO format:
`datum merge dataset1/:image_dir dataset2/:coco dataset3/:coco`

Check groups of the merged dataset for consistency:
  look for groups consisting of `person`, `hand` `head`, `foot`
`datum merge project1/ project2/ -g 'person,hand?,head,foot?'`

Merge two datasets, specify formats:
`datum merge path/to/dataset1:voc path/to/dataset2:coco`

Merge the current working tree and a dataset:
`datum merge path/to/dataset2:coco`

Merge a source from a previous revision and a dataset:
`datum merge HEAD~2:source-2 path/to/dataset2:yolo`

Merge datasets and save in different format:
`datum merge -f voc dataset1/:yolo path2/:coco -- --save-images`
