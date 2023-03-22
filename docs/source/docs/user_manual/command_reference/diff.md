diff
====

## Compare Datasets

The command compares two datasets and saves the results in the
specified directory. The current project is considered to be
"ground truth".

Datasets can be compared using different methods:
- `equality` - Annotations are compared to be equal
- `distance` - A distance metric is used

This command has multiple forms:
``` bash
1) datum diff <revpath>
2) datum diff <revpath> <revpath>
```

1 - Compares the current project's main target (`project`)
  in the working tree with the specified dataset.

2 - Compares two specified datasets.

\<revpath\> - [a dataset path or a revision path](/docs/user_manual/how_to_use_datumaro/#revpath).

Usage:
``` bash
datum diff [-h] [-o DST_DIR] [-m METHOD] [--overwrite] [-p PROJECT_DIR]
  [--iou-thresh IOU_THRESH] [-f FORMAT]
  [-iia IGNORE_ITEM_ATTR] [-ia IGNORE_ATTR] [-if IGNORE_FIELD]
  [--match-images] [--all]
  first_target [second_target]
```

Parameters:
- `<target>` (string) - Target [dataset revpaths](/docs/user_manual/how_to_use_datumaro/#revpath)
- `-m, --method` (string) - Comparison method.
- `-o, --output-dir` (string) - Output directory. By default, a new directory
  is created in the current directory.
- `--overwrite` - Allows to overwrite existing files in the output directory,
  when it is specified and is not empty.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

- Distance comparison options:
  - `--iou-thresh` (number) - The IoU threshold for spatial annotations
    (default is 0.5).
  - `-f, --format` (string) - Output format, one of `simple`
    (text files and images) and `tensorboard` (a TB log directory)

- Equality comparison options:
  - `-iia, --ignore-item-attr` (string) - Ignore an item attribute (repeatable)
  - `-ia, --ignore-attr` (string) - Ignore an annotation attribute (repeatable)
  - `-if, --ignore-field` (string) - Ignore an annotation field (repeatable)
    Default is `id` and `group`
  - `--match-images` - Match dataset items by image pixels instead of ids
  - `--all` - Include matches in the output. By default, only differences are
    printed.

<!-- markdownlint-disable-line MD028 -->Examples:
- Compare two projects by distance, match boxes if IoU > 0.7,
  save results to TensorBoard:
`datum diff other/project -o diff/ -f tensorboard --iou-thresh 0.7`

- Compare two projects for equality, exclude annotation groups
  and the `is_crowd` attribute from comparison:
`datum diff other/project/ -if group -ia is_crowd`

- Compare two datasets, specify formats:
`datum diff path/to/dataset1:voc path/to/dataset2:coco`

- Compare the current working tree and a dataset:
`datum diff path/to/dataset2:coco`

- Compare a source from a previous revision and a dataset:
`datum diff HEAD~2:source-2 path/to/dataset2:yolo`

- Compare a dataset with model inference
``` bash
datum create
datum import <...>
datum model add mymodel <...>
datum transform <...> -o inference
datum diff inference -o diff
```
