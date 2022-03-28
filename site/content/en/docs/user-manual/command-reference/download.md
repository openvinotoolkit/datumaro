---
title: 'Download datasets'
linkTitle: 'download'
description: ''
---

This command downloads a publicly available dataset and saves it to a local
directory.
In terms of syntax, this command is similar to [`convert`](../convert),
but instead of taking a local directory as the source, it takes a dataset ID.
A list of supported datasets and output formats can be found in the `--help`
output of this command.

Currently, the only source of datasets is the TensorFlow Datasets library.
Therefore, to use this command you must install TensorFlow & TFDS, which you can
do as follows:

```sh
pip install datumaro[tf,tfds]
```

To use a proxy for downloading, configure it with the conventional
[curl environment variables](https://everything.curl.dev/usingcurl/proxies/env).

Usage:

``` bash
datum download [-h] -i DATASET_ID [-f OUTPUT_FORMAT] [-o DST_DIR]
               [--overwrite] [-s SUBSET] [-- EXTRA_EXPORT_ARGS]
```

Parameters:

- `-h`, `--help` - Print the help message and exit.
- `-i`, `--dataset-id` (string) - ID of the dataset to download.
- `-f`, `--output-format` (string) - Output format. By default, the format
  of the original dataset is used.
- `-o, --output-dir` (string) - Output directory. By default, a subdirectory
  in the current directory is used.
- `--overwrite` - Allows overwriting existing files in the output directory,
  when it is not empty.
- `--subset` (string) - Which subset of the dataset to save. By default, all
  subsets are saved. Note that due to limitations of TFDS, all subsets are
  downloaded even if this option is specified.
- `-- <extra export args>` - Additional arguments for the format writer
  (use `-- -h` for help). Must be specified after the main command arguments.

Example: download the MNIST dataset, saving it in the ImageNet text format:

``` bash
datum download -i tfds:mnist -f imagenet_txt -- --save-images
```
