# convert

## Convert datasets

This command allows to convert a dataset from one format to another.
The command is a usability alias for [`create`](../create),
[`add`](../sources/#sources-add) and [`export`](../export) and just provides
a simpler way to obtain the same results in simple cases. A list of supported
formats can be found in the `--help` output of this command.

Usage:

``` bash
datum convert [-h] [-i SOURCE] [-if INPUT_FORMAT] -f OUTPUT_FORMAT
  [-o DST_DIR] [--overwrite] [-e FILTER] [--filter-mode FILTER_MODE]
  [-- EXTRA_EXPORT_ARGS]
```

Parameters:
- `-i, --input-path` (string) - Input dataset path. The current directory is
  used by default.
- `-if, --input-format` (string) - Input dataset format. Will try to detect,
  if not specified.
- `-f, --output-format` (string) - Output format
- `-o, --output-dir` (string) - Output directory. By default, a subdirectory
  in the current directory is used.
- `--overwrite` - Allows overwriting existing files in the output directory,
  when it is not empty.
- `-e, --filter` (string) - XML XPath filter expression for dataset items
- `--filter-mode` (string) - The filtering mode. Default is the `i` mode.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `-- <extra export args>` - Additional arguments for the format writer
  (use `-- -h` for help). Must be specified after the main command arguments.

Example: convert a VOC-like dataset to a COCO-like one:

``` bash
datum convert --input-format voc --input-path <path/to/voc/> \
              --output-format coco \
              -- --save-images
```
