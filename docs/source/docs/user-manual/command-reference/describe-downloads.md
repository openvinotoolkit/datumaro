# describe-downloads

## Describe downloadable datasets

This command reports reports various information about datasets that can be
downloaded with the `download` command. The information is reported either as
human-readable text (the default) or as a JSON object. The format can be selected
with the `--report-format` option.

When the JSON output format is selected, the output document has the following schema:

```json
{
    "<dataset name>": {
        "default_output_format": "<Datumaro format name>",
        "description": "<human-readable description>",
        "download_size": <total size of the downloaded files in bytes>,
        "home_url": "<URL of a web page describing the dataset>",
        "human_name": "<human-readable dataset name>",
        "num_classes": <number of classes in the dataset>,
        "subsets": {
            "<subset name>": {
                "num_items": <number of items in the subset>
            },
            ...
        },
        "version": "<version number>"
    },
    ...
}
```

`home_url` may be `null` if there is no suitable web page for the dataset.

`num_classes` may be `null` if the dataset does not involve classification.

`version` currently contains the version number supplied by TFDS.
In future versions of Datumaro, datasets might come from other sources;
the way version numbers will be set for those is to be determined.

New object members may be added in future versions of Datumaro.

Usage:

```
datum describe-downloads [-h] [--report-format {text,json}]
                         [--report-file REPORT_FILE]
```

Parameters:

- `-h`, `--help` - Print the help message and exit.
- `--report-format` (`text` or `json`) - Format in which to report the information.
  By default, `text` is used.
- `--report-file` (string) - File to which to write the report. By default,
  the report is written to the standard output stream.
