---
title: 'Print dataset info'
linkTitle: 'info'
description: ''
---

This command outputs high level dataset information such as sample count,
categories and subsets.

Usage:

``` bash
datum info [-h] [--json] [-p PROJECT_DIR] [revpath]
```

Parameters:
- `<target>` (string) - Target [dataset revpath](/docs/user-manual/how_to_use_datumaro/#revpath).
  By default, prints info about the joined `project` dataset.
- `--json` - Print output data in JSON format
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.


Examples:

- Print info about a project dataset:
`datum info -p test_project/`

- Print info about a COCO-like dataset:
`datum info path/to/dataset:coco`

Sample output:

```
format: voc
media type: image
length: 5
categories:
    labels: background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair (and 12 more)
subsets:
    trainval:
        length: 5
```

JSON output format:

<details>

```json
{
  "format": string,
  "media type": string,
  "length": integer,
  "categories": {
    "count": integer,
    "labels": [
      {
        "id": integer,
        "name": string,
        "parent": string,
        "attributes": [ string, ... ]
      },
      ...
    ]
  },
  "subsets": [
    {
      "name": string,
      "length": integer
    },
    ...
  ]
}
```

</details>
