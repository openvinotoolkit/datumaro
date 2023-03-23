---
title: "DatumaroBinary"
linkTitle: "DatumaroBinary"
description: ""
---

## Format specification

DatumaroBinary format is [Datumaro](https://github.com/openvinotoolkit/datumaro)'s own data format as same as [Datumaro format](./datumaro.md).
Basically, it provides the same function as [Datumaro format](./datumaro.md),
but the difference is that the annotation file is not JSON but binary format.
While JSON annotation file in the [Datumaro format](./datumaro.md) has the advantage of being easily viewable using any text viewer,
the DatumaroBinary format takes up significantly less storage space since it is schemaless and stores data in binary form.

### Efficient storage cost

To demonstrate the storage cost-effectiveness of DatumaroBinary,
we conducted an experiment to compare the annotation file sizes of three dataset formats:
COCO (JSON), Datumaro (JSON), and DatumaroBinary (binary).
The table below shows the sizes of each format:

| Format | COCO (JSON) | Datumaro (JSON) | DatumaroBinary (binary) |
| :----: | :---------: | :-------------: | :---------------------: |
|  Size  |    468Mb    |     1046Mb      |          301Mb          |

This table shows that DatumaroBinary reduces the size of annotation files to **64.3% (COCO) and 28.8% (Datumaro).**

For this experiment, we used the training and validation annotation files of [2017 COCO instance segmentation task](https://cocodataset.org):

```console
Dataset/
├── images/
│   ├── train/
│   │   ├── <image_name1.ext>
│   │   └── ...
│   └── val/
│       ├── <image_name1.ext>
│       └── ...
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Dataset encryption

Another advantage of the DatumaroBinary format is that it supports dataset encryption. If your dataset is hijacked by a potential attacker and you are concerned that your intellectual properties may be damaged, you can use this feature to protect your dataset from attackers. Enabling the dataset encryption feature allows you to encrypt both annotations and media or only the annotations. If you export the dataset to DatumaroBinary format with encryption, the secret key is automatically generated at the same time. You must keep this secret key separate from the exported dataset. This is because the secret key should be needed to read the exported dataset. Therefore, you have to be careful not to lose the secret key. If you would like to see an example of dataset encryption using Datumaro's Python API, please see [here](https://github.com/openvinotoolkit/datumaro/blob/develop/notebooks/09_encrypt_dataset.ipynb). For CLI usage of encryption, please see [Import encrypted datasets](#import-encrypted-datasets) and [Export datasets with encryption](#export-datasets-with-encryption) sections.

### Usage for model training

You can directly use the DatumaroBinary format for the model training using [OpenVINO™ Training Extensions](https://github.com/openvinotoolkit/training_extensions).

Supported media types:

- `Image`
- `PointCloud`

Supported annotation types:

- `Label`
- `Mask`
- `PolyLine`
- `Polygon`
- `Bbox`
- `Points`
- `Caption`
- `Cuboid3d`
- `Ellipse`

Supported annotation attributes:

- No restrictions

## Import DatumaroBinary dataset

A Datumaro project with a DatumaroBinary source can be created in the following way:

```console
datum create
datum import --format datumaro_binary <path/to/dataset>
```

It is possible to specify project name and project directory. Run
`datum create --help` for more information.

A DatumaroBinary dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->

```
└─ Dataset/
    ├── dataset_meta.json   # a list of custom labels (optional)
    ├── images/
    │   ├── <subset_name_1>/
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   └── <subset_name_2> /
    │       ├── <image_name1.ext>
    │       ├── <image_name2.ext>
    │       └── ...
    └── annotations/
        ├── <subset_name_1>.datum
        ├── <subset_name_2>.datum
        └── ...
```

If your dataset is not following the above directory structure,
it cannot detect and import your dataset as the DatumaroBinary format properly.

To add custom classes, you can use [`dataset_meta.json`](/docs/user-manual/supported_formats/#dataset-meta-file).

To make sure that the selected dataset has been added to the project, you can
run `datum project info`, which will display the project information.

### Import encrypted datasets
If you want to import the dataset with encryption, please give `--encryption-key <secret-key>` to the optional arguments:

```console
datum create
datum import --format datumaro_binary <path/to/dataset> -- --encryption-key <secret-key>
```

`<secret-key>` is a 50-bytes long base64 encoded string prefixed with `datum-`. It is auto-generated in `<output/dir>/secret_key.txt` when the dataset is exported to DatumaroBinary format with `--encryption` option. You must have a correct `<secret-key>` to import the dataset encrypted by Datumaro.

## Export to other formats

It can convert DatumaroBinary dataset into any other format [Datumaro supports](/docs/user-manual/supported_formats/).
To get the expected result, convert the dataset to formats
that support the specified task (e.g. for panoptic segmentation - VOC, CamVID)

There are several ways to convert a DatumaroBinary dataset to other dataset formats
using CLI:

- Export a dataset from DatumaroBinary to VOC format:

```console
datum create
datum import -f datumaro_binary <path/to/dataset>
datum export -f voc -o <output/dir>
```

or

```console
datum convert -if datumaro_binary -i <path/to/dataset> -f voc -o <output/dir>
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'datumaro_binary')
dataset.export('save_dir', 'voc', save_media=True)
```

## Export to DatumaroBinary

There are several ways to convert a dataset to DatumaroBinary format:

- Export a dataset from an existing project to DatumaroBinary:

```console
# export dataset into DatumaroBinary format from existing project
datum export -p <path/to/project> -f datumaro_binary -o <output/dir> \
    -- --save-media
```

- Convert a dataset from VOC format to DatumaroBinary:

```console
# converting to DatumaroBinary format from other format
datum convert -if voc -i <path/to/dataset> \
    -f datumaro_binary -o <output/dir> -- --save-media
```

## Export datasets with encryption

If you want to encrypt your dataset, please add ``--encryption` directive to your command:

```console
# export dataset into DatumaroBinary format from existing project
datum export -p <path/to/project> -f datumaro_binary -o <output/dir> \
    -- --save-media --encryption
```

> Note:
> Please keep your secret key file seperate from the exported dataset. The secret key file is in `<output/dir>/secret_key.txt`. You should have this secret key whenever you want to import your dataset. **We are not responsible for you permanently losing access to your datasets if you lose this secret key.**

If you want to encrypt the annotation files only, not the media files, please add `--no-media-encryption` in addition to ``--encryption` directive to your command:

```console
# export dataset into DatumaroBinary format from existing project
datum export -p <path/to/project> -f datumaro_binary -o <output/dir> \
    -- --save-media --encryption --no-media-encryption
```

Extra options for exporting to DatumaroBinary format:

- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--encryption` allow to encrypt your dataset with the auto-generated secret key.

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/unit/data_formats/datumaro/test_datumaro_format.py)
