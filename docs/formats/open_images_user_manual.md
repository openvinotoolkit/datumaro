# Open Images user manual

## Contents

- [Format specification](#format-specification)
- [Load Open Images dataset](#load-open-images-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export to Open Images](#export-to-open-images)
- [Particular use cases](#particular-use-cases)

## Format specification

A description of the Open Images Dataset (OID) format is available
on [its website](https://storage.googleapis.com/openimages/web/download.html).

Datumaro currently supports only the human-verified image-level label annotations from this dataset.

## Load Open Images dataset

The Open Images dataset is available for free download.

See the [`open-images-dataset` GitHub repository](https://github.com/cvdfoundation/open-images-dataset)
for information on how to download the images.

Datumaro also requires the image description files,
which can be downloaded from the following URLs:

- [complete set](https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv)
- [train set](https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv)
- [validation set](https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv)
- [test set](https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv)

Datumaro expects at least one of the files above to be present.

In addition, the following metadata files must be present as well:

- [class descriptions](https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv)
- [class hierarchy](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json)

Annotations can be downloaded from the following URLs:

- [train image labels](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv)
- [validation image labels](https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv)
- [test image labels](https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv)

The annotations are optional.

There are two ways to create Datumaro project and add OID to it:

``` bash
datum import --format open_images --input-path <path/to/dataset>
# or
datum create
datum add path -f open_images <path/to/dataset>
```

It is possible to specify project name and project directory; run
`datum create --help` for more information.

Open Images dataset directory should have the following structure:

```
└─ Dataset/
    ├── annotations/
    │   └── bbox_labels_600_hierarchy.json
    │   └── image_ids_and_rotation.csv
    │   └── oidv6-class-descriptions.csv
    │   └── *-human-imagelabels.csv
    └── images/
        ├── test
        │   ├── <image_name1.jpg>
        │   ├── <image_name2.jpg>
        │   └── ...
        ├── train
        │   ├── <image_name1.jpg>
        │   ├── <image_name2.jpg>
        │   └── ...
        └── validation
            ├── <image_name1.jpg>
            ├── <image_name2.jpg>
            └── ...
```

To use per-subset image description files instead of `image_ids_and_rotation.csv`,
place them in the `annotations` subdirectory.

##  Export to other formats

Datumaro can convert OID into any other format [Datumaro supports](../user_manual.md#supported-formats).
To get the expected result, the dataset needs to be converted to a format
that supports image-level labels.
There are a few ways to convert OID to other dataset format:

``` bash
datum project import -f open_images -i <path/to/open_images>
datum export -f cvat -o <path/to/output/dir>
# or
datum convert -if open_images -i <path/to/open_images> -f cvat -o <path/to/output/dir>
```

Some formats provide extra options for conversion.
These options are passed after double dash (`--`) in the command line.
To get information about them, run

`datum export -f <FORMAT> -- -h`

##  Export to Open Images

Converting datasets to the Open Images format is currently not supported.

## Particular use cases

Datumaro supports filtering, transformation, merging etc. for all formats
and for the Open Images format in particular. Follow
[user manual](../user_manual.md)
to get more information about these operations.

Here is an example of using Datumaro operations to solve
a particular problem with the Open Images dataset:

### Example. How to load the Open Images dataset and convert to the format used by CVAT

```bash
datum create -o project
datum add path -p project -f open_images ./open-images-dataset/
datum stats -p project
datum export -p project -o dataset -f cvat --overwrite -- --save-images
```

More examples of working with OID from code can be found in
[tests](../../tests/test_open_images_format.py).
