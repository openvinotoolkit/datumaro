---
title: 'Image zip'
linkTitle: 'Image zip'
description: ''
weight: 4
---

## Format specification

The image zip format allows to export/import unannotated datasets
with images to/from a zip archive. The format doesn't support any
annotations or attributes.

## Import Image zip dataset

There are several ways to import unannotated datasets to your Datumaro project:

- From an existing archive:

```bash
datum create
datum add -f image_zip ./images.zip
```

- From a directory with zip archives. Datumaro will import images from
  all zip files in the directory:

```bash
datum create
datum add -f image_zip ./foo
```

The directory with zip archives should have the following structure:

```
└── foo/
    ├── archive1.zip/
    |   ├── image_1.jpg
    |   ├── image_2.png
    |   ├── subdir/
    |   |   ├── image_3.jpg
    |   |   └── ...
    |   └── ...
    ├── archive2.zip/
    |   ├── image_101.jpg
    |   ├── image_102.jpg
    |   └── ...
    ...
```

Images in the archives should have a supported extension,
follow the [user manual](/docs/user-manual/media_formats/) to see the supported
extensions.

## Export to other formats

Datumaro can load dataset images from a zip archive and convert it to an
[another supported dataset format](/docs/user-manual/supported_formats),
for example:

```bash
datum create -o project
datum add -p project -f image_zip ./images.zip
datum export -p project -f coco -o ./new_dir -- --save-images
```

## Export an unannotated dataset to a zip archive

Example: exporting images from a VOC dataset to zip archives:
```bash
datum create -o project
datum add -p project -f voc ./VOC2012
datum export -p project -f image_zip -- --name voc_images.zip
```

Extra options for exporting to image_zip format:

- `--save-images` allow to export dataset with saving images
  (default: `False`);
- `--image-ext <IMAGE_EXT>` allow to specify image extension
  for exporting dataset (default: use original or `.jpg`, if none);
- `--name` name of output zipfile (default: `default.zip`);
- `--compression` allow to specify archive compression method.
  Available methods:
  `ZIP_STORED`, `ZIP_DEFLATED`, `ZIP_BZIP2`, `ZIP_LZMA` (default: `ZIP_STORED`).
  Follow [zip documentation](https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT)
  for more information.


## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/test_image_zip_format.py)
