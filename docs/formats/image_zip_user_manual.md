# Image zip user manual

## Contents
- [Format specification](#format-specification)
- [Load image zip dataset](#load-image-zip-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export unannotated dataset to zip archive](#export-unannotated-dataset-to-zip-archive)

## Format specification

- The image zip format allow to export/import unannotated datasets
with images to/from zip archive.

- The image zip format doesn't support any types of annotations
and attributes.

## Load Image zip dataset

Few ways to load unannotated datasets to your Datumaro project:

- From existing archive:

```bash
datum import -o project -f image_zip -i ./images.zip
```

- From directory with zip archives. Datumaro will loaded images from
all zip files in the directory:

```bash
# the './foo' directory should contain zip archives with images
datum import -o project -f image_zip -i ./foo
```

Images in a archives should have supported extension,
follow the [user manual](../user_manual.md#data-formats) to see the supported extensions.

## Export to other formats

Datumaro can load an images dataset from zip archive and convert it to
[another supported dataset format](../user_manual.md#supported-formats),
for example:

```bash
datum import -o project -f image_zip -i ./images.zip
datum export -f coco -o ./new_dir -- --save-images
```

## Export unannotated dataset to zip archive

Example: exporting images from VOC dataset to zip archives:
```bash
datum import -o project -f voc -i ./VOC2012
datum export -f image_zip -o ./ --overwrite -- --name voc_images.zip \
    --compression ZIP_DEFLATED
```

Extra options for export to image_zip format:

- `--save-images` allow to export dataset with saving images
(default: `False`);
- `--image-ext <IMAGE_EXT>` allow to specify image extension
for exporting dataset (default: use original or `.jpg`, if none);
- `--name` name of output zipfile (default: `default.zip`);
- `--compression` allow to specify archive compression method. Available methods:
`ZIP_STORED`, `ZIP_DEFLATED`, `ZIP_BZIP2`, `ZIP_LZMA` (default: `ZIP_STORED`).
Follow [zip documentation](https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT)
for more information.