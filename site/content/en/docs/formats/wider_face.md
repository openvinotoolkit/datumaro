---
title: WIDER Face
linkTitle: WIDER Face
description: ''
weight: 19
---

## Format specification

WIDER Face dataset is a face detection benchmark dataset,
that available for download [here](http://shuoyang1213.me/WIDERFACE/#Download).

Supported types of annotation:
- `Bbox`
- `Label`

Supported attributes for bboxes:
- `blur`:
    - 0 face without blur;
    - 1 face with normal blur;
    - 2 face with heavy blur.
- `expression`:
    - 0 face with typical expression;
    - 1 face with exaggerate expression.
- `illumination`:
    - 0 image contains normal illumination;
    - 1 image contains extreme illumination.
- `pose`:
    - 0 pose is typical;
    - 1 pose is atypical.
- `invalid`:
    - 0 image is valid;
    - 1 image is invalid.
- `occluded`:
    - 0 face without occlusion;
    - 1 face with partial occlusion;
    - 2 face with heavy occlusion.


## Import WIDER Face dataset

Importing of WIDER Face dataset into the Datumaro project:
```
datum create
datum import -f wider_face <path_to_wider_face>
```

Directory with WIDER Face dataset should has the following structure:
```
<path_to_wider_face>
├── labels.txt  # optional file with list of classes
├── wider_face_split # directory with description of bboxes for each image
│   ├── wider_face_subset1_bbx_gt.txt
│   ├── wider_face_subset2_bbx_gt.txt
│   ├── ...
├── WIDER_subset1 # instead of 'subset1' you can use any other subset name
│   └── images
│       ├── 0--label_0 # instead of 'label_<n>' you can use any other class name
│       │   ├──  0_label_0_image_01.jpg
│       │   ├──  0_label_0_image_02.jpg
│       │   ├──  ...
│       ├── 1--label_1
│       │   ├──  1_label_1_image_01.jpg
│       │   ├──  1_label_1_image_02.jpg
│       │   ├──  ...
│       ├── ...
├── WIDER_subset2
│  └── images
│      ├── ...
├── ...
```
Check [README](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip)
file of the original WIDER Face dataset to get more information
about structure of `.txt` annotation files.
Also example of WIDER Face dataset available in our
[test assets](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/widerface_dataset).

## Export WIDER Face dataset

With Datumaro you can convert WIDER Face dataset into any other
format [Datumaro supports](/docs/user-manual/supported_formats/).
Pay attention that this format should also support `Label` and/or `Bbox`
annotation types.

Few ways to export WIDER Face dataset using CLI:
```
# Using `convert` command
datum convert -if wider_face -i <path_to_wider_face> \
    -f voc -o <output_dir> -- --save-images

# Through the Datumaro project
datum create
datum import -f wider_face <path_to_wider_face>
datum export -f voc -o <output_dir> -- -save-images
```

Export WIDER Face dataset using Python API:
```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('<path_to_wider_face', 'wider_face')

# Here you can perform some transformation using dataset.transform or
# dataset.filter

dataset.export('output_dir', 'open_images', save_images=True)
```

> Note: some formats have extra export options. For particular format see the
> [docs](/docs/formats/) to get information about it.

## Export to WIDER Face dataset

Using Datumaro you can convert your dataset into the WIDER Face format,
but for succseful exporting your dataset should contain `Label` and/or `Bbox`.

Here example of exporting VOC dataset (object detection task)
into the WIDER Face format:

```
datum create
datum import -f voc_detection <path_to_voc>
datum export -f wider_face -o <output_dir> -- --save-images --image-ext='.png'
```

Available extra export options for WIDER Face dataset format:
- `--save-images` allow to export dataset with saving images.
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original)
