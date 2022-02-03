---
title: 'CelebA'
linkTitle: 'CelebA'
description: ''
weight: 1
---

## Format specification

The original CelebA dataset is available
[here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Supported annotation types:
- `Label`
- `Bbox`
- `Points` (landmarks)

Supported attributes:
- `5_o_Clock_Shadow`, `Arched_Eyebrows`, `Attractive`,
  `Bags_Under_Eyes`, `Bald`, `Bangs`, `Big_Lips`, `Big_Nose`, `Black_Hair`,
  `Blond_Hair`, `Blurry`, `Brown_Hair`, `Bushy_Eyebrows`, `Chubby`,
  `Double_Chin`, `Eyeglasses`, `Goatee`, `Gray_Hair`, `Heavy_Makeup`,
  `High_Cheekbones`, `Male`, `Mouth_Slightly_Open`, `Mustache`, `Narrow_Eyes`,
  `No_Beard`, `Oval_Face`, `Pale_Skin`, `Pointy_Nose`, `Receding_Hairline`,
  `Rosy_Cheeks`, `Sideburns`, `Smiling`, `Straight_Hair`, `Wavy_Hair`,
  `Wearing_Earrings`, `Wearing_Hat`, `Wearing_Lipstick`, `Wearing_Necklace`,
  `Wearing_Necktie`, `Young` (boolean)

## Import CelebA dataset

A Datumaro project with a CelebA source can be created in the following way:

```bash
datum create
datum import --format celeba <path/to/dataset>
```

It is also possible to import the dataset using Python API:

```python
from datumaro import Dataset

celeba_dataset = Dataset.import_from('<path/to/dataset>', 'celeba')
```

CelebA dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── dataset_meta.json # a list of non-format labels (optional)
├── Anno/
│   ├── identity_CelebA.txt
│   ├── list_attr_celeba.txt
│   ├── list_bbox_celeba.txt
│   └── list_landmarks_celeba.txt
├── Eval/
│   └── list_eval_partition.txt
└── Img/
    └── img_celeba/
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
```

The `identity_CelebA.txt` file contains labels (required).
The `list_attr_celeba.txt`, `list_bbox_celeba.txt`,
`list_landmarks_celeba.txt`, `list_eval_partition.txt` files contain
attributes, bounding boxes, landmarks and subsets respectively
(optional).

The original CelebA dataset stores images in a .7z archive. The archive
needs to be unpacked before importing.

To add custom classes, you can use [`dataset_meta.json`](/docs/user-manual/supported_formats/#dataset-meta-file).

## Export to other formats

Datumaro can convert a CelebA dataset into any other format [Datumaro supports](/docs/user-manual/supported_formats/).
To get the expected result, convert the dataset to a format
that supports labels, bounding boxes or landmarks.

There are several ways to convert a CelebA dataset to other dataset
formats using CLI:

```bash
datum create
datum import -f celeba <path/to/dataset>
datum export -f imagenet_txt -o ./save_dir -- --save-images
```
or
``` bash
datum convert -if celeba -i <path/to/dataset> \
    -f imagenet_txt -o <output/dir> -- --save-images
```

Or, using Python API:

```python
from datumaro import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'celeba')
dataset.export('save_dir', 'voc')
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_celeba_format.py)
