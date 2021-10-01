---
title: 'CelebA'
linkTitle: 'CelebA'
description: ''
weight: 1
---

## Format specification

The original CelebA dataset is available
[here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg).

Supported annotation types:
- `Label`
- `Bbox`
- `Points` (landmarks)

Supported attributes: `5_o_Clock_Shadow`, `Arched_Eyebrows`, `Attractive`,
`Bags_Under_Eyes`, `Bald`, `Bangs`, `Big_Lips`, `Big_Nose`, `Black_Hair`,
`Blond_Hair`, `Blurry Brown_Hair`, `Bushy_Eyebrows`, `Chubby`, `Double_Chin`,
`Eyeglasses`, `Goatee`, `Gray_Hair`, `Heavy_Makeup`, `High_Cheekbones`,
`Male`, `Mouth_Slightly_Open`, `Mustache Narrow_Eyes`, `No_Beard`, `Oval_Face`,
`Pale_Skin`, `Pointy_Nose`, `Receding_Hairline`, `Rosy_Cheeks`, `Sideburns Smiling`,
`Straight_Hair`, `Wavy_Hair`, `Wearing_Earrings`, `Wearing_Hat`, `Wearing_Lipstick`,
`Wearing_Necklace`, `Wearing_Necktie`, `Young`.
Attributes take values: `1` represents positive, `-1` represents negative.

## Import CelebA dataset

A Datumaro project with an CelebA source can be created in the following way:

```bash
datum create
datum import --format celeba <path/to/dataset>
```

It is also possible to import the dataset using Python API:

```python
from datumaro.components.dataset import Dataset

celeba_dataset = Dataset.import_from('<path/to/dataset>', 'celeba')
```

CelebA dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
dataset/
├── Anno
│   ├── identity_CelebA.txt
│   ├── list_attr_celeba.txt
│   ├── list_bbox_celeba.txt
│   └── list_landmarks_celeba.txt
├── Eval
│   └── list_eval_partition.txt
└── Img
    └── img_celeba
        ├── 000001.jpg
        ├── 000002.jpg
        ├── 000003.jpg
        ├── 000004.jpg
        ├── 000005.jpg
        └── ...
```

`identity_CelebA.txt` file contains labels.
`list_attr_celeba.txt` file contains attributes.
`list_bbox_celeba.txt` file contains bounding boxes.
`list_landmarks_celeba.txt` file contains landmarks.
`list_eval_partition.txt` file contains subsets.

The original CelebA dataset stores images in .7z archive. The archive
needs to be unpacked before importing.

## Export to other formats

Datumaro can convert an CelebA dataset into any other format [Datumaro supports](/docs/user-manual/supported_formats/).
To get the expected result, convert the dataset to a format
that supports labels, bounding boxes or landmarks.

There are several ways to convert an CelebA dataset to other dataset
formats using CLI:

```bash
datum create
datum import -f celeba <path/to/dataset>
datum export -f imagenet_txt -o ./save_dir -- --save-images
# or
datum convert -if celeba -i <path/to/dataset> \
    -f imagenet_txt -o <output/dir> -- --save-images
```

Or, using Python API:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'celeba')
dataset.export('save_dir', 'voc')
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_celeba_format.py)
