# Cityscapes user manual

## Contents

- [Format specification](#format-specification)
- [Load Cityscapes dataset](#load-Cityscapes-dataset)
- [Export to other formats](#export-to-other-formats)
- [Export to Cityscapes](#export-to-Cityscapes)
- [Particular use cases](#particular-use-cases)

## Format specification

Cityscapes format overview available [here](https://www.cityscapes-dataset.com/dataset-overview/).
Cityscapes format specification available [here](https://github.com/mcordts/cityscapesScripts#the-cityscapes-dataset).

Cityscapes dataset format supports `Masks` (segmentations tasks) annotations.

##  Load Cityscapes dataset

The Cityscapes dataset is available for free [download](https://www.cityscapes-dataset.com/downloads/).

There are two ways to create Datumaro project and add Cityscapes dataset to it:

``` bash
datum import --format cityscapes --input-path <path/to/dataset>
# or
datum create
datum add path -f cityscapes <path/to/dataset>
```

It is possible to specify project name and project directory run
`datum create --help` for more information.

Cityscapes dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── imgsFine/
    │   ├── leftImg8bit
    │   │   ├── <split: train,val, ...>
    │   │   |   ├── {city1}
    │   │   │   |   ├── {city1}_{seq:[0...6]}_{frame:[0...6]}_leftImg8bit.png
    │   │   │   │   └── ...
    │   │   |   ├── {city2}
    │   │   │   └── ...
    │   │   └── ...
    ├── gtFine/
    │   ├── <split: train,val, ...>
    │   │   ├── {city1}
    │   │   |   ├── {city1}_{seq:[0...6]}_{frame:[0...6]}_gtFine_color.png
    │   │   |   ├── {city1}_{seq:[0...6]}_{frame:[0...6]}_gtFine_instanceIds.png
    │   │   |   ├── {city1}_{seq:[0...6]}_{frame:[0...6]}_gtFine_labelIds.png
    │   │   │   └── ...
    │   │   ├── {city2}
    │   │   └── ...
    │   └── ...
```

Annotated files description:
1. *leftImg8bit.png - left images in 8-bit LDR format
1. *color.png - class labels are encoded by its color
1. *instanceIds.png - class and instance labels are encoded by an instance ID.
    The pixel values encode class and the individual instance: the integer part
    of a division by 1000 of each ID provides class ID, the remainder
    is the instance ID. If a certain annotation describes multiple instances,
    then the pixels have the regular ID of that class
1. *labelIds.png - class labels are encoded by its ID

To make sure that the selected dataset has been added to the project, you can run
`datum info`, which will display the project and dataset information.

##  Export to other formats

Datumaro can convert Cityscapes dataset into any other format [Datumaro supports](../user_manual.md#supported-formats).
To get the expected result, the dataset needs to be converted to formats
that support the segmentation task (e.g. PascalVOC, CamVID, etc.)
There are few ways to convert Cityscapes dataset to other dataset format:

``` bash
datum project import -f cityscapes -i <path/to/cityscapes>
datum export -f voc -o <path/to/output/dir>
# or
datum convert -if cityscapes -i <path/to/cityscapes> -f voc -o <path/to/output/dir>
```

Some formats provide extra options for conversion.
These options are passed after double dash (`--`) in the command line.
To get information about them, run

`datum export -f <FORMAT> -- -h`

##  Export to Cityscapes

There are few ways to convert dataset to Cityscapes format:

``` bash
# export dataset into Cityscapes format from existing project
datum export -p <path/to/project> -f cityscapes -o <path/to/export/dir> \
    -- --save-images
# converting to Cityscapes format from other format
datum convert -if voc -i <path/to/voc/dataset> \
    -f cityscapes -o <path/to/export/dir> -- --save-images
```

Extra options for export to cityscapes format:
- `--save-images` allow to export dataset with saving images
(by default `False`);
- `--image-ext IMAGE_EXT` allow to specify image extension
for exporting dataset (by default - keep original or use `.png`, if none).
- `--apply-colormap APPLY_COLORMAP` allow to use colormap for class masks
(`*color.png` files, by default `True`);
- `--label_map` allow to define a custom colormap. Example

``` bash
# mycolormap.txt :
# 0 0 255 sky
# 255 0 0 person
#...
datum export -f cityscapes -- --label-map mycolormap.txt

# or you can use original cityscapes colomap:
datum export -f cityscapes -- --label-map cityscapes
```

## Particular use cases

Datumaro supports filtering, transformation, merging etc. for all formats
and for the Cityscapes format in particular. Follow
[user manual](../user_manual.md)
to get more information about these operations.

There are few examples of using Datumaro operations to solve
particular problems with Cityscapes dataset:

### Example 1. How to load an original Cityscapes dataset and convert to Pascal VOC

```bash
datum create -o project
datum add path -p project -f cityscapes ./Cityscapes/
datum stats -p project
datum export -p final_project -o dataset -f voc --overwrite  -- --save-images
```

### Example 2. How to create custom Cityscapes-like dataset

```python
import numpy as np
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import Mask, DatasetItem

import datumaro.plugins.cityscapes_format as Cityscapes

label_map = OrderedDict()
label_map['background'] = (0, 0, 0)
label_map['label_1'] = (1, 2, 3)
label_map['label_2'] = (3, 2, 1)
categories = Cityscapes.make_cityscapes_categories(label_map)

dataset = Dataset.from_iterable([
    DatasetItem(id=1,
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 0, 0, 1, 1]]), label=1, id=1,
                        attributes={'is_crowd': False}),
                    Mask(image=np.array([[0, 1, 1, 0, 0]]), label=2, id=2,
                        attributes={'is_crowd': False}),
                ]
            ),
    ], categories=categories)

dataset.export('./dataset', format='cityscapes')
```

More examples of working with Cityscapes dataset from code can be found in
[tests](../../tests/test_cityscapes_format.py)
