# ADE20K 2017 user manual

## Contents
- [Format specification](#format-specification)
- [Load ADE20K 2017 dataset](#load-ade20k-2017-dataset)
- [Export to other formats](#export-to-other-formats)

## Format specification

- The original ADE20K 2017 dataset available
[here](https://www.kaggle.com/soumikrakshit/ade20k).

- Also the consistency set (for checking the annotation consistency)
available [here](https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2017_05_30_consistency.zip).

- ADE20K format supports the following type of annotations:
  - `Masks`

- The following attributes are supported on the masks:
  - `occluded` (boolean): whether the object is occluded by another object.
  - `part_level` (int): part level of objects on the masks.
  - other boolean user attributes can be specified
    in the annotation file `<image_name>_atr.txt`

# Load ADE20K 2017 dataset

- There are two ways to create Datumaro project and add ADE20K to it:

```bash
datum import --format ade20k2017 --input-path <path/to/dataset>
# or
datum create
datum add path -f ade20k2017 <path/to/dataset>
```

- Also it is possible to load dataset using Python API:

```python
from datumaro.components.dataset import Dataset

ade20k_dataset = Dataset.import_from('<path/to/dataset>', 'ade20k2017')
```

ADE20K dataset directory should has the following structure:

```
|-- Dataset/
    ├── subset1/
    │   |--- img1.jpg
    │   |--- img1_seg.png
    │   |--- img1_parts_1.png
    │   |--- img1_atr.txt
    │   |--- img2.jpg
    │   |--- img2_seg.png
    │   |--- ...
    │
    ├── subset2/
    │   |--- super_label_1/
    |       |--- img4.jpg
    |       |--- img4_seg.png
    |       |--- img4_atr.txt
    |       |--- img5.jpg
    |       |--- img5_seg.png
    |       |--- ...
    |   |--- img3.jpg
    |   |--- img3_seg.png
    |   |--- img3_atr.txt

```

The mask images `<image_name>_seg.png` contain information about the object
class segmentation masks and also separates each class into instances.
The channels R and G encode the objects class masks.
The channel B encodes the instance object masks.

The mask images `<image_name>_parts_N.png` contain segmentation mask for parts
of objects, where N is a number indicating the level in the part hierarchy.

The annotation files `<image_name>_atr.txt` describing the content of each
image. Each line in the text file contains:
- column 1: instance number,
- column 2: part level (0 for objects),
- column 3: occluded (1 for true),
- column 4: original raw name (might provide a more detailed categorization),
- column 5: class name (parsed using wordnet),
- column 6: double-quoted list of attributes, separated by commas.
Each column is separated by a `#`. See example of dataset
[here](../..//tests/assets/ade20k2017_dataset).
# Export to other formats

Datumaro can convert ADE20K into any other format [Datumaro supports](../user_manual.md#supported-formats).
To get the expected result, the dataset needs to be converted to a format
that supports segmentation masks.

There are a few ways to convert ADE20k 2017 to other dataset format using CLI:

```bash
datum import -f ade20k2017 -i <path/to/dataset>
datum export -f coco -o ./save_dir -- --save-images
# or
datum convert -if ade20k2017 -i <path/to/dataset> -f coco -o ./save_dir \
    --save-images
```

Or using Python API

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('<path/to/dataset>', 'ade202017')
dataset.export('save_dir', 'coco')
```
