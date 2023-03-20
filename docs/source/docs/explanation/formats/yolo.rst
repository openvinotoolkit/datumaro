YOLO
====

Format specification
--------------------

The YOLO dataset format is for training and validating object detection
models. Specification for this format is available
`here <https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects>`_.

You can also find official examples of working with YOLO dataset `here <https://pjreddie.com/darknet/yolo/>`_.

Supported annotation types:

- ``Bounding boxes``

YOLO format doesn't support attributes for annotations.

The format supports arbitrary subset names, except ``classes``, ``names`` and ``backup``.

.. note::

    By default, the YOLO framework does not expect any subset names,
    except ``train`` and ``valid``, Datumaro supports this as an extension.
    If there is no subset separation in a project, the data
    will be saved in the ``train`` subset.

Import YOLO dataset
-------------------

A Datumaro project with a YOLO source can be created in the following way:

.. code-block::

    datum create
    datum import --format yolo <path/to/dataset>

YOLO dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ yolo_dataset/
   │
   ├── dataset_meta.json # a list of non-format labels (optional)
   ├── obj.names  # file with list of classes
   ├── obj.data   # file with dataset information
   ├── train.txt  # list of image paths in train subset
   ├── valid.txt  # list of image paths in valid subset
   │
   ├── obj_train_data/  # directory with annotations and images for train subset
   │    ├── image1.txt  # list of labeled bounding boxes for image1
   │    ├── image1.jpg
   │    ├── image2.txt
   │    ├── image2.jpg
   │    └── ...
   │
   └── obj_valid_data/  # directory with annotations and images for valid subset
        ├── image101.txt
        ├── image101.jpg
        ├── image102.txt
        ├── image102.jpg
        └── ...
```

- ``obj.data`` should have the following content, it is not necessary to have both
  subsets, but necessary to have one of them:

``` txt
classes = 5 # optional
names = <path/to/obj.names>
train = <path/to/train.txt>
valid = <path/to/valid.txt>
backup = backup/ # optional
```
- ``obj.names`` contains a list of classes.
The line number for the class is the same as its index:
``` txt
label1  # label1 has index 0
label2  # label2 has index 1
label3  # label2 has index 2
...
```
- Files ``train.txt`` and ``valid.txt`` should have the following structure:
``` txt
<path/to/image1.jpg>
<path/to/image2.jpg>
...
```
- Files in directories ``obj_train_data/`` and ``obj_valid_data/``
should contain information about labeled bounding boxes
for images:
``` txt
# image1.txt:
# <label_index> <x_center> <y_center> <width> <height>
0 0.250000 0.400000 0.300000 0.400000
3 0.600000 0.400000 0.400000 0.266667
```
Here ``x_center``, ``y_center``, ``width``, and ``height`` are relative to the image's
width and height. The ``x_center`` and ``y_center`` are center of rectangle
(are not top-left corner).

To add custom classes, you can use [`dataset_meta.json`](/docs/user-manual/supported_formats/#dataset-meta-file).

Export to other formats
-----------------------

Datumaro can convert YOLO dataset into any other format
[Datumaro supports](/docs/user-manual/supported_formats/).
For successful conversion the output format should support
object detection task (e.g. Pascal VOC, COCO, TF Detection API etc.)

There are several ways to convert a YOLO dataset to other dataset formats:

.. code-block::

    datum create
    datum add -f yolo <path/to/yolo/>
    datum export -f voc -o <output/dir>

or

.. code-block::

    datum convert -if yolo -i <path/to/dataset> \
                  -f coco_instances -o <path/to/dataset>

Or, using Python API:

.. code-block::

    import datumaro as dm

    dataset = dm.Dataset.import_from('<path/to/dataset>', 'yolo')
    dataset.export('save_dir', 'coco_instances', save_media=True)

Export to YOLO format
---------------------

Datumaro can convert an existing dataset to YOLO format,
if the dataset supports object detection task.

Example:

.. code-block::

    datum create
    datum import -f coco_instances <path/to/dataset>
    datum export -f yolo -o <path/to/dataset> -- --save-media

Extra options for exporting to YOLO format:

- ``--save-media`` allow to export dataset with saving media files
  (default: ``False``)
- ``--image-ext <IMAGE_EXT>`` allow to specify image extension
  for exporting dataset (default: use original or ``.jpg``, if none)
- ``--add-path-prefix`` allows to specify, whether to include the
  ``data/`` path prefix in the annotation files or not (default: ``True``)

Examples
--------

Example 1. Prepare PASCAL VOC dataset for exporting to YOLO format dataset
--------------------------------------------------------------------------

.. code-block::

    datum create -o project
    datum import -p project -f voc ./VOC2012
    datum filter -p project -e '/item[subset="train" or subset="val"]'
    datum transform -p project -t map_subsets -- -s train:train -s val:valid
    datum export -p project -f yolo -- --save-media

Example 2. Remove a class from YOLO dataset
-------------------------------------------

Delete all items, which contain ``cat`` objects and remove
``cat`` from list of classes:

.. code-block::

    datum create -o project
    datum import -p project -f yolo ./yolo_dataset
    datum filter -p project -m i+a -e '/item/annotation[label!="cat"]'
    datum transform -p project -t remap_labels -- -l cat:
    datum export -p project -f yolo -o ./yolo_without_cats

Example 3. Create a custom dataset in YOLO format
-------------------------------------------------

.. code-block::

    import numpy as np
    import datumaro as dm

    dataset = dm.Dataset.from_iterable([
        dm.DatasetItem(id='image_001', subset='train',
            image=np.ones((20, 20, 3)),
            annotations=[
                dm.Bbox(3.0, 1.0, 8.0, 5.0, label=1),
                dm.Bbox(1.0, 1.0, 10.0, 1.0, label=2)
            ]
        ),
        dm.DatasetItem(id='image_002', subset='train',
            image=np.ones((15, 10, 3)),
            annotations=[
                dm.Bbox(4.0, 4.0, 4.0, 4.0, label=3)
            ]
        )
    ], categories=['house', 'bridge', 'crosswalk', 'traffic_light'])

    dataset.export('../yolo_dataset', format='yolo', save_media=True)

Example 4. Get information about objects on each image
------------------------------------------------------

If you only want information about label names for each
image, then you can get it from code:

.. code-block::

    import datumaro as dm

    dataset = dm.Dataset.import_from('./yolo_dataset', format='yolo')
    cats = dataset.categories()[dm.AnnotationType.label]

    for item in dataset:
        for ann in item.annotations:
            print(item.id, cats[ann.label].name)

And If you want complete information about each item you can run:

.. code-block::

    datum create -o project
    datum import -p project -f yolo ./yolo_dataset
    datum filter -p project --dry-run -e '/item'
