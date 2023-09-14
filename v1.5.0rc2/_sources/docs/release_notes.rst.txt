Release Notes
#############

.. toctree::
   :maxdepth: 1

v1.4.1 (2023.07)
----------------
- Report errors for COCO (stream) and Datumaro importers

v1.4.0 (2023.07)
----------------

New features
^^^^^^^^^^^^
- Add documentation and notebook example for Prune API
- Changed supported Python version range (>=3.8, <=3.11)
- Migrate OpenVINO v2023.0.0
- Add Roboflow data format support (COCO JSON, Pascal VOC XML, YOLOv5-PyTorch, YOLOv7-PyTorch, YOLOv8, YOLOv5 Oriented Bounding Boxes, Multiclass CSV, TFRecord, CreateML JSON)
- Add MissingAnnotationDetection transform
- Add OVMSLauncher
- Add Prune API
- Add TritonLauncher
- Migrate DVC v3.0.0
- Stream dataset import/export
- Support mask annotations for CVAT data format

Enhancements
^^^^^^^^^^^^
- Support list query for explorer
- update contributing.md
- Update 3rd-party.txt for release 1.4.0
- Give notice that the deprecation works will be done in datumaro==1.5.0
- Unify COCO, Datumaro, VOC, YOLO importer/exporter progress reporter descriptions
- Enhance import performance for built-in plugins
- Change default dtype of load_image() to np.uint8
- Add OTX ATSS detector model interpreter & refactor interfaces
- Refactor Launcher and ModelInterpreter
- Add CVAT data format document
- Reduce peak memory usage when importing COCO and Datumaro formats
- Enhance the error message for datum stats to be more user friendly
- Refactor dataset.py to seperate DatasetStorage

Bug fixes
^^^^^^^^^
- Create cache dir under only writable filesystem
- Fix: Dataset infos() can be broken if a transform not redefining infos() is stacked on the top
- Fix warnings in test_visualizer.py
- Fix LabelMe data format
- Prevent installing protobuf>=4
- Fix UnionMerge

v1.3.2 (2023.06)
----------------

Enhancements
^^^^^^^^^^^^
- Let CocoBase continue even if an InvalidAnnotationError is raised

Bug fixes
^^^^^^^^^
- Install dvc version to 2.x
- Replace np.append() in Validator

v1.3.1 (2023.05)
----------------

Bug fixes
^^^^^^^^^
- Fix `Cityscapes` format mis-detection problem

v1.3.0 (2023.05)
----------------

New features
^^^^^^^^^^^^
- Add `CocoRoboflowImporter`
- Add `SynthiaSfImporter` and `SynthiaAlImporter`
- Add intermediate skill document for filter
- Add `VocInstanceSegmentationImporter` and `VocInstanceSegmentationExporter`
- Add `Segment Anything` data format support
- Add `Correct` transformation
- Add `ReindexAnnotations` transform

Enhancements
^^^^^^^^^^^^
- Use autosummary for fully-automatic Python module docs generation
- Enrich stack trace for better user experience when importing
- Save and load `hashkey` for explorer
- Add `MOT` and `MOTS` data format documents
- Improve `RemoveAnnotations` to remove specific annotations with ids
- Add Jupyter notebook example of noisy label detection for detection tasks
- Add Juypter notebook examples for importing/exporting detection and segmentation data

Bug fixes
^^^^^^^^^
- Fix `Mapillary Vistas` data format
- Fix `bytes` property returning None if function is given to data
- Fix `Synthia-Rand` data format
- Fix `person_layout` categories and `action_classification` attributes in imported Pascal-VOC dataset
- Drop a malformed transform from `StackedTransform` automatically
- Fix `Cityscapes` to drop `ImgsFine` directory

v1.2.1 (2023.05)
----------------

Bug fixes
^^^^^^^^^
- Fix project level CVAT for images format import
- Fix an info message when using the convert CLI command with no args.input_format
- Fix media contents not returning bytes in arrow format

v1.2.0 (2023.04)
----------------

New features
^^^^^^^^^^^^
- Add Skill Up section to documentation
- Add LossDynamicsAnalyzer for noisy label detection
- Add Apache Arrow format support
- Add sort transform

Enhancements
^^^^^^^^^^^^
- Add multiprocessing to DatumaroBinaryBase
- Refactor merge code
- Refactor download CLI commands
- Refactor CLI commands w/ and w/o project
- Refactor Media to be initialized from explicit sources
- Refactor hl_ops.py
- Add tfds:uc_merced and tfds:eurosat download
- Migrate documentation framework to Sphinx
- Update merge tutorial for real life usecase
- Abbreviate "detect-format" to "detect" for prettifying

Bug fixes
^^^^^^^^^
- Add UserWarning if an invalid media_type comes to image statistics computation
- Fix negated `is_encrypted`
- Save extra images of PointCloud when exporting to datumaro format
- Fix log issue when importing celeba and align celeba dataset

v1.1.0 (2023.03)
----------------

New features
^^^^^^^^^^^^
- Add with_subset_dirs decorator (Add ImagenetWithSubsetDirsImporter)
- Add CommonSemanticSegmentationWithSubsetDirsImporter
- Add DatumaroBinary format
- Add Searcher CLI documentation
- Add version to dataset exported as datumaro format
- Add Ava action data format support
- Add Shift Analyzer (both covariate and label shifts)
- Add YOLO Loose format
- Add Ultralytics YOLO format

Enhancements
^^^^^^^^^^^^
- Refactor Datumaro format code and test code

Bug fixes
^^^^^^^^^
- Fix image filenames and anomaly mask appearance in MVTec exporter
- Fix CIFAR10 and 100 detect function
- Fix celeba and align_celeba detect function
- Choose the top priority detect format for all directory depths
- Fix MVTec format detect function
- Fix wrong `__len__()` of Subset when the item is removed
- Fix mask visualization bug

v1.0.0 (2023.02)
----------------

New features
^^^^^^^^^^^^
- Add Data Explorer
- Add Ellipse annotation type
- Add MVTec anomaly data support

Enhancements
^^^^^^^^^^^^
- Refactor existing tests
- Raise ImportError on importing malformed COCO directory
- Remove the duplicated and cyclical category context in documentation

Bug fixes
^^^^^^^^^
- Fix for importing CVAT image 1.1 data format exported to project level
- Fix a problem on setting log-level via CLI
- Fix code format with the latest black==23.1.0
- Fix 'Explain command cannot find the model'
- Fix a problem found on model remove CLI command

.. note::
   About the release of the developed version can be read in the `CHANGELOG.md <https://github.com/openvinotoolkit/datumaro/blob/develop/CHANGELOG.md>`_ of the develop branch.
