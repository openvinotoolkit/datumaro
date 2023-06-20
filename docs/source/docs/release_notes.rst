Release Notes
#############

.. toctree::
   :maxdepth: 1

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
