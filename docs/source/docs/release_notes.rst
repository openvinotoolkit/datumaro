Release Notes
#############

.. toctree::
   :maxdepth: 1

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
