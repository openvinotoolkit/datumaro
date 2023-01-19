# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## \[Unreleased\]
### Added
- Add Tile transformation
  (<https://github.com/openvinotoolkit/datumaro/pull/790>)
- Add Video keyframe extraction
  (<https://github.com/openvinotoolkit/datumaro/pull/791>)
- Add MergeTile transformation
  (<https://github.com/openvinotoolkit/datumaro/pull/796>)

### Changed
- Improved mask_to_rle performance
  (<https://github.com/openvinotoolkit/datumaro/pull/770>)

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- Fix MacOS CI failures
  (<https://github.com/openvinotoolkit/datumaro/pull/789>)
- Fix auto-documentation for the data_format plugins
  (<https://github.com/openvinotoolkit/datumaro/pull/793>)

### Security
- N/A

## 06/12/2022 - Release v0.4.0.1
### Added
- Support for exclusive of labels with LabelGroup
  (<https://github.com/openvinotoolkit/datumaro/pull/742>)
- Jupyter samples
  - Introducing how to merge datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/738>)
  - Introducing how to visualize dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/747>)
  - Introducing how to filter dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/748>)
  - Introducing how to transform dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/759>)
- Visualization Python API
  - Bbox feature
    (<https://github.com/openvinotoolkit/datumaro/pull/744>)
  - Label, Points, Polygon, PolyLine, and Caption visualization features
    (<https://github.com/openvinotoolkit/datumaro/pull/746>)
  - Mask, SuperResolution, Depth visualization features
    (<https://github.com/openvinotoolkit/datumaro/pull/747>)
- Documentation for Python API
  (<https://github.com/openvinotoolkit/datumaro/pull/753>)
  - dataset handler, visualizer, filter descriptions
    (<https://github.com/openvinotoolkit/datumaro/pull/761>)
- `__repr__` for Dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/750>)
- Support for exporting as CVAT video format
  (<https://github.com/openvinotoolkit/datumaro/pull/757>)
- CodeCov coverage reporting feature to CI/CD
  (<https://github.com/openvinotoolkit/datumaro/pull/756>)
- Jupyter notebook example rendering to documentation
  (<https://github.com/openvinotoolkit/datumaro/pull/758>)
- An interface to manipulate 'infos' to store the dataset meta-info
  (<https://github.com/openvinotoolkit/datumaro/pull/767>)
- 'bbox' annotation when importing a COCO dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/772>)

### Changed
- Wrap title text according to its plot width
  (<https://github.com/openvinotoolkit/datumaro/pull/769>)
- Get list of subsets and support only Image media type in visualizer
  (<https://github.com/openvinotoolkit/datumaro/pull/768>)

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- Correcting static type checking
  (<https://github.com/openvinotoolkit/datumaro/pull/743>)
- Fixing a VOC dataset export when a label contains 'space'
  (<https://github.com/openvinotoolkit/datumaro/pull/771>)

### Security
- N/A

## 06/09/2022 - Release v0.3.1
### Added
- Support for custom media types, new `PointCloud` media type,
  `DatasetItem.media` and `.media_as(type)` members
  (<https://github.com/openvinotoolkit/datumaro/pull/539>)
- \[API\] A way to request dataset and extractor media type with `media_type`
  (<https://github.com/openvinotoolkit/datumaro/pull/539>)
- BraTS format (import-only) (.npy and .nii.gz), new `MultiframeImage`
  media type (<https://github.com/openvinotoolkit/datumaro/pull/628>)
- Common Semantic Segmentation dataset format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/685>)
- An option to disable `data/` prefix inclusion in YOLO export
  (<https://github.com/openvinotoolkit/datumaro/pull/689>)
- New command `describe-downloads` to print information about downloadable datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/678>)
- Detection for Cityscapes format
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- Maximum recursion `--depth` parameter for `detect-dataset` CLI command
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- An option to save a single subset in the `download` command
  (<https://github.com/openvinotoolkit/datumaro/pull/697>)
- Common Super Resolution dataset format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/700>)
- Kinetics 400/600/700 dataset format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/706>)
- NYU Depth Dataset V2 format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/712>)

### Changed
- `env.detect_dataset()` now returns a list of detected formats at all recursion levels
  instead of just the lowest one
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- Open Images: allowed to store annotations file in root path as well
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- Improved parsing error messages in COCO, VOC and YOLO formats
  (<https://github.com/openvinotoolkit/datumaro/pull/684>,
   <https://github.com/openvinotoolkit/datumaro/pull/686>,
   <https://github.com/openvinotoolkit/datumaro/pull/687>)
- YOLO format now supports almost any subset names, except `backup`, `names` and `classes`
  (instead of just `train` and `valid`). The reserved names now raise an error on exporting.
  (<https://github.com/openvinotoolkit/datumaro/pull/688>)

### Deprecated
- `--save-images` is replaced with `--save-media` in CLI and converter API
  (<https://github.com/openvinotoolkit/datumaro/pull/539>)
- \[API\] `image`, `point_cloud` and `related_images` of `DatasetItem` are
  replaced with `media` and `media_as(type)` members and c-tor parameters
  (<https://github.com/openvinotoolkit/datumaro/pull/539>)

### Removed
- N/A

### Fixed
- Detection for LFW format
  (<https://github.com/openvinotoolkit/datumaro/pull/680>)
- Adding depth value of image when dataset is exported in VOC format
  (<https://github.com/openvinotoolkit/datumaro/pull/726>)
- Adding to handle the numerical labels in task chains properly
  (<https://github.com/openvinotoolkit/datumaro/pull/726>)
- Fixing the issue that annotations inside another annotation (polygon)
  are duplicated during import for VOC format
  (<https://github.com/openvinotoolkit/datumaro/pull/726>)

### Security
- N/A

## 21/02/2022 - Release v0.3
### Added
- Ability to import a video as frames with the `video_frames` format and
  to split a video into frames with the `datum util split_video` command
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- `--subset` parameter in the `image_dir` format
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- `MediaManager` API to control loaded media resources at runtime
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- Command to detect the format of a dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/576>)
- More comfortable access to library API via `import datumaro`
  (<https://github.com/openvinotoolkit/datumaro/pull/630>)
- CLI command-like free functions (`export`, `transform`, ...)
  (<https://github.com/openvinotoolkit/datumaro/pull/630>)
- Reading specific annotation files for train dataset in Cityscapes
  (<https://github.com/openvinotoolkit/datumaro/pull/632>)
- Random sampling transforms (`random_sampler`, `label_random_sampler`)
  to create smaller datasets from bigger ones
  (<https://github.com/openvinotoolkit/datumaro/pull/636>,
   <https://github.com/openvinotoolkit/datumaro/pull/640>)
- API to report dataset import and export progress;
  API to report dataset import and export errors and take action (skip, fail)
  (supported in COCO, VOC and YOLO formats)
  (<https://github.com/openvinotoolkit/datumaro/pull/650>)
- Support for downloading the ImageNetV2 and COCO datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/653>,
   <https://github.com/openvinotoolkit/datumaro/pull/659>)
- A way for formats to signal that they don't support detection
  (<https://github.com/openvinotoolkit/datumaro/pull/665>)
- Removal transforms to remove items/annoations/attributes from dataset
  (`remove_items`, `remove_annotations`, `remove_attributes`)
  (<https://github.com/openvinotoolkit/datumaro/pull/670>)

### Changed
- Allowed direct file paths in `datum import`. Such sources are imported like
  when the `rpath` parameter is specified, however, only the selected path
  is copied into the project
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- Improved `stats` performance, added new filtering parameters,
  image stats (`unique`, `repeated`) moved to the `dataset` section,
  removed `mean` and `std` from the `dataset` section
  (<https://github.com/openvinotoolkit/datumaro/pull/621>)
- Allowed `Image` creation from just `size` info
  (<https://github.com/openvinotoolkit/datumaro/pull/634>)
- Added image search in VOC XML-based subformats
  (<https://github.com/openvinotoolkit/datumaro/pull/634>)
- Added image path equality checks in simple merge, when applicable
  (<https://github.com/openvinotoolkit/datumaro/pull/634>)
- Supported saving box attributes when downloading the TFDS version of VOC
  (<https://github.com/openvinotoolkit/datumaro/pull/668>)
- Switched to a `pyproject.toml`-based build
  (<https://github.com/openvinotoolkit/datumaro/pull/671>)

### Deprecated
- TBD

### Removed
- Official support of Python 3.6 (due to it's EOL)
  (<https://github.com/openvinotoolkit/datumaro/pull/617>)
- Backward compatibility annotation symbols in `components.extractor`
  (<https://github.com/openvinotoolkit/datumaro/pull/630>)

### Fixed
- Prohibited calling `add`, `import` and `export` commands without a project
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- Calling `make_dataset` on empty project tree now produces the error properly
  (<https://github.com/openvinotoolkit/datumaro/pull/555>)
- Saving (overwriting) a dataset in a project when rpath is used
  (<https://github.com/openvinotoolkit/datumaro/pull/613>)
- Output image extension preserving in the `Resize` transform
  (<https://github.com/openvinotoolkit/datumaro/issues/606>)
- Memory overuse in the `Resize` transform
  (<https://github.com/openvinotoolkit/datumaro/issues/607>)
- Invalid image pixels produced by the `Resize` transform
  (<https://github.com/openvinotoolkit/datumaro/issues/618>)
- Numeric warnings that sometimes occurred in `stats` command
  (e.g. <https://github.com/openvinotoolkit/datumaro/issues/607>)
  (<https://github.com/openvinotoolkit/datumaro/pull/621>)
- Added missing item attribute merging in simple merge
  (<https://github.com/openvinotoolkit/datumaro/pull/634>)
- Inability to disambiguate VOC from LabelMe in some cases
  (<https://github.com/openvinotoolkit/datumaro/issues/658>)

### Security
- TBD

## 28/01/2022 - Release v0.2.3
### Added
- Command to download public datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/582>)
- Extension autodetection in `ByteImage`
  (<https://github.com/openvinotoolkit/datumaro/pull/595>)
- MPII Human Pose Dataset (import-only) (.mat and .json)
  (<https://github.com/openvinotoolkit/datumaro/pull/584>)
- MARS format (import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/585>)

### Changed
- The `pycocotools` dependency lower bound is raised to `2.0.4`.
  (<https://github.com/openvinotoolkit/datumaro/pull/449>)
- `smooth_line` from `datumaro.util.annotation_util` - the function
  is renamed to `approximate_line` and has updated interface
  (<https://github.com/openvinotoolkit/datumaro/pull/592>)

### Deprecated
- Python 3.6 support

### Removed
- TBD

### Fixed
- Fails in multimerge when lines are not approximated and when there are no
  label categories (<https://github.com/openvinotoolkit/datumaro/pull/592>)
- Cannot convert LabelMe dataset, that has no subsets
  (<https://github.com/openvinotoolkit/datumaro/pull/600>)

### Security
- TBD

## 24/12/2021 - Release v0.2.2
### Added
- Video reading API
  (<https://github.com/openvinotoolkit/datumaro/pull/521>)
- Python API documentation
  (<https://github.com/openvinotoolkit/datumaro/pull/526>)
- Mapillary Vistas dataset format (Import-only)
  (<https://github.com/openvinotoolkit/datumaro/pull/537>)
- Datumaro can now be installed on Windows on Python 3.9
  (<https://github.com/openvinotoolkit/datumaro/pull/547>)
- Import for SYNTHIA dataset format
  (<https://github.com/openvinotoolkit/datumaro/pull/532>)
- Support of `score` attribute in KITTI detetion
  (<https://github.com/openvinotoolkit/datumaro/pull/571>)
- Support for Accuracy Checker dataset meta files in formats
  (<https://github.com/openvinotoolkit/datumaro/pull/553>,
  <https://github.com/openvinotoolkit/datumaro/pull/569>,
  <https://github.com/openvinotoolkit/datumaro/pull/575>)
- Import for VoTT dataset format
  (<https://github.com/openvinotoolkit/datumaro/pull/573>)
- Image resizing transform
  (<https://github.com/openvinotoolkit/datumaro/pull/581>)

### Changed
- The following formats can now be detected unambiguously:
  `ade20k2017`, `ade20k2020`, `camvid`, `coco`, `cvat`, `datumaro`,
  `icdar_text_localization`, `icdar_text_segmentation`,
  `icdar_word_recognition`, `imagenet_txt`, `kitti_raw`, `label_me`, `lfw`,
  `mot_seq`, `open_images`, `vgg_face2`, `voc`, `widerface`, `yolo`
  (<https://github.com/openvinotoolkit/datumaro/pull/531>,
  <https://github.com/openvinotoolkit/datumaro/pull/536>,
  <https://github.com/openvinotoolkit/datumaro/pull/550>,
  <https://github.com/openvinotoolkit/datumaro/pull/557>,
  <https://github.com/openvinotoolkit/datumaro/pull/558>)
- Allowed Pytest-native tests
  (<https://github.com/openvinotoolkit/datumaro/pull/563>)
- Allowed export options in the `datum merge` command
  (<https://github.com/openvinotoolkit/datumaro/pull/545>)

### Deprecated
- Using `Image`, `ByteImage` from `datumaro.util.image` - these classes
  are moved to `datumaro.components.media`
  (<https://github.com/openvinotoolkit/datumaro/pull/538>)

### Removed
- Equality comparison support between `datumaro.components.media.Image`
  and `numpy.ndarray`
  (<https://github.com/openvinotoolkit/datumaro/pull/568>)

### Fixed
- Bug #560: import issue with MOT dataset when using seqinfo.ini file
  (<https://github.com/openvinotoolkit/datumaro/pull/564>)
- Empty lines in VOC subset lists are not ignored
  (<https://github.com/openvinotoolkit/datumaro/pull/587>)

### Security
- TBD

## 16/11/2021 - Release v0.2.1
### Added
- Import for CelebA dataset format.
  (<https://github.com/openvinotoolkit/datumaro/pull/484>)

### Changed
- File `people.txt` became optional in LFW
  (<https://github.com/openvinotoolkit/datumaro/pull/509>)
- File `image_ids_and_rotation.csv` became optional Open Images
  (<https://github.com/openvinotoolkit/datumaro/pull/509>)
- Allowed underscores (`_`) in subset names in COCO
  (<https://github.com/openvinotoolkit/datumaro/pull/509>)
- Allowed annotation files with arbitrary names in COCO
  (<https://github.com/openvinotoolkit/datumaro/pull/509>)
- The `icdar_text_localization` format is no longer detected in every directory
  (<https://github.com/openvinotoolkit/datumaro/pull/531>)
- Updated `pycocotools` version to 2.0.2
  (<https://github.com/openvinotoolkit/datumaro/pull/534>)

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- Unhandled exception when a file is specified as the source for a COCO or
  MOTS dataset
  (<https://github.com/openvinotoolkit/datumaro/pull/530>)
- Exporting dataset without `color` attribute into the
  `icdar_text_segmentation` format
  (<https://github.com/openvinotoolkit/datumaro/pull/556>)
### Security
- TBD

## 14/10/2021 - Release v0.2
### Added
- A new installation target: `pip install datumaro[default]`, which should
  be used by default. The simple `datumaro` is supposed for library users.
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Dataset and project versioning capabilities (Git-like)
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- "dataset revpath" concept in CLI, allowing to pass a dataset path with
  the dataset format in `diff`, `merge`, `explain` and `info` CLI commands
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `import`, `remove`, `commit`, `checkout`, `log`, `status`, `info` CLI commands
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `Coco*Extractor` classes now have an option to preserve label IDs from the
  original annotation file
  (<https://github.com/openvinotoolkit/datumaro/pull/453>)
- `patch` CLI command to patch datasets
  (<https://github.com/openvinotoolkit/datumaro/pull/401>)
- `ProjectLabels` transform to change dataset labels for merging etc.
  (<https://github.com/openvinotoolkit/datumaro/pull/401>,
   <https://github.com/openvinotoolkit/datumaro/pull/478>)
- Support for custom labels in the KITTI detection format
  (<https://github.com/openvinotoolkit/datumaro/pull/481>)
- Type annotations and docs for Annotation classes
  (<https://github.com/openvinotoolkit/datumaro/pull/493>)
- Options to control label loading behavior in `imagenet_txt` import
  (<https://github.com/openvinotoolkit/datumaro/pull/434>,
  <https://github.com/openvinotoolkit/datumaro/pull/489>)

### Changed
- A project can contain and manage multiple datasets instead of a single one.
  CLI operations can be applied to the whole project, or to separate datasets.
  Datasets are modified inplace, by default
  (<https://github.com/openvinotoolkit/datumaro/issues/328>)
- CLI help for builtin plugins doesn't require project
  (<https://github.com/openvinotoolkit/datumaro/issues/328>)
- Annotation-related classes were moved into a new module,
  `datumaro.components.annotation`
  (<https://github.com/openvinotoolkit/datumaro/pull/439>)
- Rollback utilities replaced with Scope utilities
  (<https://github.com/openvinotoolkit/datumaro/pull/444>)
- The `Project` class from `datumaro.components` is changed completely
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `diff` and `ediff` are joined into a single `diff` CLI command
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Projects use new file layout, incompatible with old projects.
  An old project can be updated with `datum project migrate`
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Inheriting `CliPlugin` is not required in plugin classes
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `Importer`s do not create `Project`s anymore and just return a list of
  extractor configurations
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)

### Deprecated
- TBD

### Removed
- `import`, `project merge` CLI commands
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Support for project hierarchies. A project cannot be a source anymore
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Project cannot have independent internal dataset anymore. All the project
  data must be stored in the project data sources
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- `datumaro_project` format
  (<https://github.com/openvinotoolkit/datumaro/pull/238>)
- Unused `path` field of `DatasetItem`
  (<https://github.com/openvinotoolkit/datumaro/pull/455>)

### Fixed
- Deprecation warning in `open_images_format.py`
  (<https://github.com/openvinotoolkit/datumaro/pull/440>)
- `lazy_image` returning unrelated data sometimes
  (<https://github.com/openvinotoolkit/datumaro/issues/409>)
- Invalid call to `pycocotools.mask.iou`
  (<https://github.com/openvinotoolkit/datumaro/pull/450>)
- Importing of Open Images datasets without image data
  (<https://github.com/openvinotoolkit/datumaro/pull/463>)
- Return value type in `Dataset.is_modified`
  (<https://github.com/openvinotoolkit/datumaro/pull/401>)
- Remapping of secondary categories in `RemapLabels`
  (<https://github.com/openvinotoolkit/datumaro/pull/401>)
- VOC dataset patching for classification and segmentation tasks
  (<https://github.com/openvinotoolkit/datumaro/pull/478>)
- Exported mask label ids in KITTI segmentation
  (<https://github.com/openvinotoolkit/datumaro/pull/481>)
- Missing `label` for `Points` read in the LFW format
  (<https://github.com/openvinotoolkit/datumaro/pull/494>)

### Security
- TBD

## 24/08/2021 - Release v0.1.11
### Added
- The Open Images format now supports bounding box
  and segmentation mask annotations
  (<https://github.com/openvinotoolkit/datumaro/pull/352>,
  <https://github.com/openvinotoolkit/datumaro/pull/388>).
- Bounding boxes values decrement transform (<https://github.com/openvinotoolkit/datumaro/pull/366>)
- Improved error reporting in `Dataset` (<https://github.com/openvinotoolkit/datumaro/pull/386>)
- Support ADE20K format (import only) (<https://github.com/openvinotoolkit/datumaro/pull/400>)
- Documentation website at <https://openvinotoolkit.github.io/datumaro> (<https://github.com/openvinotoolkit/datumaro/pull/420>)

### Changed
- Datumaro no longer depends on scikit-image
  (<https://github.com/openvinotoolkit/datumaro/pull/379>)
- `Dataset` remembers export options on saving / exporting for the first time (<https://github.com/openvinotoolkit/datumaro/pull/386>)

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- Application of `remap_labels` to dataset categories of different length (<https://github.com/openvinotoolkit/datumaro/issues/314>)
- Patching of datasets in formats (<https://github.com/openvinotoolkit/datumaro/issues/348>)
- Improved Cityscapes export performance (<https://github.com/openvinotoolkit/datumaro/pull/367>)
- Incorrect format of `*_labelIds.png` in Cityscapes export (<https://github.com/openvinotoolkit/datumaro/issues/325>, <https://github.com/openvinotoolkit/datumaro/issues/342>)
- Item id in ImageNet format (<https://github.com/openvinotoolkit/datumaro/pull/371>)
- Double quotes for ICDAR Word Recognition (<https://github.com/openvinotoolkit/datumaro/pull/375>)
- Wrong display of builtin formats in CLI (<https://github.com/openvinotoolkit/datumaro/issues/332>)
- Non utf-8 encoding of annotation files in Market-1501 export (<https://github.com/openvinotoolkit/datumaro/pull/392>)
- Import of ICDAR, PASCAL VOC and VGGFace2 images from subdirectories on WIndows
  (<https://github.com/openvinotoolkit/datumaro/pull/392>)
- Saving of images with Unicode paths on Windows (<https://github.com/openvinotoolkit/datumaro/pull/392>)
- Calling `ProjectDataset.transform()` with a string argument (<https://github.com/openvinotoolkit/datumaro/issues/402>)
- Attributes casting for CVAT format (<https://github.com/openvinotoolkit/datumaro/pull/403>)
- Loading of custom project plugins (<https://github.com/openvinotoolkit/datumaro/issues/404>)
- Reading, writing anno file and saving name of the subset for test subset
  (<https://github.com/openvinotoolkit/datumaro/pull/447>)

### Security
- Fixed unsafe unpickling in CIFAR import (<https://github.com/openvinotoolkit/datumaro/pull/362>)

## 14/07/2021 - Release v0.1.10
### Added
- Support for import/export zip archives with images (<https://github.com/openvinotoolkit/datumaro/pull/273>)
- Subformat importers for VOC and COCO (<https://github.com/openvinotoolkit/datumaro/pull/281>)
- Support for KITTI dataset segmentation and detection format (<https://github.com/openvinotoolkit/datumaro/pull/282>)
- Updated YOLO format user manual (<https://github.com/openvinotoolkit/datumaro/pull/295>)
- `ItemTransform` class, which describes item-wise dataset `Transform`s (<https://github.com/openvinotoolkit/datumaro/pull/297>)
- `keep-empty` export parameter in VOC format (<https://github.com/openvinotoolkit/datumaro/pull/297>)
- A base class for dataset validation plugins (<https://github.com/openvinotoolkit/datumaro/pull/299>)
- Partial support for the Open Images format;
  only images and image-level labels can be read/written
  (<https://github.com/openvinotoolkit/datumaro/pull/291>,
  <https://github.com/openvinotoolkit/datumaro/pull/315>).
- Support for Supervisely Point Cloud dataset format (<https://github.com/openvinotoolkit/datumaro/pull/245>, <https://github.com/openvinotoolkit/datumaro/pull/353>)
- Support for KITTI Raw / Velodyne Points dataset format (<https://github.com/openvinotoolkit/datumaro/pull/245>)
- Support for CIFAR-100 and documentation for CIFAR-10/100 (<https://github.com/openvinotoolkit/datumaro/pull/301>)

### Changed
- Tensorflow AVX check is made optional in API and disabled by default (<https://github.com/openvinotoolkit/datumaro/pull/305>)
- Extensions for images in ImageNet_txt are now mandatory (<https://github.com/openvinotoolkit/datumaro/pull/302>)
- Several dependencies now have lower bounds (<https://github.com/openvinotoolkit/datumaro/pull/308>)

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- Incorrect image layout on saving and a problem with ecoding on loading (<https://github.com/openvinotoolkit/datumaro/pull/284>)
- An error when XPath filter is applied to the dataset or its subset (<https://github.com/openvinotoolkit/datumaro/issues/259>)
- Tracking of `Dataset` changes done by transforms (<https://github.com/openvinotoolkit/datumaro/pull/297>)
- Improved CLI startup time in several cases (<https://github.com/openvinotoolkit/datumaro/pull/306>)

### Security
- Known issue: loading CIFAR can result in arbitrary code execution (<https://github.com/openvinotoolkit/datumaro/issues/327>)

## 03/06/2021 - Release v0.1.9
### Added
- Support for escaping in attribute values in LabelMe format (<https://github.com/openvinotoolkit/datumaro/issues/49>)
- Support for Segmentation Splitting (<https://github.com/openvinotoolkit/datumaro/pull/223>)
- Support for CIFAR-10/100 dataset format (<https://github.com/openvinotoolkit/datumaro/pull/225>, <https://github.com/openvinotoolkit/datumaro/pull/243>)
- Support for COCO panoptic and stuff format (<https://github.com/openvinotoolkit/datumaro/pull/210>)
- Documentation file and integration tests for Pascal VOC format (<https://github.com/openvinotoolkit/datumaro/pull/228>)
- Support for MNIST and MNIST in CSV dataset formats (<https://github.com/openvinotoolkit/datumaro/pull/234>)
- Documentation file for COCO format (<https://github.com/openvinotoolkit/datumaro/pull/241>)
- Documentation file and integration tests for YOLO format (<https://github.com/openvinotoolkit/datumaro/pull/246>)
- Support for Cityscapes dataset format (<https://github.com/openvinotoolkit/datumaro/pull/249>)
- Support for Validator configurable threshold (<https://github.com/openvinotoolkit/datumaro/pull/250>)

### Changed
- LabelMe format saves dataset items with their relative paths by subsets
  without changing names (<https://github.com/openvinotoolkit/datumaro/pull/200>)
- Allowed arbitrary subset count and names in classification and detection
  splitters (<https://github.com/openvinotoolkit/datumaro/pull/207>)
- Annotation-less dataset elements are now participate in subset splitting (<https://github.com/openvinotoolkit/datumaro/pull/211>)
- Classification task in LFW dataset format (<https://github.com/openvinotoolkit/datumaro/pull/222>)
- Testing is now performed with pytest instead of unittest (<https://github.com/openvinotoolkit/datumaro/pull/248>)

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- Added support for auto-merging (joining) of datasets with no labels and
  having labels (<https://github.com/openvinotoolkit/datumaro/pull/200>)
- Allowed explicit label removal in `remap_labels` transform (<https://github.com/openvinotoolkit/datumaro/pull/203>)
- Image extension in CVAT format export (<https://github.com/openvinotoolkit/datumaro/pull/214>)
- Added a label "face" for bounding boxes in Wider Face (<https://github.com/openvinotoolkit/datumaro/pull/215>)
- Allowed adding "difficult", "truncated", "occluded" attributes when
  converting to Pascal VOC if these attributes are not present (<https://github.com/openvinotoolkit/datumaro/pull/216>)
- Empty lines in YOLO annotations are ignored (<https://github.com/openvinotoolkit/datumaro/pull/221>)
- Export in VOC format when no image info is available (<https://github.com/openvinotoolkit/datumaro/pull/239>)
- Fixed saving attribute in WiderFace extractor (<https://github.com/openvinotoolkit/datumaro/pull/251>)

### Security
- TBD

## 31/03/2021 - Release v0.1.8
### Added
- TBD

### Changed
- Added an option to allow undeclared annotation attributes in CVAT format
  export (<https://github.com/openvinotoolkit/datumaro/pull/192>)
- COCO exports images in separate dirs by subsets. Added an option to control
  this (<https://github.com/openvinotoolkit/datumaro/pull/195>)

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- Instance masks of `background` class no more introduce an instance (<https://github.com/openvinotoolkit/datumaro/pull/188>)
- Added support for label attributes in Datumaro format (<https://github.com/openvinotoolkit/datumaro/pull/192>)

### Security
- TBD

## 24/03/2021 - Release v0.1.7
### Added
- OpenVINO plugin examples (<https://github.com/openvinotoolkit/datumaro/pull/159>)
- Dataset validation for classification and detection datasets (<https://github.com/openvinotoolkit/datumaro/pull/160>)
- Arbitrary image extensions in formats (import and export) (<https://github.com/openvinotoolkit/datumaro/issues/166>)
- Ability to set a custom subset name for an imported dataset (<https://github.com/openvinotoolkit/datumaro/issues/166>)
- CLI support for NDR(<https://github.com/openvinotoolkit/datumaro/pull/178>)

### Changed
- Common ICDAR format is split into 3 sub-formats (<https://github.com/openvinotoolkit/datumaro/pull/174>)

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- The ability to work with file names containing Cyrillic and spaces (<https://github.com/openvinotoolkit/datumaro/pull/148>)
- Image reading and saving in ICDAR formats (<https://github.com/openvinotoolkit/datumaro/pull/174>)
- Unnecessary image loading on dataset saving (<https://github.com/openvinotoolkit/datumaro/pull/176>)
- Allowed spaces in ICDAR captions (<https://github.com/openvinotoolkit/datumaro/pull/182>)
- Saving of masks in VOC when masks are not requested (<https://github.com/openvinotoolkit/datumaro/pull/184>)

### Security
- TBD

## 03/02/2021 - Release v0.1.6.1 (hotfix)
### Added
- TBD

### Changed
- TBD

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- Images with no annotations are exported again in VOC formats (<https://github.com/openvinotoolkit/datumaro/pull/123>)
- Inference result for only one output layer in OpenVINO launcher (<https://github.com/openvinotoolkit/datumaro/pull/125>)

### Security
- TBD

## 02/26/2021 - Release v0.1.6
### Added
- `Icdar13/15` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/96>)
- Laziness, source caching, tracking of changes and partial updating for `Dataset` (<https://github.com/openvinotoolkit/datumaro/pull/102>)
- `Market-1501` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/108>)
- `LFW` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/110>)
- Support of polygons' and masks' confusion matrices and mismathing classes in
  `diff` command (<https://github.com/openvinotoolkit/datumaro/pull/117>)
- Add near duplicate image removal plugin (<https://github.com/openvinotoolkit/datumaro/pull/113>)
- Sampler Plugin that analyzes inference result from the given dataset and
  selects samples for annotation(<https://github.com/openvinotoolkit/datumaro/pull/115>)

### Changed
- OpenVINO model launcher is updated for OpenVINO r2021.1 (<https://github.com/openvinotoolkit/datumaro/pull/100>)

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- High memory consumption and low performance of mask import/export, #53 (<https://github.com/openvinotoolkit/datumaro/pull/101>)
- Masks, covered by class 0 (background), should be exported with holes inside
(<https://github.com/openvinotoolkit/datumaro/pull/104>)
- `diff` command invocation problem with missing class methods (<https://github.com/openvinotoolkit/datumaro/pull/117>)

### Security
- TBD

## 01/23/2021 - Release v0.1.5
### Added
- `WiderFace` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/65>, <https://github.com/openvinotoolkit/datumaro/pull/90>)
- Function to transform annotations to labels (<https://github.com/openvinotoolkit/datumaro/pull/66>)
- Dataset splits for classification, detection and re-id tasks (<https://github.com/openvinotoolkit/datumaro/pull/68>, <https://github.com/openvinotoolkit/datumaro/pull/81>)
- `VGGFace2` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/69>, <https://github.com/openvinotoolkit/datumaro/pull/82>)
- Unique image count statistic (<https://github.com/openvinotoolkit/datumaro/pull/87>)
- Installation with pip by name `datumaro`

### Changed
- `Dataset` class extended with new operations: `save`, `load`, `export`, `import_from`, `detect`, `run_model` (<https://github.com/openvinotoolkit/datumaro/pull/71>)
- Allowed importing `Extractor`-only defined formats
  (in `Project.import_from`, `dataset.import_from` and CLI/`project import`) (<https://github.com/openvinotoolkit/datumaro/pull/71>)
- `datum project ...` commands replaced with `datum ...` commands (<https://github.com/openvinotoolkit/datumaro/pull/84>)
- Supported more image formats in `ImageNet` extractors (<https://github.com/openvinotoolkit/datumaro/pull/85>)
- Allowed adding `Importer`-defined formats as project sources (`source add`) (<https://github.com/openvinotoolkit/datumaro/pull/86>)
- Added max search depth in `ImageDir` format and importers (<https://github.com/openvinotoolkit/datumaro/pull/86>)

### Deprecated
- `datum project ...` CLI context (<https://github.com/openvinotoolkit/datumaro/pull/84>)

### Removed
- TBD

### Fixed
- Allow plugins inherited from `Extractor` (instead of only `SourceExtractor`)
  (<https://github.com/openvinotoolkit/datumaro/pull/70>)
- Windows installation with `pip` for `pycocotools` (<https://github.com/openvinotoolkit/datumaro/pull/73>)
- `YOLO` extractor path matching on Windows (<https://github.com/openvinotoolkit/datumaro/pull/73>)
- Fixed inplace file copying when saving images (<https://github.com/openvinotoolkit/datumaro/pull/76>)
- Fixed `labelmap` parameter type checking in `VOC` converter (<https://github.com/openvinotoolkit/datumaro/pull/76>)
- Fixed model copying on addition in CLI (<https://github.com/openvinotoolkit/datumaro/pull/94>)

### Security
- TBD

## 12/10/2020 - Release v0.1.4
### Added
- `CamVid` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/57>)
- Ability to install `opencv-python-headless` dependency with `DATUMARO_HEADLESS=1`
  environment variable instead of `opencv-python` (<https://github.com/openvinotoolkit/datumaro/pull/62>)

### Changed
- Allow empty supercategory in COCO (<https://github.com/openvinotoolkit/datumaro/pull/54>)
- Allow Pascal VOC to search in subdirectories (<https://github.com/openvinotoolkit/datumaro/pull/50>)

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- TBD

### Security
- TBD

## 10/28/2020 - Release v0.1.3
### Added
- `ImageNet` and `ImageNetTxt` dataset formats (<https://github.com/openvinotoolkit/datumaro/pull/41>)

### Changed
- TBD

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- Default `label-map` parameter value for VOC converter (<https://github.com/openvinotoolkit/datumaro/pull/34>)
- Randomness of random split transform (<https://github.com/openvinotoolkit/datumaro/pull/38>)
- `Transform.subsets()` method (<https://github.com/openvinotoolkit/datumaro/pull/38>)
- Supported unknown image formats in TF Detection API converter (<https://github.com/openvinotoolkit/datumaro/pull/40>)
- Supported empty attribute values in CVAT extractor (<https://github.com/openvinotoolkit/datumaro/pull/45>)

### Security
- TBD

## 10/05/2020 - Release v0.1.2
### Added
- `ByteImage` class to represent encoded images in memory and avoid recoding
  on save (<https://github.com/openvinotoolkit/datumaro/pull/27>)

### Changed
- Implementation of format plugins simplified (<https://github.com/openvinotoolkit/datumaro/pull/22>)
- `default` is now a default subset name, instead of `None`. The values are
  interchangeable. (<https://github.com/openvinotoolkit/datumaro/pull/22>)
- Improved performance of transforms (<https://github.com/openvinotoolkit/datumaro/pull/22>)

### Deprecated
- TBD

### Removed
- `image/depth` value from VOC export (<https://github.com/openvinotoolkit/datumaro/pull/27>)

### Fixed
- Zero division errors in dataset statistics (<https://github.com/openvinotoolkit/datumaro/pull/31>)

### Security
- TBD

## 09/24/2020 - Release v0.1.1
### Added
- `reindex` option in COCO and CVAT converters (<https://github.com/openvinotoolkit/datumaro/pull/18>)
- Support for relative paths in LabelMe format (<https://github.com/openvinotoolkit/datumaro/pull/19>)
- MOTS png mask format support (<https://github.com/openvinotoolkit/datumaro/21>)

### Changed
- TBD

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- TBD

### Security
- TBD

## 09/10/2020 - Release v0.1.0
### Added
- Initial release

## Template
```
## [Unreleased]
### Added
- TBD

### Changed
- TBD

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- TBD

### Security
- TBD
```
