# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Support for escaping in attribiute values in LabelMe format (<https://github.com/openvinotoolkit/datumaro/issues/49>)

### Changed
- LabelMe format saves dataset items with their relative paths by subsets without changing names (<https://github.com/openvinotoolkit/datumaro/pull/200>)
- Allowed arbitrary subset count and names in classification and detection splitters (<https://github.com/openvinotoolkit/datumaro/pull/207>)
- Annotation-less dataset elements are now participate in subset splitting (<https://github.com/openvinotoolkit/datumaro/pull/211>)

### Deprecated
-

### Removed
-

### Fixed
- Added support for auto-merging (joining) of datasets with no labels and having labels (<https://github.com/openvinotoolkit/datumaro/pull/200>)
- Allowed explicit label removal in `remap_labels` transform (<https://github.com/openvinotoolkit/datumaro/pull/203>)
- Image extension in CVAT format export (<https://github.com/openvinotoolkit/datumaro/pull/214>)
- Empty lines in YOLO annotations are ignored (<https://github.com/openvinotoolkit/datumaro/pull/221>)

### Security
-

## 31/03/2021 - Release v0.1.8
### Added
-

### Changed
- Added an option to allow undeclared annotation attributes in CVAT format export (<https://github.com/openvinotoolkit/datumaro/pull/192>)
- COCO exports images in separate dirs by subsets. Added an option to control this (<https://github.com/openvinotoolkit/datumaro/pull/195>)

### Deprecated
-

### Removed
-

### Fixed
- Instance masks of `background` class no more introduce an instance (<https://github.com/openvinotoolkit/datumaro/pull/188>)
- Added support for label attributes in Datumaro format (<https://github.com/openvinotoolkit/datumaro/pull/192>)

### Security
-

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
-

### Removed
-

### Fixed
- The ability to work with file names containing Cyrillic and spaces (<https://github.com/openvinotoolkit/datumaro/pull/148>)
- Image reading and saving in ICDAR formats (<https://github.com/openvinotoolkit/datumaro/pull/174>)
- Unnecessary image loading on dataset saving (<https://github.com/openvinotoolkit/datumaro/pull/176>)
- Allowed spaces in ICDAR captions (<https://github.com/openvinotoolkit/datumaro/pull/182>)
- Saving of masks in VOC when masks are not requested (<https://github.com/openvinotoolkit/datumaro/pull/184>)

### Security
-

## 03/02/2021 - Release v0.1.6.1 (hotfix)
### Added
-

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
- Images with no annotations are exported again in VOC formats (<https://github.com/openvinotoolkit/datumaro/pull/123>)
- Inference result for only one output layer in OpenVINO launcher (<https://github.com/openvinotoolkit/datumaro/pull/125>)

### Security
-

## 02/26/2021 - Release v0.1.6
### Added
- `Icdar13/15` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/96>)
- Laziness, source caching, tracking of changes and partial updating for `Dataset` (<https://github.com/openvinotoolkit/datumaro/pull/102>)
- `Market-1501` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/108>)
- `LFW` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/110>)
- Support of polygons' and masks' confusion matrices and mismathing classes in `diff` command (<https://github.com/openvinotoolkit/datumaro/pull/117>)
- Add near duplicate image removal plugin (<https://github.com/openvinotoolkit/datumaro/pull/113>)
- Sampler Plugin that analyzes inference result from the given dataset and selects samples for annotation(<https://github.com/openvinotoolkit/datumaro/pull/115>)

### Changed
- OpenVINO model launcher is updated for OpenVINO r2021.1 (<https://github.com/openvinotoolkit/datumaro/pull/100>)

### Deprecated
-

### Removed
-

### Fixed
- High memory consumption and low performance of mask import/export, #53 (<https://github.com/openvinotoolkit/datumaro/pull/101>)
- Masks, covered by class 0 (background), should be exported with holes inside (<https://github.com/openvinotoolkit/datumaro/pull/104>)
- `diff` command invocation problem with missing class methods (<https://github.com/openvinotoolkit/datumaro/pull/117>)

### Security
-

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
- Allowed importing `Extractor`-only defined formats (in `Project.import_from`, `dataset.import_from` and CLI/`project import`) (<https://github.com/openvinotoolkit/datumaro/pull/71>)
- `datum project ...` commands replaced with `datum ...` commands (<https://github.com/openvinotoolkit/datumaro/pull/84>)
- Supported more image formats in `ImageNet` extractors (<https://github.com/openvinotoolkit/datumaro/pull/85>)
- Allowed adding `Importer`-defined formats as project sources (`source add`) (<https://github.com/openvinotoolkit/datumaro/pull/86>)
- Added max search depth in `ImageDir` format and importers (<https://github.com/openvinotoolkit/datumaro/pull/86>)

### Deprecated
- `datum project ...` CLI context (<https://github.com/openvinotoolkit/datumaro/pull/84>)

### Removed
-

### Fixed
- Allow plugins inherited from `Extractor` (instead of only `SourceExtractor`) (<https://github.com/openvinotoolkit/datumaro/pull/70>)
- Windows installation with `pip` for `pycocotools` (<https://github.com/openvinotoolkit/datumaro/pull/73>)
- `YOLO` extractor path matching on Windows (<https://github.com/openvinotoolkit/datumaro/pull/73>)
- Fixed inplace file copying when saving images (<https://github.com/openvinotoolkit/datumaro/pull/76>)
- Fixed `labelmap` parameter type checking in `VOC` converter (<https://github.com/openvinotoolkit/datumaro/pull/76>)
- Fixed model copying on addition in CLI (<https://github.com/openvinotoolkit/datumaro/pull/94>)

### Security
-

## 12/10/2020 - Release v0.1.4
### Added
- `CamVid` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/57>)
- Ability to install `opencv-python-headless` dependency with `DATUMARO_HEADLESS=1`
  enviroment variable instead of `opencv-python` (<https://github.com/openvinotoolkit/datumaro/pull/62>)

### Changed
- Allow empty supercategory in COCO (<https://github.com/openvinotoolkit/datumaro/pull/54>)
- Allow Pascal VOC to search in subdirectories (<https://github.com/openvinotoolkit/datumaro/pull/50>)

### Deprecated
-

### Removed
-

### Fixed
-

### Security
-

## 10/28/2020 - Release v0.1.3
### Added
- `ImageNet` and `ImageNetTxt` dataset formats (<https://github.com/openvinotoolkit/datumaro/pull/41>)

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
- Default `label-map` parameter value for VOC converter (<https://github.com/openvinotoolkit/datumaro/pull/34>)
- Randomness of random split transform (<https://github.com/openvinotoolkit/datumaro/pull/38>)
- `Transform.subsets()` method (<https://github.com/openvinotoolkit/datumaro/pull/38>)
- Supported unknown image formats in TF Detection API converter (<https://github.com/openvinotoolkit/datumaro/pull/40>)
- Supported empty attribute values in CVAT extractor (<https://github.com/openvinotoolkit/datumaro/pull/45>)

### Security
-


## 10/05/2020 - Release v0.1.2
### Added
- `ByteImage` class to represent encoded images in memory and avoid recoding on save (<https://github.com/openvinotoolkit/datumaro/pull/27>)

### Changed
- Implementation of format plugins simplified (<https://github.com/openvinotoolkit/datumaro/pull/22>)
- `default` is now a default subset name, instead of `None`. The values are interchangeable. (<https://github.com/openvinotoolkit/datumaro/pull/22>)
- Improved performance of transforms (<https://github.com/openvinotoolkit/datumaro/pull/22>)

### Deprecated
-

### Removed
- `image/depth` value from VOC export (<https://github.com/openvinotoolkit/datumaro/pull/27>)

### Fixed
- Zero division errors in dataset statistics (<https://github.com/openvinotoolkit/datumaro/pull/31>)

### Security
-


## 09/24/2020 - Release v0.1.1
### Added
- `reindex` option in COCO and CVAT converters (<https://github.com/openvinotoolkit/datumaro/pull/18>)
- Support for relative paths in LabelMe format (<https://github.com/openvinotoolkit/datumaro/pull/19>)
- MOTS png mask format support (<https://github.com/openvinotoolkit/datumaro/21>)

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
-

### Security
-


## 09/10/2020 - Release v0.1.0
### Added
- Initial release

## Template
```
## [Unreleased]
### Added
-

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
-

### Security
-
```
