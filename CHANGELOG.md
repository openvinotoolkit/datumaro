# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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

## 02/26/2021 - Release v0.1.6
### Added
- `Icdar13/15` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/96>)
- Laziness, source caching, tracking of changes and partial updating for `Dataset` (<https://github.com/openvinotoolkit/datumaro/pull/102>)
- `Market-1501` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/108>)
- `LFW` dataset format (<https://github.com/openvinotoolkit/datumaro/pull/110>)
- Support of polygons' and masks' confusion matrices and mismathing classes in `diff` command (<https://github.com/openvinotoolkit/datumaro/pull/117>)
- Add near duplicate image removal plugin (<https://github.com/openvinotoolkit/datumaro/pull/113>)

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
