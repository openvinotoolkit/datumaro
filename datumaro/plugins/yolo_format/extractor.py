# Copyright (C) 2019-2022 Intel Corporation
# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import os
import os.path as osp
import re
from collections import OrderedDict
from functools import cached_property
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import yaml

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    LabelCategories,
    Points,
    PointsCategories,
    Polygon,
    Skeleton,
)
from datumaro.components.errors import (
    DatasetImportError,
    InvalidAnnotationError,
    UndeclaredLabelError,
)
from datumaro.components.extractor import DatasetItem, Extractor, SourceExtractor
from datumaro.components.media import Image
from datumaro.util.image import (
    DEFAULT_IMAGE_META_FILE_NAME,
    ImageMeta,
    load_image,
    load_image_meta_file,
)
from datumaro.util.meta_file_util import get_meta_file, has_meta_file, parse_meta_file
from datumaro.util.os_util import split_path

from ...util import parse_json_file, take_by
from .format import YoloPath, YOLOv8Path, YOLOv8PoseFormat

T = TypeVar("T")


class YoloExtractor(SourceExtractor):
    RESERVED_CONFIG_KEYS = YoloPath.RESERVED_CONFIG_KEYS

    class Subset(Extractor):
        def __init__(self, name: str, parent: YoloExtractor):
            super().__init__()
            self._name = name
            self._parent = parent
            self.items: Dict[str, Union[str, DatasetItem]] = OrderedDict()

        def __iter__(self):
            for item_id in self.items:
                item = self._parent._get(item_id, self._name)
                if item is not None:
                    yield item

        def __len__(self):
            return len(self.items)

        def categories(self):
            return self._parent.categories()

    def __init__(
        self,
        config_path: str,
        image_info: Union[None, str, ImageMeta] = None,
        **kwargs,
    ) -> None:
        if not osp.isfile(config_path):
            raise DatasetImportError(f"Can't read dataset descriptor file '{config_path}'")

        super().__init__(**kwargs)

        rootpath = osp.dirname(config_path)
        self._config_path = config_path
        self._path = rootpath

        assert image_info is None or isinstance(image_info, (str, dict))
        if image_info is None:
            image_info = osp.join(rootpath, DEFAULT_IMAGE_META_FILE_NAME)
            if not osp.isfile(image_info):
                image_info = {}
        if isinstance(image_info, str):
            image_info = load_image_meta_file(image_info)

        self._image_info = image_info

        self._categories = self._load_categories()

        # The original format is like this:
        #
        # classes = 2
        # train  = data/train.txt
        # valid  = data/test.txt
        # names = data/obj.names
        # backup = backup/
        #
        # To support more subset names, we disallow subsets
        # called 'classes', 'names' and 'backup'.
        subsets = {k: v for k, v in self._config.items() if k not in self.RESERVED_CONFIG_KEYS}

        for subset_name, list_path in subsets.items():
            subset = YoloExtractor.Subset(subset_name, self)
            subset.items = OrderedDict(
                (self.name_from_path(p), self.localize_path(p))
                for p in self._iterate_over_image_paths(subset_name, list_path)
            )
            subsets[subset_name] = subset

        self._subsets: Dict[str, YoloExtractor.Subset] = subsets

    def _iterate_over_image_paths(self, subset_name: str, list_path: str):
        list_path = osp.join(self._path, self.localize_path(list_path))
        if not osp.isfile(list_path):
            raise InvalidAnnotationError(f"Can't find '{subset_name}' subset list file")

        with open(list_path, "r", encoding="utf-8") as f:
            yield from (p for p in f if p.strip())

    @cached_property
    def _config(self) -> Dict[str, str]:
        with open(self._config_path, "r", encoding="utf-8") as f:
            config_lines = f.readlines()

        config = {}

        for line in config_lines:
            match = re.match(r"^\s*(\w+)\s*=\s*(.+)$", line)
            if not match:
                continue

            key = match.group(1)
            value = match.group(2)
            config[key] = value

        return config

    @staticmethod
    def localize_path(path: str) -> str:
        """
        Removes the "data/" prefix from the path
        """

        path = osp.normpath(path.strip()).replace("\\", "/")
        default_base = "data/"
        if path.startswith(default_base):
            path = path[len(default_base) :]
        return path

    @classmethod
    def name_from_path(cls, path: str) -> str:
        """
        Obtains <image name> from the path like [data/]<subset>_obj/<image_name>.ext

        <image name> can be <a/b/c/filename>, so it is
        more involved than just calling "basename()".
        """

        path = cls.localize_path(path)

        parts = split_path(path)
        if 1 < len(parts) and not osp.isabs(path):
            path = osp.join(*parts[1:])  # pylint: disable=no-value-for-parameter

        return osp.splitext(path)[0]

    @classmethod
    def _image_loader(cls, *args, **kwargs):
        return load_image(*args, **kwargs, keep_exif=True)

    def _get_labels_path_from_image_path(self, image_path: str) -> str:
        return osp.splitext(image_path)[0] + YoloPath.LABELS_EXT

    def _get(self, item_id: str, subset_name: str) -> Optional[DatasetItem]:
        subset = self._subsets[subset_name]
        item = subset.items[item_id]

        if isinstance(item, str):
            try:
                image_size = self._image_info.get(item_id)
                image_path = osp.join(self._path, item)

                if image_size:
                    image = Image(path=image_path, size=image_size)
                else:
                    image = Image(path=image_path, data=self._image_loader)

                anno_path = self._get_labels_path_from_image_path(image.path)
                annotations = self._parse_annotations(
                    anno_path, image, item_id=(item_id, subset_name)
                )

                item = DatasetItem(
                    id=item_id, subset=subset_name, media=image, annotations=annotations
                )
                subset.items[item_id] = item
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(item_id, subset_name))
                subset.items.pop(item_id)
                item = None

        return item

    @staticmethod
    def _parse_field(value: str, cls: Type[T], field_name: str) -> T:
        try:
            return cls(value)
        except Exception as e:
            raise InvalidAnnotationError(
                f"Can't parse {field_name} from '{value}'. Expected {cls}"
            ) from e

    def _parse_annotations(
        self, anno_path: str, image: Image, *, item_id: Tuple[str, str]
    ) -> List[Annotation]:
        lines = []
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        annotations = []

        if lines:
            # Use image info as late as possible to avoid unnecessary image loading
            if image.size is None:
                raise DatasetImportError(
                    f"Can't find image info for '{self.localize_path(image.path)}'"
                )
            image_height, image_width = image.size

        for line in lines:
            try:
                annotations.append(
                    self._load_one_annotation(line.split(), image_height, image_width)
                )
            except Exception as e:
                self._ctx.error_policy.report_annotation_error(e, item_id=item_id)

        return annotations

    def _load_one_annotation(
        self, parts: List[str], image_height: int, image_width: int
    ) -> Annotation:
        if len(parts) != 5:
            raise InvalidAnnotationError(
                f"Unexpected field count {len(parts)} in the bbox description. "
                "Expected 5 fields (label, xc, yc, w, h)."
            )
        label_id, xc, yc, w, h = parts

        label_id = self._parse_field(label_id, int, "bbox label id")
        if label_id not in self._categories[AnnotationType.label]:
            raise UndeclaredLabelError(str(label_id))

        w = self._parse_field(w, float, "bbox width")
        h = self._parse_field(h, float, "bbox height")
        x = self._parse_field(xc, float, "bbox center x") - w * 0.5
        y = self._parse_field(yc, float, "bbox center y") - h * 0.5

        return Bbox(
            x * image_width,
            y * image_height,
            w * image_width,
            h * image_height,
            label=label_id,
        )

    def _load_categories(self) -> Dict[AnnotationType, LabelCategories]:
        names_path = self._config.get("names")
        if not names_path:
            raise InvalidAnnotationError(f"Failed to parse names file path from config")

        names_path = osp.join(self._path, self.localize_path(names_path))

        if has_meta_file(osp.dirname(names_path)):
            return {
                AnnotationType.label: LabelCategories.from_iterable(
                    parse_meta_file(osp.dirname(names_path)).keys()
                )
            }

        label_categories = LabelCategories()

        with open(names_path, "r", encoding="utf-8") as f:
            for label in f:
                label = label.strip()
                if label:
                    label_categories.add(label)

        return {AnnotationType.label: label_categories}

    def __iter__(self):
        subsets = self._subsets
        pbars = self._ctx.progress_reporter.split(len(subsets))
        for pbar, (subset_name, subset) in zip(pbars, subsets.items()):
            for item in pbar.iter(subset, desc=f"Parsing '{subset_name}'"):
                yield item

    def __len__(self):
        return sum(len(s) for s in self._subsets.values())

    def get_subset(self, name):
        return self._subsets[name]


class YOLOv8Extractor(YoloExtractor):
    RESERVED_CONFIG_KEYS = YOLOv8Path.RESERVED_CONFIG_KEYS

    def __init__(
        self,
        *args,
        config_file=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    @cached_property
    def _config(self) -> Dict[str, Any]:
        with open(self._config_path) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError:
                raise InvalidAnnotationError("Failed to parse config file")

    def _load_categories(self) -> Dict[AnnotationType, LabelCategories]:
        if has_meta_file(self._path):
            return {
                AnnotationType.label: LabelCategories.from_iterable(
                    parse_meta_file(self._path).keys()
                )
            }

        if (names := self._config.get("names")) is not None:
            if isinstance(names, dict):
                return {
                    AnnotationType.label: LabelCategories.from_iterable(
                        [names[i] for i in range(len(names))]
                    )
                }
            if isinstance(names, list):
                return {AnnotationType.label: LabelCategories.from_iterable(names)}

        raise InvalidAnnotationError(f"Failed to parse names from config")

    def _get_labels_path_from_image_path(self, image_path: str) -> str:
        relative_image_path = osp.relpath(
            image_path, osp.join(self._path, YOLOv8Path.IMAGES_FOLDER_NAME)
        )
        relative_labels_path = osp.splitext(relative_image_path)[0] + YOLOv8Path.LABELS_EXT
        return osp.join(self._path, YOLOv8Path.LABELS_FOLDER_NAME, relative_labels_path)

    @classmethod
    def name_from_path(cls, path: str) -> str:
        """
        Obtains <image name> from the path like [data/]images/<subset>/<image_name>.ext

        <image name> can be <a/b/c/filename>, so it is
        more involved than just calling "basename()".
        """
        path = cls.localize_path(path)

        parts = split_path(path)
        if 2 < len(parts) and not osp.isabs(path):
            path = osp.join(*parts[2:])  # pylint: disable=no-value-for-parameter
        return osp.splitext(path)[0]

    def _iterate_over_image_paths(
        self, subset_name: str, subset_images_source: Union[str, List[str]]
    ):
        if isinstance(subset_images_source, str):
            if subset_images_source.endswith(YoloPath.SUBSET_LIST_EXT):
                yield from super()._iterate_over_image_paths(subset_name, subset_images_source)
            else:
                path = osp.join(self._path, self.localize_path(subset_images_source))
                if not osp.isdir(path):
                    raise InvalidAnnotationError(f"Can't find '{subset_name}' subset image folder")
                yield from (
                    osp.relpath(osp.join(root, file), self._path)
                    for root, dirs, files in os.walk(path)
                    for file in files
                    if osp.isfile(osp.join(root, file))
                )
        else:
            yield from subset_images_source


class YOLOv8SegmentationExtractor(YOLOv8Extractor):
    def _load_segmentation_annotation(
        self, parts: List[str], image_height: int, image_width: int
    ) -> Polygon:
        label_id = self._parse_field(parts[0], int, "Polygon label id")
        if label_id not in self._categories[AnnotationType.label]:
            raise UndeclaredLabelError(str(label_id))

        points = [
            self._parse_field(
                value, float, f"polygon point {idx // 2} {'x' if idx % 2 == 0 else 'y'}"
            )
            for idx, value in enumerate(parts[1:])
        ]
        scaled_points = [
            value * size for value, size in zip(points, cycle((image_width, image_height)))
        ]
        return Polygon(scaled_points, label=label_id)

    def _load_one_annotation(
        self, parts: List[str], image_height: int, image_width: int
    ) -> Annotation:
        if len(parts) > 5 and len(parts) % 2 == 1:
            return self._load_segmentation_annotation(parts, image_height, image_width)
        raise InvalidAnnotationError(
            f"Unexpected field count {len(parts)} in the polygon description. "
            "Expected odd number > 5 of fields for segment annotation (label, x1, y1, x2, y2, x3, y3, ...)"
        )


class YOLOv8OrientedBoxesExtractor(YOLOv8Extractor):
    @staticmethod
    def _check_is_rectangle(p1, p2, p3, p4):
        p12_angle = math.atan2(p2[0] - p1[0], p2[1] - p1[1])
        p23_angle = math.atan2(p3[0] - p2[0], p3[1] - p2[1])
        p43_angle = math.atan2(p3[0] - p4[0], p3[1] - p4[1])
        p14_angle = math.atan2(p4[0] - p1[0], p4[1] - p1[1])

        if abs(p12_angle - p43_angle) > 0.001 or abs(p23_angle - p14_angle) > 0.001:
            raise InvalidAnnotationError(
                "Given points do not form a rectangle: opposite sides have different slope angles."
            )

    def _load_one_annotation(
        self, parts: List[str], image_height: int, image_width: int
    ) -> Annotation:
        if len(parts) != 9:
            raise InvalidAnnotationError(
                f"Unexpected field count {len(parts)} in the bbox description. "
                "Expected 9 fields (label, x1, y1, x2, y2, x3, y3, x4, y4)."
            )
        label_id = self._parse_field(parts[0], int, "bbox label id")
        if label_id not in self._categories[AnnotationType.label]:
            raise UndeclaredLabelError(str(label_id))
        points = [
            (
                self._parse_field(x, float, f"bbox point {idx} x") * image_width,
                self._parse_field(y, float, f"bbox point {idx} y") * image_height,
            )
            for idx, (x, y) in enumerate(take_by(parts[1:], 2))
        ]
        self._check_is_rectangle(*points)

        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = points

        width = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        height = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        rotation = math.atan2(y2 - y1, x2 - x1)
        if rotation < 0:
            rotation += math.pi * 2

        center_x = (x1 + x2 + x3 + x4) / 4
        center_y = (y1 + y2 + y3 + y4) / 4

        return Bbox(
            x=center_x - width / 2,
            y=center_y - height / 2,
            w=width,
            h=height,
            label=label_id,
            attributes=(dict(rotation=math.degrees(rotation)) if abs(rotation) > 0.00001 else {}),
        )


class YOLOv8PoseExtractor(YOLOv8Extractor):
    @cached_property
    def _kpt_shape(self):
        if YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME not in self._config:
            raise InvalidAnnotationError(
                f"Failed to parse {YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME} from config"
            )
        kpt_shape = self._config[YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME]
        if not isinstance(kpt_shape, list) or len(kpt_shape) != 2:
            raise InvalidAnnotationError(
                f"Failed to parse {YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME} from config"
            )
        if kpt_shape[1] not in [2, 3]:
            raise InvalidAnnotationError(
                f"Unexpected values per point {kpt_shape[1]} in field"
                f"{YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME}. Expected 2 or 3."
            )
        if not isinstance(kpt_shape[0], int) or kpt_shape[0] < 0:
            raise InvalidAnnotationError(
                f"Unexpected number of points {kpt_shape[0]} in field "
                f"{YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME}. Expected non-negative integer."
            )

        return kpt_shape

    @cached_property
    def _skeleton_id_to_label_id(self):
        point_categories = self._categories.get(
            AnnotationType.points, PointsCategories.from_iterable([])
        )
        return {index: label_id for index, label_id in enumerate(sorted(point_categories.items))}

    def _load_categories(self) -> Dict[AnnotationType, LabelCategories]:
        if "names" not in self._config:
            raise InvalidAnnotationError(f"Failed to parse names from config")

        if has_meta_file(self._path):
            dataset_meta = parse_json_file(get_meta_file(self._path))
            point_categories = PointsCategories.from_iterable(
                dataset_meta.get("point_categories", [])
            )
            categories = {
                AnnotationType.label: LabelCategories.from_iterable(
                    dataset_meta["label_categories"]
                )
            }
            if len(point_categories) > 0:
                categories[AnnotationType.points] = point_categories
            return categories

        number_of_points, _ = self._kpt_shape
        names = self._config["names"]
        if isinstance(names, dict):
            if set(names.keys()) != set(range(len(names))):
                raise InvalidAnnotationError(
                    f"Failed to parse names from config: non-sequential label ids"
                )
            skeleton_labels = [names[i] for i in range(len(names))]
        elif isinstance(names, list):
            skeleton_labels = names
        else:
            raise InvalidAnnotationError(f"Failed to parse names from config")

        def make_children_names(skeleton_label):
            return [
                f"{skeleton_label}_point_{point_index}" for point_index in range(number_of_points)
            ]

        point_labels = [
            (child_name, skeleton_label)
            for skeleton_label in skeleton_labels
            for child_name in make_children_names(skeleton_label)
        ]

        point_categories = PointsCategories.from_iterable(
            [
                (
                    index,
                    make_children_names(skeleton_label),
                    set(),
                )
                for index, skeleton_label in enumerate(skeleton_labels)
            ]
        )
        categories = {
            AnnotationType.label: LabelCategories.from_iterable(skeleton_labels + point_labels)
        }
        if len(point_categories) > 0:
            categories[AnnotationType.points] = point_categories

        return categories

    def _load_one_annotation(
        self, parts: List[str], image_height: int, image_width: int
    ) -> Annotation:
        number_of_points, values_per_point = self._kpt_shape
        if len(parts) != 5 + number_of_points * values_per_point:
            raise InvalidAnnotationError(
                f"Unexpected field count {len(parts)} in the skeleton description. "
                "Expected 5 fields (label, xc, yc, w, h)"
                f"and then {values_per_point} for each of {number_of_points} points"
            )

        skeleton_id = self._parse_field(parts[0], int, "skeleton label id")
        label_id = self._skeleton_id_to_label_id.get(skeleton_id, -1)
        if label_id not in self._categories[AnnotationType.label]:
            raise UndeclaredLabelError(str(skeleton_id))
        if self._categories[AnnotationType.label][label_id].parent != "":
            raise InvalidAnnotationError("WTF")

        point_labels = self._categories[AnnotationType.points][label_id].labels
        point_label_ids = [
            self._categories[AnnotationType.label].find(
                name=point_label,
                parent=self._categories[AnnotationType.label][label_id].name,
            )[0]
            for point_label in point_labels
        ]

        points = [
            Points(
                [
                    self._parse_field(parts[x_index], float, f"skeleton point {point_index} x")
                    * image_width,
                    self._parse_field(parts[y_index], float, f"skeleton point {point_index} y")
                    * image_height,
                ],
                (
                    [
                        self._parse_field(
                            parts[visibility_index],
                            int,
                            f"skeleton point {point_index} visibility",
                        ),
                    ]
                    if values_per_point == 3
                    else [Points.Visibility.visible.value]
                ),
                label=label_id,
            )
            for point_index, label_id in enumerate(point_label_ids)
            for x_index, y_index, visibility_index in [
                (
                    5 + point_index * values_per_point,
                    5 + point_index * values_per_point + 1,
                    5 + point_index * values_per_point + 2,
                ),
            ]
        ]
        return Skeleton(
            points,
            label=label_id,
        )
