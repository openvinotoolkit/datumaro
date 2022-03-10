# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT
import glob
import logging as log
import os
import os.path as osp

import numpy as np

from datumaro.components.annotation import (
    AnnotationType,
    CompiledMask,
    LabelCategories,
    Mask,
    MaskCategories,
    Polygon,
)
from datumaro.components.extractor import DatasetItem, SourceExtractor
from datumaro.components.media import Image
from datumaro.util import parse_json_file
from datumaro.util.image import find_images, lazy_image, load_image
from datumaro.util.mask_tools import bgr2index
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

from .format import (
    MapillaryVistasLabelMaps,
    MapillaryVistasPath,
    MapillaryVistasTask,
    make_mapillary_instance_categories,
    parse_config_file,
)


class _MapillaryVistasExtractor(SourceExtractor):
    def __init__(
        self, path, task, subset=None, use_original_config=False, keep_original_category_ids=False
    ):
        assert osp.isdir(path), path
        self._path = path
        if subset is None:
            subset = osp.basename(self._path)
        super().__init__(subset=subset)

        annotations_dirs = [d for d in os.listdir(path) if d in MapillaryVistasPath.ANNOTATION_DIRS]

        if len(annotations_dirs) == 0:
            raise NotADirectoryError(
                "Can't find annotation directory at %s. "
                "Expected one of these directories: %s"
                % (path, ",".join(MapillaryVistasPath.ANNOTATIONS_DIR_PATTERNS))
            )
        elif len(annotations_dirs) > 1:
            log.warning(
                "Directory(-es): %s will be skipped, dataset should contain "
                "only one annotation directory" % ",".join(annotations_dirs[1:])
            )

        self._use_original_config = use_original_config
        self._format_version = annotations_dirs[0]
        self._annotations_dir = osp.join(path, annotations_dirs[0])
        self._images_dir = osp.join(path, MapillaryVistasPath.IMAGES_DIR)
        self._task = task

        if self._task == MapillaryVistasTask.instances:
            if has_meta_file(path):
                self._categories = make_mapillary_instance_categories(parse_meta_file(path))
            else:
                self._categories = self._load_instances_categories()
            self._items = self._load_instances_items()
        else:
            panoptic_config = self._load_panoptic_config(self._annotations_dir)
            self._categories = self._load_panoptic_categories(
                panoptic_config["categories"], keep_original_category_ids
            )
            self._items = self._load_panoptic_items(panoptic_config)

    @staticmethod
    def _load_panoptic_config(path):
        panoptic_config_path = osp.join(
            path, MapillaryVistasPath.PANOPTIC_DIR, MapillaryVistasPath.PANOPTIC_CONFIG
        )

        if not osp.isfile(panoptic_config_path):
            raise FileNotFoundError(
                "Can't find panoptic config file: '%s' at '%s'"
                % (MapillaryVistasPath.PANOPTIC_CONFIG, panoptic_config_path)
            )

        return parse_json_file(panoptic_config_path)

    def _load_panoptic_categories(self, categories_info, keep_original_ids):
        label_cat = LabelCategories()
        label_map, color_map = {}, {}

        if keep_original_ids:
            for cat in sorted(categories_info, key=lambda cat: cat["id"]):
                label_map[cat["id"]] = cat["id"]
                color_map[cat["id"]] = tuple(map(int, cat["color"]))

                while len(label_cat) < cat["id"]:
                    label_cat.add(name=f"class-{len(label_cat)}")

                label_cat.add(name=cat["name"], parent=cat.get("supercategory"))
        else:
            for idx, cat in enumerate(categories_info):
                label_map[cat["id"]] = idx
                color_map[idx] = tuple(map(int, cat["color"]))

                label_cat.add(name=cat["name"], parent=cat.get("supercategory"))

        self._label_map = label_map
        mask_cat = MaskCategories(color_map)
        mask_cat.inverse_colormap  # pylint: disable=pointless-statement

        return {AnnotationType.label: label_cat, AnnotationType.mask: mask_cat}

    def _load_panoptic_items(self, config):
        items = {}

        images_info = {
            img["id"]: {
                "path": osp.join(self._images_dir, img["file_name"]),
                "height": img.get("height"),
                "width": img.get("width"),
            }
            for img in config["images"]
        }

        for item_ann in config["annotations"]:
            item_id = item_ann["image_id"]
            image = None
            if images_info.get(item_id):
                image = Image(
                    path=images_info[item_id]["path"],
                    size=self._get_image_size(images_info[item_id]),
                )

            annotations = []
            mask_path = osp.join(
                self._annotations_dir, MapillaryVistasPath.PANOPTIC_DIR, item_ann["file_name"]
            )
            mask = lazy_image(mask_path, loader=self._load_pan_mask)
            mask = CompiledMask(instance_mask=mask)

            for segment_info in item_ann["segments_info"]:
                cat_id = self._get_label_id(segment_info)
                segment_id = segment_info["id"]
                attributes = {"is_crowd": bool(segment_info["iscrowd"])}
                annotations.append(
                    Mask(
                        image=mask.lazy_extract(segment_id),
                        label=cat_id,
                        id=segment_id,
                        group=segment_id,
                        attributes=attributes,
                    )
                )

            items[item_id] = DatasetItem(
                id=item_id, subset=self._subset, annotations=annotations, media=image
            )

        self._load_polygons(items)
        return items.values()

    def _load_instances_categories(self):
        config_file = MapillaryVistasPath.CONFIG_FILES[self._format_version]
        label_map = None

        if self._use_original_config:
            label_map = MapillaryVistasLabelMaps[self._format_version]
        else:
            try:
                label_map = parse_config_file(osp.join(self._path, config_file))
            except FileNotFoundError:
                label_map = parse_config_file(osp.join(osp.dirname(self._path), config_file))
        return make_mapillary_instance_categories(label_map)

    def _load_instances_items(self):
        items = {}

        instances_dir = osp.join(self._annotations_dir, MapillaryVistasPath.INSTANCES_DIR)
        for instance_path in find_images(instances_dir, recursive=True):
            item_id = osp.splitext(osp.relpath(instance_path, instances_dir))[0]

            mask = load_image(instance_path, dtype=np.uint32)

            annotations = []
            for uval in np.unique(mask):
                label_id, instance_id = uval >> 8, uval & 255
                annotations.append(
                    Mask(image=self._lazy_extract_mask(mask, uval), label=label_id, id=instance_id)
                )

            items[item_id] = DatasetItem(id=item_id, subset=self._subset, annotations=annotations)

        class_dir = osp.join(self._annotations_dir, MapillaryVistasPath.CLASS_DIR)
        for class_path in find_images(class_dir, recursive=True):
            item_id = osp.splitext(osp.relpath(class_path, class_dir))[0]
            if item_id in items:
                continue

            from PIL import Image as PILImage

            class_mask = np.array(PILImage.open(class_path))
            classes = np.unique(class_mask)

            annotations = []
            for label_id in classes:
                annotations.append(
                    Mask(label=label_id, image=self._lazy_extract_mask(class_mask, label_id))
                )

            items[item_id] = DatasetItem(id=item_id, subset=self._subset, annotations=annotations)

        for image_path in find_images(self._images_dir, recursive=True):
            item_id = osp.splitext(osp.relpath(image_path, self._images_dir))[0]
            image = Image(path=image_path)
            if item_id in items:
                items[item_id].media = image
            else:
                items[item_id] = DatasetItem(id=item_id, subset=self._subset, media=image)

        self._load_polygons(items)
        return items.values()

    def _load_polygons(self, items):
        polygons_dir = osp.join(self._annotations_dir, MapillaryVistasPath.POLYGON_DIR)
        for item_path in glob.glob(osp.join(polygons_dir, "**", "*.json"), recursive=True):
            item_id = osp.splitext(osp.relpath(item_path, polygons_dir))[0]
            item = items.get(item_id)
            item_info = {}
            item_info = parse_json_file(item_path)

            image_size = self._get_image_size(item_info)
            if image_size and item.has_image:
                item.media = Image(path=item.image.path, size=image_size)

            polygons = item_info["objects"]
            annotations = []
            for polygon in polygons:
                label = polygon["label"]
                label_id = self._categories[AnnotationType.label].find(label)[0]
                if label_id is None:
                    label_id = self._categories[AnnotationType.label].add(label)

                points = [coord for point in polygon["polygon"] for coord in point]
                annotations.append(Polygon(label=label_id, points=points))

            if item is None:
                items[item_id] = DatasetItem(
                    id=item_id, subset=self._subset, annotations=annotations
                )
            else:
                item.annotations.extend(annotations)

    @staticmethod
    def _get_image_size(image_info):
        image_size = image_info.get("height"), image_info.get("width")
        if all(image_size):
            return int(image_size[0]), int(image_size[1])
        return None

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

    @staticmethod
    def _load_pan_mask(path):
        mask = load_image(path)
        mask = bgr2index(mask)
        return mask

    def _get_label_id(self, ann):
        cat_id = ann.get("category_id")
        if cat_id in [0, None]:
            return None
        return self._label_map[cat_id]


class MapillaryVistasInstancesExtractor(_MapillaryVistasExtractor):
    def __init__(self, path, **kwargs):
        kwargs["task"] = MapillaryVistasTask.instances
        super().__init__(path, **kwargs)


class MapillaryVistasPanopticExtractor(_MapillaryVistasExtractor):
    def __init__(self, path, **kwargs):
        kwargs["task"] = MapillaryVistasTask.panoptic
        super().__init__(path, **kwargs)
