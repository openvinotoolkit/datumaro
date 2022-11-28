# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os.path as osp
from typing import List, Optional, Tuple, Type, TypeVar

import numpy as np
from defusedxml import ElementTree

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    CompiledMask,
    Label,
    Mask,
)
from datumaro.components.errors import (
    DatasetImportError,
    InvalidAnnotationError,
    InvalidFieldError,
    MissingFieldError,
    UndeclaredLabelError,
)
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.media import Image
from datumaro.util.image import find_images
from datumaro.util.mask_tools import invert_colormap, lazy_mask
from datumaro.util.meta_file_util import has_meta_file

from .format import (
    VocInstColormap,
    VocPath,
    VocTask,
    make_voc_categories,
    parse_label_map,
    parse_meta_file,
)

_inverse_inst_colormap = invert_colormap(VocInstColormap)

T = TypeVar("T")


class _VocExtractor(SubsetBase):
    def __init__(self, path, task, **kwargs):
        if not osp.isfile(path):
            raise DatasetImportError(f"Can't find txt subset list file at '{path}'")
        self._path = path
        self._dataset_dir = osp.dirname(osp.dirname(osp.dirname(path)))

        self._task = task

        super().__init__(subset=osp.splitext(osp.basename(path))[0], **kwargs)

        self._categories = self._load_categories(self._dataset_dir)

        label_color = lambda label_idx: self._categories[AnnotationType.mask].colormap.get(
            label_idx, None
        )
        log.debug(
            "Loaded labels: %s",
            ", ".join(
                "'%s' %s" % (l.name, ("(%s, %s, %s)" % c) if c else "")
                for i, l, c in (
                    (i, l, label_color(i))
                    for i, l in enumerate(self._categories[AnnotationType.label].items)
                )
            ),
        )
        self._items = {item: None for item in self._load_subset_list(path)}

    def _get_label_id(self, label: str) -> int:
        label_id, _ = self._categories[AnnotationType.label].find(label)
        if label_id is None:
            raise UndeclaredLabelError(label)
        return label_id

    def _load_categories(self, dataset_path):
        label_map = None
        if has_meta_file(dataset_path):
            label_map = parse_meta_file(dataset_path)
        else:
            label_map_path = osp.join(dataset_path, VocPath.LABELMAP_FILE)
            if osp.isfile(label_map_path):
                label_map = parse_label_map(label_map_path)

        return make_voc_categories(label_map)

    def _load_subset_list(self, subset_path):
        subset_list = []
        with open(subset_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line[0] == "#":
                    continue

                if self._task == VocTask.person_layout:
                    objects = line.split('"')
                    if 1 < len(objects):
                        if len(objects) == 3:
                            line = objects[1]
                        else:
                            raise InvalidAnnotationError(
                                f"{osp.basename(subset_path)}:{i+1}: "
                                "unexpected number of quotes in filename, expected 0 or 2"
                            )
                    else:
                        line = line.split()[0]
                else:
                    line = line.strip()
                subset_list.append(line)
            return subset_list


class VocClassificationExtractor(_VocExtractor):
    def __init__(self, path, **kwargs):
        super().__init__(path, VocTask.classification, **kwargs)

    def __iter__(self):
        annotations = self._load_annotations()

        image_dir = osp.join(self._dataset_dir, VocPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/"): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        for item_id in self._ctx.progress_reporter.iter(
            self._items, desc=f"Parsing labels in '{self._subset}'"
        ):
            log.debug("Reading item '%s'", item_id)
            image = images.get(item_id)
            if image:
                image = Image(path=image)
            yield DatasetItem(
                id=item_id, subset=self._subset, media=image, annotations=annotations.get(item_id)
            )

    def _load_annotations(self):
        annotations = {}
        task_dir = osp.dirname(self._path)
        for label_id, label in enumerate(self._categories[AnnotationType.label]):
            ann_file = osp.join(task_dir, f"{label.name}_{self._subset}.txt")
            if not osp.isfile(ann_file):
                continue

            with open(ann_file, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line[0] == "#":
                        continue

                    parts = line.rsplit(maxsplit=1)
                    if len(parts) != 2:
                        raise InvalidAnnotationError(
                            f"{osp.basename(ann_file)}:{i+1}: "
                            "invalid number of fields in line, expected 2"
                        )

                    item, present = parts
                    if present not in ["-1", "0", "1"]:
                        # Both -1 and 0 are used in the original VOC, they mean the same
                        raise InvalidAnnotationError(
                            f"{osp.basename(ann_file)}:{i+1}: "
                            f"unexpected class existence value '{present}', expected -1, 0 or 1"
                        )

                    if present == "1":
                        annotations.setdefault(item, []).append(Label(label_id))

        return annotations


class _VocXmlExtractor(_VocExtractor):
    def __iter__(self):
        image_dir = osp.join(self._dataset_dir, VocPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/"): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        anno_dir = osp.join(self._dataset_dir, VocPath.ANNOTATIONS_DIR)

        for item_id in self._ctx.progress_reporter.iter(
            self._items, desc=f"Parsing boxes in '{self._subset}'"
        ):
            log.debug("Reading item '%s'" % item_id)
            size = None

            try:
                anns = []
                image = None

                ann_file = osp.join(anno_dir, item_id + ".xml")
                if osp.isfile(ann_file):
                    root_elem = ElementTree.parse(ann_file).getroot()
                    if root_elem.tag != "annotation":
                        raise MissingFieldError("annotation")

                    height = self._parse_field(root_elem, "size/height", int, required=False)
                    width = self._parse_field(root_elem, "size/width", int, required=False)
                    if height and width:
                        size = (height, width)

                    filename_elem = root_elem.find("filename")
                    if filename_elem is not None:
                        image = osp.join(image_dir, filename_elem.text)

                    anns = self._parse_annotations(root_elem, item_id=(item_id, self._subset))

                if image is None:
                    image = images.pop(item_id, None)

                if image or size:
                    image = Image(path=image, size=size)

                yield DatasetItem(id=item_id, subset=self._subset, media=image, annotations=anns)
            except ElementTree.ParseError as e:
                readable_wrapper = InvalidAnnotationError("Failed to parse XML file")
                readable_wrapper.__cause__ = e
                self._ctx.error_policy.report_item_error(
                    readable_wrapper, item_id=(item_id, self._subset)
                )
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(item_id, self._subset))

    @staticmethod
    def _parse_field(root, xpath: str, cls: Type[T] = str, required: bool = True) -> Optional[T]:
        elem = root.find(xpath)
        if elem is None:
            if required:
                raise MissingFieldError(xpath)
            else:
                return None

        if cls is str:
            return elem.text

        try:
            return cls(elem.text)
        except Exception as e:
            raise InvalidFieldError(xpath) from e

    @staticmethod
    def _parse_bool_field(root, xpath: str, default: bool = False) -> Optional[bool]:
        elem = root.find(xpath)
        if elem is None:
            return default

        if elem.text not in ["0", "1"]:
            raise InvalidFieldError(xpath)
        return elem.text == "1"

    def _parse_annotations(self, root_elem, *, item_id: Tuple[str, str]) -> List[Annotation]:
        item_annotations = []

        for obj_id, object_elem in enumerate(root_elem.iterfind("object")):
            try:
                obj_id += 1
                attributes = {}
                group = obj_id

                obj_label_id = self._get_label_id(self._parse_field(object_elem, "name"))

                obj_bbox = self._parse_bbox(object_elem)

                for key in ["difficult", "truncated", "occluded"]:
                    attributes[key] = self._parse_bool_field(object_elem, key, default=False)

                pose_elem = object_elem.find("pose")
                if pose_elem is not None:
                    attributes["pose"] = pose_elem.text

                point_elem = object_elem.find("point")
                if point_elem is not None:
                    point_x = self._parse_field(point_elem, "x", float)
                    point_y = self._parse_field(point_elem, "y", float)
                    attributes["point"] = (point_x, point_y)

                actions_elem = object_elem.find("actions")
                actions = {
                    a: False
                    for a in self._categories[AnnotationType.label].items[obj_label_id].attributes
                }
                if actions_elem is not None:
                    for action_elem in actions_elem:
                        actions[action_elem.tag] = self._parse_bool_field(
                            actions_elem, action_elem.tag
                        )
                for action, present in actions.items():
                    attributes[action] = present

                has_parts = False
                for part_elem in object_elem.findall("part"):
                    part_label_id = self._get_label_id(self._parse_field(part_elem, "name"))
                    part_bbox = self._parse_bbox(part_elem)

                    if self._task is not VocTask.person_layout:
                        break
                    has_parts = True
                    item_annotations.append(Bbox(*part_bbox, label=part_label_id, group=group))

                attributes_elem = object_elem.find("attributes")
                if attributes_elem is not None:
                    for attr_elem in attributes_elem.iter("attribute"):
                        attributes[self._parse_field(attr_elem, "name")] = self._parse_field(
                            attr_elem, "value"
                        )

                if self._task is VocTask.person_layout and not has_parts:
                    continue
                if self._task is VocTask.action_classification and not actions:
                    continue

                item_annotations.append(
                    Bbox(
                        *obj_bbox, label=obj_label_id, attributes=attributes, id=obj_id, group=group
                    )
                )
            except Exception as e:
                self._ctx.error_policy.report_annotation_error(e, item_id=item_id)

        return item_annotations

    @classmethod
    def _parse_bbox(cls, object_elem):
        bbox_elem = object_elem.find("bndbox")
        if not bbox_elem:
            raise MissingFieldError("bndbox")

        xmin = cls._parse_field(bbox_elem, "xmin", float)
        xmax = cls._parse_field(bbox_elem, "xmax", float)
        ymin = cls._parse_field(bbox_elem, "ymin", float)
        ymax = cls._parse_field(bbox_elem, "ymax", float)
        return [xmin, ymin, xmax - xmin, ymax - ymin]


class VocDetectionExtractor(_VocXmlExtractor):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.detection, **kwargs)


class VocLayoutExtractor(_VocXmlExtractor):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.person_layout, **kwargs)


class VocActionExtractor(_VocXmlExtractor):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.action_classification, **kwargs)


class VocSegmentationExtractor(_VocExtractor):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.segmentation, **kwargs)

    def __iter__(self):
        image_dir = osp.join(self._dataset_dir, VocPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/"): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        for item_id in self._ctx.progress_reporter.iter(
            self._items, desc=f"Parsing segmentation in '{self._subset}'"
        ):
            log.debug("Reading item '%s'", item_id)

            image = images.get(item_id)
            if image:
                image = Image(path=image)

            try:
                yield DatasetItem(
                    id=item_id,
                    subset=self._subset,
                    media=image,
                    annotations=self._load_annotations(item_id),
                )
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(item_id, self._subset))

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

    def _load_annotations(self, item_id):
        item_annotations = []

        class_mask = None
        segm_path = osp.join(
            self._dataset_dir, VocPath.SEGMENTATION_DIR, item_id + VocPath.SEGM_EXT
        )
        if osp.isfile(segm_path):
            inverse_cls_colormap = self._categories[AnnotationType.mask].inverse_colormap
            class_mask = lazy_mask(segm_path, inverse_cls_colormap)

        instances_mask = None
        inst_path = osp.join(self._dataset_dir, VocPath.INSTANCES_DIR, item_id + VocPath.SEGM_EXT)
        if osp.isfile(inst_path):
            instances_mask = lazy_mask(inst_path, _inverse_inst_colormap)

        label_cat = self._categories[AnnotationType.label]

        if instances_mask is not None:
            compiled_mask = CompiledMask(class_mask, instances_mask)

            if class_mask is not None:
                instance_labels = compiled_mask.get_instance_labels()
            else:
                instance_labels = {i: None for i in range(compiled_mask.instance_count)}

            for instance_id, label_id in instance_labels.items():
                if len(label_cat) <= label_id:
                    self._ctx.error_policy.report_annotation_error(
                        UndeclaredLabelError(str(label_id)), item_id=(item_id, self._subset)
                    )

                image = compiled_mask.lazy_extract(instance_id)

                item_annotations.append(Mask(image=image, label=label_id, group=instance_id))
        elif class_mask is not None:
            log.warning("Item %s: only class segmentations available", item_id)

            class_mask = class_mask()
            classes = np.unique(class_mask)
            for label_id in classes:
                if len(label_cat) <= label_id:
                    self._ctx.error_policy.report_annotation_error(
                        UndeclaredLabelError(str(label_id)), item_id=(item_id, self._subset)
                    )

                image = self._lazy_extract_mask(class_mask, label_id)
                item_annotations.append(Mask(image=image, label=label_id))

        return item_annotations
