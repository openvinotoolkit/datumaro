# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import OrderedDict
from copy import deepcopy

from defusedxml import ElementTree

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Points,
    Polygon,
    PolyLine,
)
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.media import Image

from .format import CvatPath


def _find_meta_root(path: str):
    context = ElementTree.iterparse(path, events=("start", "end"))
    context = iter(context)

    meta_root = None

    for event, elem in context:
        if elem.tag == "meta" and event == "start":
            meta_root = elem
        elif elem.tag == "meta" and event == "end":
            break

    if meta_root is None:
        raise DatasetImportError("CVAT XML file should have <meta> tag.")

    return meta_root, context


class CvatBase(SubsetBase):
    _SUPPORTED_SHAPES = ("box", "polygon", "polyline", "points")

    def __init__(self, path, subset=None, save_hash=False):
        assert osp.isfile(path), path
        rootpath = osp.dirname(path)
        images_dir = ""
        if osp.isdir(osp.join(rootpath, CvatPath.IMAGES_DIR)):
            images_dir = osp.join(rootpath, CvatPath.IMAGES_DIR)
        self._images_dir = images_dir
        self._path = path
        self._save_hash = save_hash

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        super().__init__(subset=subset)

        items, categories = self._parse(path)
        self._items = list(self._load_items(items).values())
        self._categories = categories

    def _parse(self, path):
        meta_root, context = _find_meta_root(path)

        categories, frame_size, attribute_types = self._parse_meta(meta_root)

        items = OrderedDict()

        track = None
        shape = None
        tag = None
        attributes = None
        image = None
        subset = None
        for ev, el in context:
            if ev == "start":
                if el.tag == "track":
                    track = {
                        "id": el.attrib["id"],
                        "label": el.attrib.get("label"),
                        "group": int(el.attrib.get("group_id", 0)),
                        "height": frame_size[0],
                        "width": frame_size[1],
                    }
                    subset = el.attrib.get("subset")
                elif el.tag == "image":
                    image = {
                        "name": el.attrib.get("name"),
                        "frame": el.attrib["id"],
                        "width": el.attrib.get("width"),
                        "height": el.attrib.get("height"),
                    }
                    subset = el.attrib.get("subset")
                elif el.tag in self._SUPPORTED_SHAPES and (track or image):
                    attributes = {}
                    shape = {
                        "type": None,
                        "attributes": attributes,
                    }
                    if track:
                        shape.update(track)
                        shape["track_id"] = int(track["id"])
                    if image:
                        shape.update(image)
                elif el.tag == "tag" and image:
                    attributes = {}
                    tag = {
                        "frame": image["frame"],
                        "attributes": attributes,
                        "group": int(el.attrib.get("group_id", 0)),
                        "label": el.attrib["label"],
                    }
            elif ev == "end":
                if el.tag == "attribute" and attributes is not None:
                    attr_value = el.text or ""
                    attr_type = attribute_types.get(el.attrib["name"])
                    if el.text in ["true", "false"]:
                        attr_value = attr_value == "true"
                    elif attr_type is not None and attr_type != "text":
                        try:
                            attr_value = float(attr_value)
                        except ValueError:
                            pass
                    attributes[el.attrib["name"]] = attr_value
                elif el.tag in self._SUPPORTED_SHAPES:
                    if track is not None:
                        shape["frame"] = el.attrib["frame"]
                        shape["outside"] = el.attrib.get("outside") == "1"
                        shape["keyframe"] = el.attrib.get("keyframe") == "1"
                    if image is not None:
                        shape["label"] = el.attrib.get("label")
                        shape["group"] = int(el.attrib.get("group_id", 0))

                    shape["type"] = el.tag
                    shape["occluded"] = el.attrib.get("occluded") == "1"
                    shape["z_order"] = int(el.attrib.get("z_order", 0))

                    if el.tag == "box":
                        shape["points"] = list(
                            map(
                                float,
                                [
                                    el.attrib["xtl"],
                                    el.attrib["ytl"],
                                    el.attrib["xbr"],
                                    el.attrib["ybr"],
                                ],
                            )
                        )
                    else:
                        shape["points"] = []
                        for pair in el.attrib["points"].split(";"):
                            shape["points"].extend(map(float, pair.split(",")))

                    if subset is None or subset == self._subset:
                        frame_desc = items.get(shape["frame"], {"annotations": []})
                        frame_desc["annotations"].append(self._parse_shape_ann(shape, categories))
                        items[shape["frame"]] = frame_desc
                    shape = None

                elif el.tag == "tag":
                    if subset is None or subset == self._subset:
                        frame_desc = items.get(tag["frame"], {"annotations": []})
                        frame_desc["annotations"].append(self._parse_tag_ann(tag, categories))
                        items[tag["frame"]] = frame_desc
                    tag = None
                elif el.tag == "track":
                    track = None
                elif el.tag == "image":
                    if subset is None or subset == self._subset:
                        frame_desc = items.get(image["frame"], {"annotations": []})
                        frame_desc.update(
                            {
                                "name": image.get("name"),
                                "height": image.get("height"),
                                "width": image.get("width"),
                            }
                        )
                        items[image["frame"]] = frame_desc
                    image = None
                el.clear()

        return items, categories

    @staticmethod
    def _parse_meta(meta_root):
        categories = {}

        frame_size = None
        original_size = [item for item in meta_root.iter("original_size")]

        if len(original_size) > 1:
            raise DatasetImportError("CVAT XML file should have only one <original_size> tag.")
        elif len(original_size) == 1:
            frame_size = (
                int(original_size[0].find("height").text),
                int(original_size[0].find("width").text),
            )

        mode = None
        labels = OrderedDict()

        for label in meta_root.iter("label"):
            name = label.find("name").text
            labels[name] = [
                {
                    "name": attr.find("name").text,
                    "input_type": attr.find("input_type").text,
                }
                for attr in label.iter("attribute")
            ]

        common_attrs = ["occluded"]
        if mode == "interpolation":
            common_attrs.append("keyframe")
            common_attrs.append("outside")
            common_attrs.append("track_id")

        label_cat = LabelCategories(attributes=common_attrs)
        attribute_types = {}
        for label, attrs in labels.items():
            attr_names = {v["name"] for v in attrs}
            label_cat.add(label, attributes=attr_names)
            for attr in attrs:
                attribute_types[attr["name"]] = attr["input_type"]

        categories[AnnotationType.label] = label_cat
        return categories, frame_size, attribute_types

    @classmethod
    def _parse_shape_ann(cls, ann, categories):
        ann_id = ann.get("id", 0)
        ann_type = ann["type"]

        attributes = ann.get("attributes") or {}
        if "occluded" in categories[AnnotationType.label].attributes:
            attributes["occluded"] = ann.get("occluded", False)
        if "outside" in ann:
            attributes["outside"] = ann["outside"]
        if "keyframe" in ann:
            attributes["keyframe"] = ann["keyframe"]
        if "track_id" in ann:
            attributes["track_id"] = ann["track_id"]

        group = ann.get("group")

        label = ann.get("label")
        label_id = categories[AnnotationType.label].find(label)[0]

        z_order = ann.get("z_order", 0)
        points = ann.get("points", [])

        if ann_type == "polyline":
            return PolyLine(
                points,
                label=label_id,
                z_order=z_order,
                id=ann_id,
                attributes=attributes,
                group=group,
            )

        elif ann_type == "polygon":
            return Polygon(
                points,
                label=label_id,
                z_order=z_order,
                id=ann_id,
                attributes=attributes,
                group=group,
            )

        elif ann_type == "points":
            return Points(
                points,
                label=label_id,
                z_order=z_order,
                id=ann_id,
                attributes=attributes,
                group=group,
            )

        elif ann_type == "box":
            x, y = points[0], points[1]
            w, h = points[2] - x, points[3] - y
            return Bbox(
                x,
                y,
                w,
                h,
                label=label_id,
                z_order=z_order,
                id=ann_id,
                attributes=attributes,
                group=group,
            )

        else:
            raise NotImplementedError("Unknown annotation type '%s'" % ann_type)

    @classmethod
    def _parse_tag_ann(cls, ann, categories):
        label = ann.get("label")
        label_id = categories[AnnotationType.label].find(label)[0]
        group = ann.get("group")
        attributes = ann.get("attributes")
        return Label(label_id, attributes=attributes, group=group)

    def _load_items(self, parsed):
        for frame_id, item_desc in parsed.items():
            name = item_desc.get("name", "frame_%06d.png" % int(frame_id))
            image = osp.join(self._images_dir, name)
            image_size = (item_desc.get("height"), item_desc.get("width"))
            if all(image_size):
                image = Image(path=image, size=tuple(map(int, image_size)))
            else:
                image = Image(path=image)

            subset = item_desc.get("subset")
            if subset is not None and subset != self._subset:
                continue
            parsed[frame_id] = DatasetItem(
                id=osp.splitext(name)[0],
                subset=self._subset,
                media=image,
                annotations=item_desc.get("annotations"),
                attributes={"frame": int(frame_id)},
                save_hash=self._save_hash,
            )
        return parsed


class CvatImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        annot_file = context.require_file("*.xml")

        with context.probe_text_file(
            annot_file,
            'must be an XML file with an "annotations" root element',
        ) as f:
            _, root_elem = next(ElementTree.iterparse(f, events=("start",)))
            if root_elem.tag != "annotations":
                raise Exception

    @staticmethod
    def find_subsets(meta_root):
        subsets = [item.text for item in meta_root.iter("subset")]
        if len(subsets) == 0:
            raise DatasetImportError("CVAT XML should include <subset> tags.")
        return subsets

    @classmethod
    def find_sources(cls, path):
        source_files = cls._find_sources_recursive(path, ".xml", "cvat")
        sources = []

        for source in source_files:
            path = source["url"]
            meta_root, _ = _find_meta_root(path)

            if meta_root.find("project") is not None:
                for subset in cls.find_subsets(meta_root):
                    source_clone = deepcopy(source)
                    source_clone["options"] = {"subset": subset}
                    sources += [source_clone]
            elif meta_root.find("task") is not None:
                sources += [source]
            else:
                raise DatasetImportError(
                    "CVAT XML file should have a <meta> -> <task> or <meta> -> <project> subtree."
                )

        return sources
