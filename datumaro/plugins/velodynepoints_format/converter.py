
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import logging as log
from collections import OrderedDict
from itertools import chain
from xml.sax.saxutils import XMLGenerator

from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.extractor import (AnnotationType, DatasetItem,
    LabelCategories)
from datumaro.util import cast
from datumaro.util.image import ByteImage, save_image

from .format import VelodynePointsPath


class XmlAnnotationWriter:

    def __init__(self, file, tracklets):
        self.version = "1.1"
        self._file = file
        self.xmlgen = XMLGenerator(self._file, 'utf-8')
        self._level = 0
        self._tracklets = tracklets
        self._class_id = 0
        self._tracking_level = 0
        self._item_version = 1
        self._pose_state = True
        self._item_state = True
        self._item = 1
        self._header = """<?xml version="{}" encoding="{}" standalone="{}" ?>""".format("1.0", 'UTF-8', 'yes')
        self._doctype = """<!DOCTYPE {}>""".format('boost_serialization')
        self._serialization = """<boost_serialization signature="{}" version="{}">""".format("serialization::archive",
                                                                                             "9")

    def _indent(self, newline=True):
        if newline:
            self.xmlgen.ignorableWhitespace("\n")
        self.xmlgen.ignorableWhitespace("  " * self._level)

    def _write_headers(self):
        self._file.write(self._header)

    def _write_doctype(self):
        self._indent(newline=True)
        self._file.write(self._doctype)

    def _open_serialization(self):
        self._indent(newline=True)
        self.xmlgen.startElement("boost_serialization", {"signature": "serialization::archive", "version": "9"})

    def _close_serialization(self):
        self._indent(newline=True)
        self.xmlgen.endElement("boost_serialization")

    def _add_count(self, item):
        self.xmlgen.startElement("count", {})
        self.xmlgen.characters(str(len(item)))
        self.xmlgen.endElement("count")

    def _open_tracklet(self):
        self._indent(newline=True)
        self.xmlgen.startElement("tracklets",
                                 {"class_id": str(self._class_id), "tracking_level": str(self._tracking_level),
                                  "version": "0"})
        self._class_id += 1
        self._level += 1
        self._indent()

    def open_root(self):
        self._write_headers()
        self._write_doctype()
        self._open_serialization()
        self._open_tracklet()

    def _add_item_version(self):
        self._indent(newline=True)
        self.xmlgen.startElement("item_version", {})
        self.xmlgen.characters(str(self._item_version))
        self._item_version += 1
        self.xmlgen.endElement("item_version")

    def _close_tracklet(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("tracklets")

    def _start_item(self):
        self._indent()
        if self._item_state:
            self.xmlgen.startElement("item",
                                     {"class_id": str(self._class_id), "tracking_level": str(self._tracking_level),
                                      "version": str(self._item)})
            if self._item == 2:
                self._item_state = False
        else:
            self.xmlgen.startElement("item", {})
        self._item += 1
        self._class_id += 1
        self._level += 1

    def _end_item(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("item")

    def _open_pose(self):
        self._indent(newline=True)
        if self._pose_state:
            self.xmlgen.startElement("poses",
                                     {"class_id": str(self._class_id), "tracking_level": str(self._tracking_level),
                                      "version": "0"})
            self._class_id += 1
            self._pose_state = False
        else:
            self.xmlgen.startElement("poses", {})
        self._level += 1
        self._indent()

    def _close_pose(self):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement("poses")

    def _add_pose(self, poses):
        self._open_pose()
        self._add_count(poses)
        self._add_item_version()
        for pose in poses:
            self._start_item()
            for element, value in pose.items():
                self._indent(newline=True)
                self.xmlgen.startElement(element, {})
                self.xmlgen.characters(str(value))
                self.xmlgen.endElement(element)
            self._end_item()
        self._close_pose()

    def generate_tracklets(self):
        self._write_headers()
        self._write_doctype()
        self._open_serialization()
        self._open_tracklet()
        if self._tracklets:
            self._add_count(self._tracklets)
            self._add_item_version()

            for tracklet in self._tracklets:
                self._start_item()
                for element, value in tracklet.items():
                    if isinstance(value, list):
                        self._add_pose(value)
                    else:
                        self._indent(newline=True)
                        self.xmlgen.startElement(element, {})
                        self.xmlgen.characters(str(value))
                        self.xmlgen.endElement(element)

                self._end_item()
        self._close_tracklet()
        self._close_serialization()


class _SubsetWriter:
    def __init__(self, file, name, extractor, context):
        self._file = file
        self._name = name
        self._extractor = extractor
        self._context = context
        self._tracklets = []
        self.create_tracklets(self._extractor)

    def create_tracklets(self, subset):

        for i, data in enumerate(subset):

            index = self._write_item(data, i)
            for item in data.annotations:
                if item.type == AnnotationType.cuboid:
                    if item.label is None:
                        log.warning("Item %s: skipping a %s with no label",
                                    item.id, item.type.name)

                    label_name = self._get_label(item.label).name

                    tracklet = {
                        "objectType": label_name,
                        "h": item.points[0],
                        "w": item.points[1],
                        "l": item.points[2],
                        "first_frame": index if index is not None else data.attributes.get('frame', 0),
                        "poses": []
                    }
                    pose = {
                        "tx": item.points[3],
                        "ty": item.points[4],
                        "tz": item.points[5],
                        "rx": item.points[6],
                        "ry": item.points[7],
                        "rz": item.points[8],
                        "state": 2,
                        "occlusion": -1,
                        "occlusion_kf": 1 if item.attributes.get("occluded", False) else 0,
                        "truncation": -1,
                        "amt_occlusion": -1,
                        "amt_border_l": -1,
                        "amt_border_r": -1,
                        "amt_occlusion_kf": -1,
                        "amt_border_kf": -1,
                    }
                    tracklet["poses"].append(pose)
                    tracklet["finished"] = 1
                    self._tracklets.append(tracklet)

                    label_name = self._get_label(item.label).name
                    for attr_name, attr_value in item.attributes.items():
                        if attr_name in self._context._builtin_attrs:
                            continue
                        if isinstance(attr_value, bool):
                            attr_value = 'true' if attr_value else 'false'
                        if self._context._allow_undeclared_attrs or \
                                attr_name in self._get_label_attrs(item.label):
                            continue
                        else:
                            log.warning("Item %s: skipping undeclared "
                                        "attribute '%s' for label '%s' "
                                        "(allow with --allow-undeclared-attrs option)",
                                        item.id, attr_name, label_name)

        tracklets = XmlAnnotationWriter(self._file, self._tracklets)
        tracklets.generate_tracklets()

    def _get_label(self, label_id):
        if label_id is None:
            return ""
        label_cat = self._extractor.categories().get(
            AnnotationType.label, LabelCategories())
        return label_cat.items[label_id]

    def _get_label_attrs(self, label):
        label_cat = self._extractor.categories().get(
            AnnotationType.label, LabelCategories())
        if isinstance(label, int):
            label = label_cat[label]
        return set(chain(label.attributes, label_cat.attributes)) - \
            self._context._builtin_attrs

    def _write_item(self, item, index):
        if not self._context._reindex:
            index = cast(item.attributes.get('frame'), int, index)
        image_info = OrderedDict([("id", str(index)), ])
        if isinstance(item.pcd, str) and osp.isfile(item.pcd):
            path = item.pcd.replace(os.sep, '/')
            filename = path.rsplit("/", maxsplit=1)[-1]
        else:
            filename = self._context._make_pcd_filename(item)
        image_info["name"] = filename

        if item.has_pcd:
            if self._context._save_images:
                velodyne_dir = osp.join(self._context._save_dir, "velodyne_points")
                self._context._image_dir = osp.join(velodyne_dir, "data")

                self._context._save_pcd(item,
                                        osp.join(self._context._image_dir, filename))

                if item.related_images:

                    for i, related_image in enumerate(item.related_images):

                        image_path = related_image["save_path"]
                        related_dir = osp.join(self._context._save_dir, image_path)
                        path = osp.join(related_dir, "data")
                        os.makedirs(path, exist_ok=True)

                        try:
                            name = related_image["name"]
                            image = related_image["image"]
                        except AttributeError:
                            name = f"{i}.jpg"

                        path = osp.join(path, name)

                        if isinstance(image, ByteImage):
                            with open(path, 'wb') as f:
                                f.write(image.get_bytes())
                        else:
                            save_image(path, image.data)

        else:
            log.debug("Item '%s' has no image info", item.id)

        if index is not None:
            return index

class VelodynePointsConverter(Converter):
    DEFAULT_IMAGE_EXT = ".pcd"

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('--reindex', action='store_true',
            help="Assign new indices to frames (default: %(default)s)")
        parser.add_argument('--allow-undeclared-attrs', action='store_true',
            help="Write annotation attributes even if they are not present in "
                "the input dataset metainfo (default: %(default)s)")
        return parser

    def __init__(self, extractor, save_dir, reindex=False,
            allow_undeclared_attrs=False, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        self._reindex = reindex
        self._builtin_attrs = VelodynePointsPath.BUILTIN_ATTRS
        self._allow_undeclared_attrs = allow_undeclared_attrs

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():

            if subset_name != "tracklets":
                subset_name = "tracklets"

            with open(osp.join(self._save_dir, f'{subset_name}.xml'), 'w') as f:
                _SubsetWriter(f, subset_name, subset, self)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            cls.convert(dataset.get_subset(subset), save_dir=save_dir, **kwargs)

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        pcd_dir = osp.abspath(osp.join(save_dir, "velodyne_points/data"))

        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.has_pcd):
                continue

            pcd_path = osp.join(pcd_dir, conv._make_pcd_filename(item))
            if osp.isfile(pcd_path):
                os.unlink(pcd_path)

            if kwargs:
                for path in kwargs.get('related_paths'):
                    image_dir = osp.abspath(osp.join(save_dir,path))
                for image in kwargs["image_names"]:
                    image_path = osp.join(image_dir, image)
                    if osp.isfile(image_path):
                        os.unlink(image_path)

