# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from itertools import chain
from xml.sax.saxutils import XMLGenerator
import logging as log
import os
import os.path as osp

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.extractor import DatasetItem
from datumaro.util import cast, pairs

from .format import CvatPath


class XmlAnnotationWriter:
    VERSION = '1.1'

    def __init__(self, f):
        self.xmlgen = XMLGenerator(f, 'utf-8')
        self._level = 0

    def _indent(self, newline = True):
        if newline:
            self.xmlgen.ignorableWhitespace('\n')
        self.xmlgen.ignorableWhitespace('  ' * self._level)

    def _add_version(self):
        self._indent()
        self.xmlgen.startElement('version', {})
        self.xmlgen.characters(self.VERSION)
        self.xmlgen.endElement('version')

    def open_root(self):
        self.xmlgen.startDocument()
        self.xmlgen.startElement('annotations', {})
        self._level += 1
        self._add_version()

    def _add_meta(self, meta):
        self._level += 1
        for k, v in meta.items():
            if isinstance(v, OrderedDict):
                self._indent()
                self.xmlgen.startElement(k, {})
                self._add_meta(v)
                self._indent()
                self.xmlgen.endElement(k)
            elif isinstance(v, list):
                self._indent()
                self.xmlgen.startElement(k, {})
                for tup in v:
                    self._add_meta(OrderedDict([tup]))
                self._indent()
                self.xmlgen.endElement(k)
            else:
                self._indent()
                self.xmlgen.startElement(k, {})
                self.xmlgen.characters(v)
                self.xmlgen.endElement(k)
        self._level -= 1

    def write_meta(self, meta):
        self._indent()
        self.xmlgen.startElement('meta', {})
        self._add_meta(meta)
        self._indent()
        self.xmlgen.endElement('meta')

    def open_track(self, track):
        self._indent()
        self.xmlgen.startElement('track', track)
        self._level += 1

    def open_image(self, image):
        self._indent()
        self.xmlgen.startElement('image', image)
        self._level += 1

    def open_box(self, box):
        self._indent()
        self.xmlgen.startElement('box', box)
        self._level += 1

    def open_polygon(self, polygon):
        self._indent()
        self.xmlgen.startElement('polygon', polygon)
        self._level += 1

    def open_polyline(self, polyline):
        self._indent()
        self.xmlgen.startElement('polyline', polyline)
        self._level += 1

    def open_points(self, points):
        self._indent()
        self.xmlgen.startElement('points', points)
        self._level += 1

    def open_tag(self, tag):
        self._indent()
        self.xmlgen.startElement("tag", tag)
        self._level += 1

    def add_attribute(self, attribute):
        self._indent()
        self.xmlgen.startElement('attribute', {'name': attribute['name']})
        self.xmlgen.characters(attribute['value'])
        self.xmlgen.endElement('attribute')

    def _close_element(self, element):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement(element)

    def close_box(self):
        self._close_element('box')

    def close_polygon(self):
        self._close_element('polygon')

    def close_polyline(self):
        self._close_element('polyline')

    def close_points(self):
        self._close_element('points')

    def close_tag(self):
        self._close_element('tag')

    def close_image(self):
        self._close_element('image')

    def close_track(self):
        self._close_element('track')

    def close_root(self):
        self._close_element('annotations')
        self.xmlgen.endDocument()

class _SubsetWriter:
    def __init__(self, file, name, extractor, context):
        self._writer = XmlAnnotationWriter(file)
        self._name = name
        self._extractor = extractor
        self._context = context
        self._item_count = 0

    def is_empty(self):
        return self._item_count == 0

    def write(self):
        self._writer.open_root()
        self._write_meta()

        for index, item in enumerate(self._extractor):
            self._write_item(item, index)

        self._writer.close_root()

    def _write_item(self, item, index):
        if not self._context._reindex:
            index = cast(item.attributes.get('frame'), int, index)
        image_info = OrderedDict([ ("id", str(index)), ])
        filename = self._context._make_image_filename(item)
        image_info["name"] = filename
        if item.has_image:
            size = item.image.size
            if size:
                h, w = size
                image_info["width"] = str(w)
                image_info["height"] = str(h)

            if self._context._save_images:
                self._context._save_image(item,
                    osp.join(self._context._images_dir, filename))
        else:
            log.debug("Item '%s' has no image info", item.id)
        self._writer.open_image(image_info)

        for ann in item.annotations:
            if ann.type in {AnnotationType.points, AnnotationType.polyline,
                    AnnotationType.polygon, AnnotationType.bbox}:
                self._write_shape(ann, item)
            elif ann.type == AnnotationType.label:
                self._write_tag(ann, item)
            else:
                continue

        self._writer.close_image()

        self._item_count += 1

    def _write_meta(self):
        label_cat = self._extractor.categories().get(
            AnnotationType.label, LabelCategories())
        meta = OrderedDict([
            ("task", OrderedDict([
                ("id", ""),
                ("name", self._name),
                ("size", str(len(self._extractor))),
                ("mode", "annotation"),
                ("overlap", ""),
                ("start_frame", "0"),
                ("stop_frame", str(len(self._extractor))),
                ("frame_filter", ""),
                ("z_order", "True"),

                ("labels", [
                    ("label", OrderedDict([
                        ("name", label.name),
                        ("attributes", [
                            ("attribute", OrderedDict([
                                ("name", attr),
                                ("mutable", "True"),
                                ("input_type", "text"),
                                ("default_value", ""),
                                ("values", ""),
                            ])) for attr in self._get_label_attrs(label)
                        ])
                    ])) for label in label_cat.items
                ]),
            ])),
        ])
        self._writer.write_meta(meta)

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

    def _write_shape(self, shape, item):
        if shape.label is None:
            log.warning("Item %s: skipping a %s with no label",
                item.id, shape.type.name)
            return

        label_name = self._get_label(shape.label).name
        shape_data = OrderedDict([
            ("label", label_name),
            ("occluded", str(int(shape.attributes.get('occluded', False)))),
        ])

        if shape.type == AnnotationType.bbox:
            shape_data.update(OrderedDict([
                ("xtl", "{:.2f}".format(shape.points[0])),
                ("ytl", "{:.2f}".format(shape.points[1])),
                ("xbr", "{:.2f}".format(shape.points[2])),
                ("ybr", "{:.2f}".format(shape.points[3]))
            ]))
        else:
            shape_data.update(OrderedDict([
                ("points", ';'.join((
                    ','.join((
                        "{:.2f}".format(x),
                        "{:.2f}".format(y)
                    )) for x, y in pairs(shape.points))
                )),
            ]))

        shape_data['z_order'] = str(int(shape.z_order))
        if shape.group:
            shape_data['group_id'] = str(shape.group)

        if shape.type == AnnotationType.bbox:
            self._writer.open_box(shape_data)
        elif shape.type == AnnotationType.polygon:
            self._writer.open_polygon(shape_data)
        elif shape.type == AnnotationType.polyline:
            self._writer.open_polyline(shape_data)
        elif shape.type == AnnotationType.points:
            self._writer.open_points(shape_data)
        else:
            raise NotImplementedError("unknown shape type")

        for attr_name, attr_value in shape.attributes.items():
            if attr_name in self._context._builtin_attrs:
                continue
            if isinstance(attr_value, bool):
                attr_value = 'true' if attr_value else 'false'
            if self._context._allow_undeclared_attrs or \
                    attr_name in self._get_label_attrs(shape.label):
                self._writer.add_attribute(OrderedDict([
                    ("name", str(attr_name)),
                    ("value", str(attr_value)),
                ]))
            else:
                log.warning("Item %s: skipping undeclared "
                    "attribute '%s' for label '%s' "
                    "(allow with --allow-undeclared-attrs option)",
                    item.id, attr_name, label_name)

        if shape.type == AnnotationType.bbox:
            self._writer.close_box()
        elif shape.type == AnnotationType.polygon:
            self._writer.close_polygon()
        elif shape.type == AnnotationType.polyline:
            self._writer.close_polyline()
        elif shape.type == AnnotationType.points:
            self._writer.close_points()
        else:
            raise NotImplementedError("unknown shape type")

    def _write_tag(self, label, item):
        if label.label is None:
            log.warning("Item %s: skipping a %s with no label",
                item.id, label.type.name)
            return

        label_name = self._get_label(label.label).name
        tag_data = OrderedDict([
            ('label', label_name),
        ])
        if label.group:
            tag_data['group_id'] = str(label.group)
        self._writer.open_tag(tag_data)

        for attr_name, attr_value in label.attributes.items():
            if attr_name in self._context._builtin_attrs:
                continue
            if isinstance(attr_value, bool):
                attr_value = 'true' if attr_value else 'false'
            if self._context._allow_undeclared_attrs or \
                    attr_name in self._get_label_attrs(label.label):
                self._writer.add_attribute(OrderedDict([
                    ("name", str(attr_name)),
                    ("value", str(attr_value)),
                ]))
            else:
                log.warning("Item %s: skipping undeclared "
                    "attribute '%s' for label '%s' "
                    "(allow with --allow-undeclared-attrs option)",
                    item.id, attr_name, label_name)

        self._writer.close_tag()

class CvatConverter(Converter):
    DEFAULT_IMAGE_EXT = CvatPath.IMAGE_EXT

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
        self._builtin_attrs = CvatPath.BUILTIN_ATTRS
        self._allow_undeclared_attrs = allow_undeclared_attrs

    def apply(self):
        self._images_dir = osp.join(self._save_dir, CvatPath.IMAGES_DIR)
        os.makedirs(self._images_dir, exist_ok=True)

        for subset_name, subset in self._extractor.subsets().items():
            ann_path = osp.join(self._save_dir, '%s.xml' % subset_name)
            with open(ann_path, 'w', encoding='utf-8') as f:
                writer = _SubsetWriter(f, subset_name, subset, self)
                writer.write()

            if self._patch and subset_name in self._patch.updated_subsets and \
                    writer.is_empty():
                os.remove(ann_path)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            conv = cls(dataset.get_subset(subset), save_dir=save_dir, **kwargs)
            conv._patch = patch
            conv.apply()

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        # Find images that needs to be removed
        # images from different subsets are stored in the common directory
        # Avoid situations like:
        # (a, test): added
        # (a, train): removed
        # where the second line removes images from the first.
        ids_to_remove = {}
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.has_image):
                ids_to_remove[item_id] = (item, False)
            else:
                ids_to_remove.setdefault(item_id, (item, True))

        for item, to_remove in ids_to_remove.values():
            if not to_remove:
                continue

            image_path = osp.join(save_dir, CvatPath.IMAGES_DIR,
                conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.unlink(image_path)
