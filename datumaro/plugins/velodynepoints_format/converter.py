import logging as log
import os
import os.path as osp
from collections import OrderedDict
from itertools import chain
from xml.sax.saxutils import XMLGenerator
import shutil

from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.extractor import (AnnotationType, DatasetItem,
    LabelCategories)
from datumaro.util import cast, pairs

# pose states
from datumaro.util.image import ByteImage, save_image

POSES_STATES = {"UNSET": 0, "INTERP" : 1,  "LABELED" : 2}

#occlusion state
OCCLUSION_STATE = { "OCCLUSION_UNSET": -1, "VISIBLE": 0, "PARTLY": 1, "FULLY": 2}

#truncation states
TRUNCATION_STATE = {"TRUNCATION_UNSET" : -1, "IN_IMAGE" : 0, "TRUNCATED": 1, "OUT_IMAGE" :2, "BEHIND_IMAGE" :     99}


class XmlAnnotationWriter:

    def __init__(self, file, tracklets):
        self.version = "1.1"
        self._file = file
        # self._annotation = tracklets
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

    def _get_label_attrs(self, label):
        label_cat = self._extractor.categories().get(
            AnnotationType.label, LabelCategories())
        if isinstance(label, int):
            label = label_cat[label]
        return set(chain(label.attributes, label_cat.attributes)) - \
            self._context._builtin_attrs

    def create_tracklets(self, subset):

        for i, data in enumerate(subset):

            for index, item in enumerate(data.annotations):
                if item.type == AnnotationType.cuboid:
                    if item.label is None:
                        log.warning("Item %s: skipping a %s with no label",
                                    item.id, item.type.name)

                    label_name = self._get_label(item.label).name

                    self._write_item(data, i)

                    tracklet = {
                        "objectType": label_name,
                        "h": item.points[0],
                        "w": item.points[1],
                        "l": item.points[2],
                        "first_frame": data.attributes.get('frame',""),
                        "poses": []
                    }
                    pose = {
                        "tx": item.points[3],
                        "ty": item.points[4],
                        "tz": item.points[5],
                        "rx": item.points[6],
                        "ry": item.points[7],
                        "rz": item.points[8],
                        "state": 2,  # pose state
                        "occlusion": -1,  # occusion state
                        "occlusion_kf": False,  # is this an occlusion keyframe
                        "truncation": -1,  # truncation state
                        "amt_occlusion": -1,  # Mechanical Turk occlusion label
                        "amt_border_l": -1,  # Mechanical Turk left boundary label (relative)
                        "amt_border_r": -1,  # Mechanical Turk right boundary label (relative)
                        "amt_occlusion_kf": -1,  # Mechanical Turk occlusion keyframe
                        "amt_border_kf": -1,  # Mechanical Turk border keyframe
                    }
                    tracklet["poses"].append(pose)
                    tracklet["finished"] = 1
                    self._tracklets.append(tracklet)

        tracklets = XmlAnnotationWriter(self._file, self._tracklets)
        tracklets.generate_tracklets()

    def _get_label(self, label_id):
        if label_id is None:
            return ""
        label_cat = self._extractor.categories().get(
            AnnotationType.label, LabelCategories())
        return label_cat.items[label_id]

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
                dir = osp.join(self._context._save_dir, "velodyne_points")
                self._context._image_dir = osp.join(dir, "data")

                self._context._save_pcd(item,
                                        osp.join(self._context._image_dir, filename))

                if item.related_images:
                    related_dir = osp.join(self._context._save_dir, f"image_0{index}")

                    for i, related_image in enumerate(item.related_images):
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
        self._builtin_attrs = []
        self._allow_undeclared_attrs = allow_undeclared_attrs

    def apply(self):

        for subset_name, subset in self._extractor.subsets().items():

            with open(osp.join(self._save_dir, 'tracklets.xml'), 'w') as f:
                _SubsetWriter(f, subset_name, subset, self)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            cls.convert(dataset.get_subset(subset), save_dir=save_dir, **kwargs)

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        images_dir = osp.join(save_dir, cls._images_dir)
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.has_image):
                continue

            image_path = osp.join(images_dir, conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.unlink(image_path)
