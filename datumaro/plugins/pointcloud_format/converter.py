import logging as log
import os
import os.path as osp
import random
import string
from pathlib import Path
from collections import OrderedDict

import json
import uuid

from datumaro.util.image import save_image, ByteImage

from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.extractor import (AnnotationType, DatasetItem, LabelCategories, Owner)
from datumaro.util import cast, pairs

from .format import PointCloudPath

class PointCloudParser:
    _SUPPORTED_SHAPES = 'cuboid'
    _REQUIRED_FILES = ('KEY_ID', 'FRAME', 'META')

    def __init__(self, subset, context):

        self._annotation = subset

        self._object_keys = {}
        self._figure_keys = {}
        self._video_keys = {}
        self._context = context
        self._user = {}
        self._label_objects = []

        key_id_data = {
            "tags": {},
            "objects": {},
            "figures": {},
            "videos": {}
        }
        self._key_id_data = key_id_data

        meta_data = {
            "classes": [],
            "tags": [],
            "projectType": "point_clouds"
        }
        self._meta_data = meta_data

        self._image_json = {
            "name": "",
            "meta": {
                "sensorsData": {
                    "extrinsicMatrix": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "intrinsicMatrix": [0, 0, 0, 0, 0, 0, 0, 0, 0]
                }
            }
        }

        self._frame_data = {}

        self.generate_user()
        self.generate_objects()
        self.generate_frames()

    def set_objects_key(self, id):
        self._object_keys[id] = str(uuid.uuid4())
        self._key_id_data["objects"].update({self._object_keys[id]: id})

    def set_figures_key(self, id):
        self._figure_keys[id] = str(uuid.uuid4())
        self._key_id_data["figures"].update({self._figure_keys[id]: id})

    def set_videos_key(self, id):
        self._video_keys[id] = str(uuid.uuid4())
        self._key_id_data["videos"].update({self._video_keys[id]: id})

    def get_object_key(self, id):
        return self._object_keys.get(id, None)

    def get_figure_key(self, id):
        return self._figure_keys.get(id, None)

    def get_video_key(self, id):

        return self._video_keys.get(id, None)

    def generate_user(self):
        for data in self._annotation:
            for item in data.annotations:
                if not self._user:
                    if item.type == AnnotationType.owner:
                        self._user["name"] = item.name
                        self._user["createdAt"] = item.createdAt
                        self._user["updatedAt"] = item.updatedAt
                        break

    def generate_objects(self):
        # label_cat = self._annotation.categories().get(AnnotationType.label, LabelCategories())

        for data in self._annotation:
            for item in data.annotations:
                if item.type == AnnotationType.label:
                    classes = {
                        "id": item.id,
                        "title": item.name,
                        "color": item.color,
                        "shape": "cuboid_3d",
                        "geometry_config": {},
                        "hotkey": ""
                    }
                    self.set_objects_key(item.id)

                    label_object = {
                        "key": self.get_object_key(item.id),
                        "classTitle": item.name,
                        "tags": [],
                        "labelerLogin": self._user["name"],
                        "createdAt": str(self._user["createdAt"]),
                        "updatedAt": str(self._user["updatedAt"])
                    }
                    self._meta_data["classes"].append(classes)
                    self._label_objects.append(label_object)

    def generate_frames(self):
        for i, data in enumerate(self._annotation):
            frame_data = []
            if data.pcd:
                self._write_item(data, data.id)
                if not self.get_video_key(int(data.attributes['frame'])):
                    self.set_videos_key(int(data.attributes["frame"]))

            for item in data.annotations:

                if item.type == AnnotationType.cuboid:

                    self.set_figures_key(item.id)
                    figures = {
                        "key": self.get_figure_key(item.id),
                        "objectKey": self.get_object_key(item.attributes["label_id"]),
                        'geometryType': "cuboid_3d",
                        "geometry": {
                            "position": {
                                "x": item.points[0],
                                "y": item.points[1],
                                "z": item.points[2]
                            },
                            "rotation": {
                                "x": item.points[3],
                                "y": item.points[4],
                                "z": item.points[5]
                            },
                            "dimensions": {
                                "x": item.points[6],
                                "y": item.points[7],
                                "z": item.points[8]
                            }
                        },
                        "labelerLogin": self._user["name"],
                        "createdAt": str(self._user["createdAt"]),
                        "updatedAt": str(self._user["updatedAt"])
                    }
                    frame_data.append(figures)
            if frame_data:
                self._frame_data[int(data.attributes["frame"])] = frame_data

    def _write_item(self, item, index):
        if not self._context._reindex:
            index = cast(item.attributes.get('frame'), int, index)
        image_info = OrderedDict([("id", str(index)), ])
        if isinstance(item.pcd, str) and osp.isfile(item.pcd):
            path = item.pcd.replace(os.sep, '/')
            filename= path.rsplit("/", maxsplit=1)[-1]
        else:
            filename = self._context._make_pcd_filename(item)
        image_info["name"] = filename
        PointCloudPath.WRITE_FILES.append({index: filename})

        if item.has_pcd:
            if self._context._save_images:
                self._context._image_dir = osp.join(self._context._default_dir, "pointcloud")

                related_dir = osp.join(self._context._default_dir, "related_images")
                related_dir = osp.join(related_dir, f"{filename.rsplit('.', maxsplit=1)[0]}_pcd" )

                self._context._save_pcd(item,
                                          osp.join(self._context._image_dir, filename))

                for i, rimage in enumerate(item.related_images):

                    try:
                        name = rimage["name"]
                    except AttributeError:
                        name = "".join(random.choice(string.ascii_lowercase) for i in range(6))
                        name = f"{name}.jpg"

                    path = osp.join(related_dir, name)

                    path = osp.abspath(path)
                    os.makedirs(osp.dirname(path), exist_ok=True)

                    if isinstance(rimage["image"], ByteImage):
                        with open(path, 'wb') as f:
                            f.write(item.get_bytes())
                    else:
                        save_image(path, rimage["image"].data)

                    path = osp.join(related_dir, f"{name}.json")
                    self._image_json["name"] = name
                    with open(path, "w") as f:
                        json.dump(self._image_json, f, indent=4)
        else:
            log.debug("Item '%s' has no image info", item.id)

    def write_key_id_data(self, f):
        json.dump(self._key_id_data, f, indent=4)

    def write_meta_data(self, f):
        json.dump(self._meta_data, f, indent=4)

    def write_frame_data(self, f, key):
        frame = {
            "description": "",
            "key": self.get_video_key(int(key)),
            "tags": [],
            "objects": self._label_objects,
            "figures": {}
        }

        if self._frame_data.get(key):
            frame["figures"] = self._frame_data[key]

        json.dump(frame, f, indent=4)


class PointCloudConverter(Converter):
    DEFAULT_IMAGE_EXT = PointCloudPath.IMAGE_EXT

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
        self._builtin_attrs = PointCloudPath.BUILTIN_ATTRS
        self._allow_undeclared_attrs = allow_undeclared_attrs

    def apply(self):
        self._default_dir = osp.join(self._save_dir, PointCloudPath.DEFAULT_DIR)
        os.makedirs(self._default_dir, exist_ok=True)
        self._annotation_dir = osp.join(self._default_dir, PointCloudPath.ANNNOTATION_DIR)
        os.makedirs(self._annotation_dir, exist_ok=True)

        for subset_name, subset in self._extractor.subsets().items():
            pointcloud = PointCloudParser(subset, self)


            for file_name in PointCloudPath.WRITE_FILES:

                write_dir = self._save_dir
                if isinstance(file_name, dict):
                    key = list(file_name.keys())[0]
                    file_name = f"{list(file_name.values())[0]}.json"
                    write_dir = self._annotation_dir

                with open(osp.join(write_dir, file_name), "w") as f:
                    if file_name == "key_id_map.json":
                        pointcloud.write_key_id_data(f)
                    elif file_name == "meta.json":
                        pointcloud.write_meta_data(f)
                    else:
                        pointcloud.write_frame_data(f, key)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            cls.convert(dataset.get_subset(subset), save_dir=save_dir, **kwargs)

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        images_dir = osp.join(save_dir, PointCloudPath.IMAGES_DIR)
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
