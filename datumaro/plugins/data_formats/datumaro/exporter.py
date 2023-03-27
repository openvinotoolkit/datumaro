# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=no-self-use

import os
import os.path as osp
import shutil
from contextlib import contextmanager
from multiprocessing.pool import Pool
from typing import Optional

import numpy as np
import pycocotools.mask as mask_utils

from datumaro.components.annotation import (
    Annotation,
    Bbox,
    Caption,
    Cuboid3d,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RleMask,
    _Shape,
)
from datumaro.components.crypter import NULL_CRYPTER
from datumaro.components.dataset import ItemStatus
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetItem
from datumaro.components.exporter import ExportContextComponent, Exporter
from datumaro.components.media import Image, MediaElement, PointCloud
from datumaro.util import cast, dump_json_file

from .format import DATUMARO_FORMAT_VERSION, DatumaroPath


class _SubsetWriter:
    def __init__(self, context: Exporter, ann_file: str, export_context: ExportContextComponent):
        self._context = context

        self._data = {
            "dm_format_version": DATUMARO_FORMAT_VERSION,
            "infos": {},
            "categories": {},
            "items": [],
        }

        self.ann_file = ann_file
        self.export_context = export_context

    @property
    def infos(self):
        return self._data["infos"]

    @property
    def categories(self):
        return self._data["categories"]

    @property
    def items(self):
        return self._data["items"]

    def is_empty(self):
        return not self.items

    @staticmethod
    @contextmanager
    def context_save_media(
        item: DatasetItem, context: ExportContextComponent, encryption: bool = False
    ) -> None:
        """Implicitly change the media path and save it if save_media=True.
        When done, revert it's path as before.

        Parameters
        ----------
            item: Dataset item to save its media
            context: Context instance to help the media export
            encryption: If false, prevent the media from being encrypted
        """
        if item.media is None:
            yield
        elif isinstance(item.media, Image):
            image = item.media_as(Image)
            path = image.path

            if context.save_media:
                # Temporarily update image path and save it.
                fname = context.make_image_filename(item)
                context.save_image(item, encryption=encryption, fname=fname, subdir=item.subset)
                image._path = fname

            yield
            image._path = path
        elif isinstance(item.media, PointCloud):
            pcd = item.media_as(PointCloud)
            path = pcd.path

            if context.save_media:
                # Temporarily update pcd path and save it.
                fname = context.make_pcd_filename(item)
                context.save_point_cloud(item, fname=fname, subdir=item.subset)
                pcd._path = fname

                # Temporarily update pcd related images paths and save them.
                for i, img in enumerate(sorted(pcd.extra_images, key=lambda v: v.path)):
                    img.__path = img.path
                    fname = osp.join(item.id, f"image_{i}{context.find_image_ext(img)}")
                    img._path = fname

                    if img.has_data:
                        context.save_image(
                            item,
                            basedir=context.related_images_dir,
                            subdir=item.subset,
                            fname=fname,
                        )

            yield
            pcd._path = path
            if context.save_media:
                for img in pcd.extra_images:
                    img._path = img.__path
                    del img.__path
        else:
            raise NotImplementedError

    def add_item(self, item: DatasetItem, *args, **kwargs):
        annotations = []
        item_desc = {
            "id": item.id,
            "annotations": annotations,
        }

        if item.attributes:
            item_desc["attr"] = item.attributes

        with self.context_save_media(item, self.export_context):
            if isinstance(item.media, Image):
                image = item.media_as(Image)
                item_desc["image"] = {
                    "path": image.path,
                }
                if item.media.has_size:  # avoid occasional loading
                    item_desc["image"]["size"] = image.size
            elif isinstance(item.media, PointCloud):
                pcd = item.media_as(PointCloud)

                item_desc["point_cloud"] = {"path": pcd.path}

                related_images = [
                    {"path": img.path, "size": img.size} if img.has_size else {"path": img.path}
                    for img in pcd.extra_images
                ]

                if related_images:
                    item_desc["related_images"] = related_images
            elif isinstance(item.media, MediaElement):
                item_desc["media"] = {"path": item.media.path}

        self.items.append(item_desc)

        for ann in item.annotations:
            if isinstance(ann, Label):
                converted_ann = self._convert_label_object(ann)
            elif isinstance(ann, Mask):
                converted_ann = self._convert_mask_object(ann)
            elif isinstance(ann, Points):
                converted_ann = self._convert_points_object(ann)
            elif isinstance(ann, PolyLine):
                converted_ann = self._convert_polyline_object(ann)
            elif isinstance(ann, Polygon):
                converted_ann = self._convert_polygon_object(ann)
            elif isinstance(ann, Bbox):
                converted_ann = self._convert_bbox_object(ann)
            elif isinstance(ann, Caption):
                converted_ann = self._convert_caption_object(ann)
            elif isinstance(ann, Cuboid3d):
                converted_ann = self._convert_cuboid_3d_object(ann)
            elif isinstance(ann, Ellipse):
                converted_ann = self._convert_ellipse_object(ann)
            else:
                raise NotImplementedError()
            annotations.append(converted_ann)

    def add_infos(self, infos):
        self._data["infos"].update(infos)

    def add_categories(self, categories):
        for ann_type, desc in categories.items():
            if isinstance(desc, LabelCategories):
                converted_desc = self._convert_label_categories(desc)
            elif isinstance(desc, MaskCategories):
                converted_desc = self._convert_mask_categories(desc)
            elif isinstance(desc, PointsCategories):
                converted_desc = self._convert_points_categories(desc)
            else:
                raise NotImplementedError()
            self.categories[ann_type.name] = converted_desc

    def write(self, *args, **kwargs):
        dump_json_file(self.ann_file, self._data)

    def _convert_annotation(self, obj):
        assert isinstance(obj, Annotation)

        ann_json = {
            "id": cast(obj.id, int),
            "type": cast(obj.type.name, str),
            "attributes": obj.attributes,
            "group": cast(obj.group, int, 0),
        }
        return ann_json

    def _convert_label_object(self, obj):
        converted = self._convert_annotation(obj)

        converted.update(
            {
                "label_id": cast(obj.label, int),
            }
        )
        return converted

    def _convert_mask_object(self, obj):
        converted = self._convert_annotation(obj)

        if isinstance(obj, RleMask):
            rle = obj.rle
        else:
            rle = mask_utils.encode(np.require(obj.image, dtype=np.uint8, requirements="F"))

        if isinstance(rle["counts"], str):
            counts = rle["counts"]
        else:
            counts = rle["counts"].decode("ascii")

        converted.update(
            {
                "label_id": cast(obj.label, int),
                "rle": {
                    # serialize as compressed COCO mask
                    "counts": counts,
                    "size": list(int(c) for c in rle["size"]),
                },
                "z_order": obj.z_order,
            }
        )
        return converted

    def _convert_shape_object(self, obj):
        assert isinstance(obj, _Shape)
        converted = self._convert_annotation(obj)

        converted.update(
            {
                "label_id": cast(obj.label, int),
                "points": [float(p) for p in obj.points],
                "z_order": obj.z_order,
            }
        )
        return converted

    def _convert_polyline_object(self, obj):
        return self._convert_shape_object(obj)

    def _convert_polygon_object(self, obj):
        return self._convert_shape_object(obj)

    def _convert_bbox_object(self, obj):
        converted = self._convert_shape_object(obj)
        converted.pop("points", None)
        converted["bbox"] = [float(p) for p in obj.get_bbox()]
        return converted

    def _convert_points_object(self, obj):
        converted = self._convert_shape_object(obj)

        converted.update(
            {
                "visibility": [int(v.value) for v in obj.visibility],
            }
        )
        return converted

    def _convert_caption_object(self, obj):
        converted = self._convert_annotation(obj)

        converted.update(
            {
                "caption": cast(obj.caption, str),
            }
        )
        return converted

    def _convert_cuboid_3d_object(self, obj):
        converted = self._convert_annotation(obj)
        converted.update(
            {
                "label_id": cast(obj.label, int),
                "position": [float(p) for p in obj.position],
                "rotation": [float(p) for p in obj.rotation],
                "scale": [float(p) for p in obj.scale],
            }
        )
        return converted

    def _convert_ellipse_object(self, obj: Ellipse):
        return self._convert_shape_object(obj)

    def _convert_attribute_categories(self, attributes):
        return sorted(attributes)

    def _convert_labels_label_groups(self, labels):
        return sorted(labels)

    def _convert_label_categories(self, obj):
        converted = {
            "labels": [],
            "label_groups": [],
            "attributes": self._convert_attribute_categories(obj.attributes),
        }
        for label in obj.items:
            converted["labels"].append(
                {
                    "name": cast(label.name, str),
                    "parent": cast(label.parent, str),
                    "attributes": self._convert_attribute_categories(label.attributes),
                }
            )
        for label_group in obj.label_groups:
            converted["label_groups"].append(
                {
                    "name": cast(label_group.name, str),
                    "group_type": cast(label_group.group_type, str),
                    "labels": self._convert_labels_label_groups(label_group.labels),
                }
            )
        return converted

    def _convert_mask_categories(self, obj):
        converted = {
            "colormap": [],
        }
        for label_id, color in obj.colormap.items():
            converted["colormap"].append(
                {
                    "label_id": int(label_id),
                    "r": int(color[0]),
                    "g": int(color[1]),
                    "b": int(color[2]),
                }
            )
        return converted

    def _convert_points_categories(self, obj):
        converted = {
            "items": [],
        }
        for label_id, item in obj.items.items():
            converted["items"].append(
                {
                    "label_id": int(label_id),
                    "labels": [cast(label, str) for label in item.labels],
                    "joints": [list(map(int, j)) for j in item.joints],
                }
            )
        return converted


class DatumaroExporter(Exporter):
    DEFAULT_IMAGE_EXT = DatumaroPath.IMAGE_EXT
    PATH_CLS = DatumaroPath

    def create_writer(
        self, subset: str, images_dir: str, pcd_dir: str, related_images_dir: str
    ) -> _SubsetWriter:
        export_context = ExportContextComponent(
            save_dir=self._save_dir,
            save_media=self._save_media,
            images_dir=images_dir,
            pcd_dir=pcd_dir,
            related_images_dir=related_images_dir,
            crypter=NULL_CRYPTER,
            image_ext=self._image_ext,
            default_image_ext=self._default_image_ext,
        )

        return _SubsetWriter(
            context=self,
            ann_file=osp.join(self._annotations_dir, subset + self.PATH_CLS.ANNOTATION_EXT),
            export_context=export_context,
        )

    def apply(self, pool: Optional[Pool] = None, *args, **kwargs):
        os.makedirs(self._save_dir, exist_ok=True)

        images_dir = osp.join(self._save_dir, self.PATH_CLS.IMAGES_DIR)
        os.makedirs(images_dir, exist_ok=True)
        self._images_dir = images_dir

        annotations_dir = osp.join(self._save_dir, self.PATH_CLS.ANNOTATIONS_DIR)
        os.makedirs(annotations_dir, exist_ok=True)
        self._annotations_dir = annotations_dir

        self._pcd_dir = osp.join(self._save_dir, self.PATH_CLS.PCD_DIR)
        self._related_images_dir = osp.join(self._save_dir, self.PATH_CLS.RELATED_IMAGES_DIR)

        writers = {
            subset: self.create_writer(
                subset, self._images_dir, self._pcd_dir, self._related_images_dir
            )
            for subset in self._extractor.subsets()
        }

        for writer in writers.values():
            writer.add_infos(self._extractor.infos())
            writer.add_categories(self._extractor.categories())

        for item in self._extractor:
            subset = item.subset or DEFAULT_SUBSET_NAME
            writers[subset].add_item(item, pool)

        for subset, writer in writers.items():
            if self._patch and subset in self._patch.updated_subsets and writer.is_empty():
                if osp.isfile(writer.ann_file):
                    # Remove subsets that became empty
                    os.remove(writer.ann_file)
                continue

            writer.write(pool)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            conv = cls(dataset.get_subset(subset), save_dir=save_dir, **kwargs)
            conv._patch = patch
            conv.apply()

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.media):
                continue

            image_path = osp.join(
                save_dir, cls.PATH_CLS.IMAGES_DIR, item.subset, conv._make_image_filename(item)
            )
            if osp.isfile(image_path):
                os.unlink(image_path)

            pcd_path = osp.join(
                save_dir, cls.PATH_CLS.PCD_DIR, item.subset, conv._make_pcd_filename(item)
            )
            if osp.isfile(pcd_path):
                os.unlink(pcd_path)

            related_images_path = osp.join(
                save_dir, cls.PATH_CLS.RELATED_IMAGES_DIR, item.subset, item.id
            )
            if osp.isdir(related_images_path):
                shutil.rmtree(related_images_path)
