# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import Any, Dict, List, Optional, Type

import json_stream
from json_stream.base import StreamingJSONList, StreamingJSONObject

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Cuboid3d,
    Ellipse,
    Label,
    LabelCategories,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RleMask,
)
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import DatasetImportError, MediaTypeError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, MediaElement, MediaType, PointCloud
from datumaro.util import parse_json_file
from datumaro.version import __version__

from .format import DATUMARO_FORMAT_VERSION, DatumaroPath


class DefaultReader:
    def __init__(
        self, path: str, subset: str, rootpath: str, images_dir: str, pcd_dir: str
    ) -> None:
        self._subset = subset
        self._rootpath = rootpath
        self._images_dir = images_dir
        self._pcd_dir = pcd_dir

        self._reader = self._init_reader(path)
        self.media_type = self._load_media_type(self._reader)
        self.infos = self._load_infos(self._reader)
        self.categories = self._load_categories(self._reader)
        self.items = self._load_items(self._reader)

    def _init_reader(self, path: str):
        return parse_json_file(path)

    @staticmethod
    def _load_media_type(parsed) -> Type[MediaElement]:
        media_type = parsed.get("media_type", MediaType.IMAGE)
        return MediaType(media_type).media

    @staticmethod
    def _load_infos(parsed) -> Dict:
        return parsed.get("infos", {})

    @staticmethod
    def _load_categories(parsed) -> Dict:
        categories = {}

        parsed_label_cat = parsed["categories"].get(AnnotationType.label.name)
        if parsed_label_cat:
            label_categories = LabelCategories(attributes=parsed_label_cat.get("attributes", []))
            for item in parsed_label_cat["labels"]:
                label_categories.add(
                    item["name"],
                    parent=item["parent"],
                    attributes=item.get("attributes", []),
                )

            for item in parsed_label_cat.get("label_groups", []):
                label_categories.add_label_group(
                    item["name"], labels=item["labels"], group_type=item["group_type"]
                )

            categories[AnnotationType.label] = label_categories

        parsed_mask_cat = parsed["categories"].get(AnnotationType.mask.name)
        if parsed_mask_cat:
            colormap = {}
            for item in parsed_mask_cat["colormap"]:
                colormap[int(item["label_id"])] = (item["r"], item["g"], item["b"])

            mask_categories = MaskCategories(colormap=colormap)
            categories[AnnotationType.mask] = mask_categories

        parsed_points_cat = parsed["categories"].get(AnnotationType.points.name)
        if parsed_points_cat:
            point_categories = PointsCategories()
            for item in parsed_points_cat["items"]:
                point_categories.add(int(item["label_id"]), item["labels"], joints=item["joints"])

            categories[AnnotationType.points] = point_categories

        return categories

    def _load_items(self, parsed) -> List:
        items = []

        item_descs = parsed["items"]
        while item_descs:
            item_desc = item_descs.pop()
            item = self._parse_item(item_desc)
            items.append(item)

        return items

    def _parse_item(self, item_desc):
        item_id = item_desc["id"]

        media = None
        image_info = item_desc.get("image")
        if image_info:
            image_filename = image_info.get("path") or item_id + DatumaroPath.IMAGE_EXT
            image_path = osp.join(self._images_dir, self._subset, image_filename)
            if not osp.isfile(image_path):
                # backward compatibility
                old_image_path = osp.join(self._images_dir, image_filename)
                if osp.isfile(old_image_path):
                    image_path = old_image_path

            media = Image.from_file(path=image_path, size=image_info.get("size"))
            if self.media_type == MediaElement:
                self.media_type = Image

        pcd_info = item_desc.get("point_cloud")
        if media and pcd_info:
            raise MediaTypeError("Dataset cannot contain multiple media types")
        if pcd_info:
            pcd_path = pcd_info.get("path")
            point_cloud = osp.join(self._pcd_dir, self._subset, pcd_path)

            related_images = None
            ri_info = item_desc.get("related_images")
            if ri_info:
                related_images = [
                    Image.from_file(
                        size=ri.get("size"),
                        path=osp.join(self._images_dir, self._subset, ri.get("path")),
                    )
                    for ri in ri_info
                ]

            media = PointCloud.from_file(path=point_cloud, extra_images=related_images)
            if self.media_type == MediaElement:
                self.media_type = PointCloud

        media_desc = item_desc.get("media")
        if not media and media_desc and media_desc.get("path"):
            media = MediaElement(path=media_desc.get("path"))

        annotations = self._load_annotations(item_desc)

        item = DatasetItem(
            id=item_id,
            subset=self._subset,
            annotations=annotations,
            media=media,
            attributes=item_desc.get("attr"),
        )

        return item

    @staticmethod
    def _load_annotations(item):
        parsed = item["annotations"]
        loaded = []

        for ann in parsed:
            ann_id = ann.get("id")
            ann_type = AnnotationType[ann["type"]]
            attributes = ann.get("attributes")
            group = ann.get("group")

            label_id = ann.get("label_id")
            z_order = ann.get("z_order")
            points = ann.get("points")

            if ann_type == AnnotationType.label:
                loaded.append(Label(label=label_id, id=ann_id, attributes=attributes, group=group))

            elif ann_type == AnnotationType.mask:
                rle = ann["rle"]
                rle["counts"] = rle["counts"].encode("ascii")
                loaded.append(
                    RleMask(
                        rle=rle,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.polyline:
                loaded.append(
                    PolyLine(
                        points,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.polygon:
                loaded.append(
                    Polygon(
                        points,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.bbox:
                x, y, w, h = ann["bbox"]
                loaded.append(
                    Bbox(
                        x,
                        y,
                        w,
                        h,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.points:
                loaded.append(
                    Points(
                        points,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.caption:
                caption = ann.get("caption")
                loaded.append(Caption(caption, id=ann_id, attributes=attributes, group=group))

            elif ann_type == AnnotationType.cuboid_3d:
                loaded.append(
                    Cuboid3d(
                        ann.get("position"),
                        ann.get("rotation"),
                        ann.get("scale"),
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                    )
                )

            elif ann_type == AnnotationType.ellipse:
                loaded.append(
                    Ellipse(
                        *points,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.hash_key:
                continue
            else:
                raise NotImplementedError()

        return loaded

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        yield from self.items


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, StreamingJSONObject):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, StreamingJSONList):
        return [_to_dict(v) for v in obj]
    return obj


class StreamDefaultReader(DefaultReader):
    def __init__(
        self, path: str, subset: str, rootpath: str, images_dir: str, pcd_dir: str
    ) -> None:
        super().__init__(path, subset, rootpath, images_dir, pcd_dir)
        self._length = None

    def __len__(self):
        if self._length is None:
            self._length = sum(1 for _ in self)
        return self._length

    def __iter__(self):
        with open(self._reader) as fp:
            data = json_stream.load(fp)
            items = data.get("items", None)
            if items is None:
                raise DatasetImportError('Annotation JSON file should have "items" entity.')

            length = 0
            for item in items:
                item_desc = _to_dict(item)
                length += 1
                yield self._parse_item(item_desc)

            if self._length != length:
                self._length = length

    def _init_reader(self, path: str):
        return path

    @staticmethod
    def _load_media_type(path) -> Type[MediaElement]:
        with open(path, "r") as fp:
            data = json_stream.load(fp)
            media_type = data.get("media_type", MediaType.IMAGE)
            return MediaType(media_type).media

    @staticmethod
    def _load_infos(path) -> Dict:
        with open(path, "r") as fp:
            data = json_stream.load(fp)
            infos = data.get("infos", {})
            if isinstance(infos, StreamingJSONObject):
                infos = _to_dict(infos)

            return infos

    @staticmethod
    def _load_categories(path) -> Dict:
        with open(path, "r") as fp:
            data = json_stream.load(fp)
            categories = data.get("categories", {})
            if isinstance(categories, StreamingJSONObject):
                categories = _to_dict(categories)

            return DefaultReader._load_categories({"categories": categories})

    def _load_items(self, parsed) -> List:
        return []


class DatumaroBase(SubsetBase):
    LEGACY_VERSION = "legacy"
    CURRENT_DATUMARO_FORMAT_VERSION = DATUMARO_FORMAT_VERSION

    # If Datumaro format version goes up, it will be
    # ALLOWED_VERSIONS = {LEGACY_VERSION, 1.0, ..., CURRENT_DATUMARO_FORMAT_VERSION}
    ALLOWED_VERSIONS = {LEGACY_VERSION, CURRENT_DATUMARO_FORMAT_VERSION}

    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        stream: bool = True,
        ctx: Optional[ImportContext] = None,
    ):
        assert osp.isfile(path), path
        subset = osp.splitext(osp.basename(path))[0] if subset is None else subset
        self._stream = stream
        super().__init__(subset=subset, ctx=ctx)
        self._init_path(path)

        dm_version = self._get_dm_format_version(path)

        # when backward compatibility happen, we should implement version specific readers
        if dm_version not in self.ALLOWED_VERSIONS:
            raise DatasetImportError(
                f"Datumaro format version of the given dataset is {dm_version}, "
                f"but not supported by this Datumaro version: {__version__}. "
                f"The allowed datumaro format versions are {self.ALLOWED_VERSIONS}. "
                "Please install the latest Datumaro."
            )

        self._load_impl(path)

    def _init_path(self, path: str):
        """Initialize path variables"""
        rootpath = ""
        if path.endswith(osp.join(DatumaroPath.ANNOTATIONS_DIR, osp.basename(path))):
            rootpath = path.rsplit(DatumaroPath.ANNOTATIONS_DIR, maxsplit=1)[0]
        self._rootpath = rootpath

        images_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.IMAGES_DIR)):
            images_dir = osp.join(rootpath, DatumaroPath.IMAGES_DIR)
        self._images_dir = images_dir

        pcd_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.PCD_DIR)):
            pcd_dir = osp.join(rootpath, DatumaroPath.PCD_DIR)
        self._pcd_dir = pcd_dir

    @property
    def is_stream(self) -> bool:
        return self._stream

    def infos(self):
        return self._reader.infos

    def categories(self):
        return self._reader.categories

    def media_type(self):
        return self._reader.media_type

    def _load_impl(self, path: str) -> None:
        """Actual implementation of loading Datumaro format."""
        self._reader = (
            DefaultReader(path, self._subset, self._rootpath, self._images_dir, self._pcd_dir)
            if not self._stream
            else StreamDefaultReader(
                path, self._subset, self._rootpath, self._images_dir, self._pcd_dir
            )
        )
        return self._reader

    def _get_dm_format_version(self, path) -> str:
        """
        Get Datumaro format at exporting the dataset

        Note that the legacy Datumaro doesn't store the version into exported dataset.
        Thus it returns DatumaroBase.LEGACY_VERSION
        """
        with open(path, "r") as fp:
            version = json_stream.load(fp).get("dm_format_version", self.LEGACY_VERSION)
        return version

    def __len__(self) -> int:
        return len(self._reader)

    def __iter__(self) -> DatasetItem:
        yield from self._reader
