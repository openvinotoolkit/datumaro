# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

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
from datumaro.components.errors import DatasetImportError
from datumaro.components.media import Image, MediaElement, PointCloud
from datumaro.util import parse_json_file
from datumaro.version import VERSION

from .format import DATUMARO_FORMAT_VERSION, DatumaroPath


class DatumaroBase(SubsetBase):
    LEGACY_VERSION = "legacy"
    CURRENT_DATUMARO_FORMAT_VERSION = DATUMARO_FORMAT_VERSION

    # If Datumaro format version goes up, it will be
    # ALLOWED_VERSIONS = {LEGACY_VERSION, 1.0, ..., CURRENT_DATUMARO_FORMAT_VERSION}
    ALLOWED_VERSIONS = {LEGACY_VERSION, CURRENT_DATUMARO_FORMAT_VERSION}

    def __init__(self, path):
        assert osp.isfile(path), path

        dm_version = self._get_dm_format_version(path)

        # when backward compatibility happen, we should implement version specific readers
        if dm_version not in self.ALLOWED_VERSIONS:
            raise DatasetImportError(
                f"Datumaro format version of the given dataset is {dm_version}, "
                f"but not supported by this Datumaro version: {VERSION}. "
                f"The allowed datumaro format versions are {self.ALLOWED_VERSIONS}. "
                "Please install the latest Datumaro."
            )

        self.default_reader(path=path)

    def default_reader(self, path: str):
        """
        Default Datumaro reader for the latest version
        """
        rootpath = ""
        if path.endswith(osp.join(DatumaroPath.ANNOTATIONS_DIR, osp.basename(path))):
            rootpath = path.rsplit(DatumaroPath.ANNOTATIONS_DIR, maxsplit=1)[0]

        images_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.IMAGES_DIR)):
            images_dir = osp.join(rootpath, DatumaroPath.IMAGES_DIR)
        self._images_dir = images_dir

        pcd_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.PCD_DIR)):
            pcd_dir = osp.join(rootpath, DatumaroPath.PCD_DIR)
        self._pcd_dir = pcd_dir

        related_images_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.RELATED_IMAGES_DIR)):
            related_images_dir = osp.join(rootpath, DatumaroPath.RELATED_IMAGES_DIR)
        self._related_images_dir = related_images_dir

        super().__init__(subset=osp.splitext(osp.basename(path))[0])
        self._load_impl(path)

    def _get_dm_format_version(self, path: str):
        """
        Get Datumaro format at exporting the dataset

        Note that the regacy Datumaro doesn't store the version into exported dataset.
        Thus it returns DatumaroBase.REGACY_VERSION
        """
        self._parsed_anns = parse_json_file(path)
        return self._parsed_anns.get("dm_format_version", self.LEGACY_VERSION)

    def _load_impl(self, path: str) -> None:
        """Actual implementation of loading Datumaro format."""
        self._infos = self._load_infos(self._parsed_anns)
        self._categories = self._load_categories(self._parsed_anns)
        self._items = self._load_items(self._parsed_anns)

    @staticmethod
    def _load_infos(parsed):
        return parsed.get("infos", {})

    @staticmethod
    def _load_categories(parsed):
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

    def _load_items(self, parsed):
        items = []
        for item_desc in parsed["items"]:
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

                media = Image(path=image_path, size=image_info.get("size"))
                self._media_type = Image

            pcd_info = item_desc.get("point_cloud")
            if media and pcd_info:
                raise DatasetImportError("Dataset cannot contain multiple media types")
            if pcd_info:
                pcd_path = pcd_info.get("path")
                point_cloud = osp.join(self._pcd_dir, self._subset, pcd_path)

                related_images = None
                ri_info = item_desc.get("related_images")
                if ri_info:
                    related_images = [
                        Image(size=ri.get("size"), path=ri.get("path")) for ri in ri_info
                    ]

                media = PointCloud(point_cloud, extra_images=related_images)
                self._media_type = PointCloud

            media_desc = item_desc.get("media")
            if not media and media_desc and media_desc.get("path"):
                media = MediaElement(path=media_desc.get("path"))
                self._media_type = MediaElement

            annotations = self._load_annotations(item_desc)

            item = DatasetItem(
                id=item_id,
                subset=self._subset,
                annotations=annotations,
                media=media,
                attributes=item_desc.get("attr"),
            )

            items.append(item)

        return items

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

            else:
                raise NotImplementedError()

        return loaded
