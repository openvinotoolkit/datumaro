# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from typing import List, Optional

from defusedxml import ElementTree as ET

from datumaro.components.annotation import AnnotationType, Cuboid3d, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import InvalidAnnotationError
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image, PointCloud
from datumaro.util import cast
from datumaro.util.image import find_images
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

from .format import KittiRawPath, OcclusionStates, TruncationStates


class Kitti3dBase(DatasetBase):
    def __init__(
        self,
        subset_name: str,
        path: str,
        items: OrderedDict[str, str],
        categories: CategoriesInfo,
        image_info: ImageMeta,
    ):
        super().__init__()
        self._subset_name = subset_name
        self._path = path
        self._items = items
        self._categories = categories
        self._image_info = image_info

    def __iter__(self) -> Iterator[DatasetItem]:
        for item_id in self._items:
            item = self._get(item_id)
            if item is not None:
                yield item

    def _get(self, item_id: str) -> Optional[DatasetItem]:
        item = self._items.get(item_id)

        if not isinstance(item, str):
            return None

        try:
            image_size = self._image_info.get(item_id)
            image = Image.from_file(path=osp.join(self._path, item), size=image_size)

            anno_path = osp.splitext(image.path)[0] + ".txt"
            annotations = self._parse_annotations(
                anno_path,
                image,
                label_categories=self._categories[AnnotationType.label],
            )

            item = DatasetItem(
                id=item_id, subset=self._subset_name, media=image, annotations=annotations
            )
            return item
        except (UndeclaredLabelError, InvalidAnnotationError) as e:
            self._ctx.error_policy.report_annotation_error(e, item_id=(item_id, self._subset_name))
        except Exception as e:
            self._ctx.error_policy.report_item_error(e, item_id=(item_id, self._subset_name))

        return None

    @classmethod
    def _parse_annotations(
        cls,
        anno_path: str,
        image: ImageFromFile,
        *,
        label_categories: LabelCategories,
    ) -> List[Annotation]:
        lines = []
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        annotations = []

        if lines:
            # Use image info as late as possible to avoid unnecessary image loading
            if image.size is None:
                raise DatasetImportError(f"Can't find image info for '{localize_path(image.path)}'")
            image_height, image_width = image.size

        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                raise InvalidAnnotationError(
                    f"Unexpected field count {len(parts)} in the bbox description. "
                    "Expected 5 fields (label, xc, yc, w, h)."
                )
            label_id, xc, yc, w, h = parts

            label_id = cls._parse_field(label_id, int, "bbox label id")
            if label_id not in label_categories:
                raise UndeclaredLabelError(str(label_id))

            w = cls._parse_field(w, float, "bbox width")
            h = cls._parse_field(h, float, "bbox height")
            x = cls._parse_field(xc, float, "bbox center x") - w * 0.5
            y = cls._parse_field(yc, float, "bbox center y") - h * 0.5
            annotations.append(
                Bbox(
                    x * image_width,
                    y * image_height,
                    w * image_width,
                    h * image_height,
                    label=label_id,
                    id=idx,
                    group=idx,
                )
            )

        return annotations

    @classmethod
    def _parse_field(cls, value: str, desired_type: Type[T], field_name: str) -> T:
        try:
            return desired_type(value)
        except Exception as e:
            raise InvalidAnnotationError(
                f"Can't parse {field_name} from '{value}'. Expected {desired_type}"
            ) from e


class KittiRawBase(SubsetBase):
    # http://www.cvlibs.net/datasets/kitti/raw_data.php
    # https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_raw_data.zip
    # Check cpp header implementation for field meaning

    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        assert osp.isfile(path), path
        self._rootdir = osp.dirname(path)

        super().__init__(subset=subset, media_type=PointCloud, ctx=ctx)

        items, categories = self._parse(path)
        self._categories = categories
        self._items = list(self._load_items(items).values())

    @classmethod
    def _parse(cls, path):
        tracks = []
        track = None
        shape = None
        attr = None
        labels = {}
        point_tags = {"tx", "ty", "tz", "rx", "ry", "rz"}

        # Can fail with "XML declaration not well-formed" on documents with
        # <?xml ... standalone="true"?>
        #                       ^^^^
        # (like the original Kitti dataset), while
        # <?xml ... standalone="yes"?>
        #                       ^^^
        # works.
        tree = ET.iterparse(path, events=("start", "end"))
        for ev, elem in tree:
            if ev == "start":
                if elem.tag == "item":
                    if track is None:
                        track = {
                            "shapes": [],
                            "scale": {},
                            "label": None,
                            "attributes": {},
                            "start_frame": None,
                            "length": None,
                        }
                    else:
                        shape = {
                            "points": {},
                            "attributes": {},
                            "occluded": None,
                            "occluded_kf": False,
                            "truncated": None,
                        }

                elif elem.tag == "attribute":
                    attr = {}

            elif ev == "end":
                if elem.tag == "item":
                    assert track is not None

                    if shape:
                        track["shapes"].append(shape)
                        shape = None
                    else:
                        assert track["length"] == len(track["shapes"])

                        if track["label"]:
                            labels.setdefault(track["label"], set())

                            for a in track["attributes"]:
                                labels[track["label"]].add(a)

                            for s in track["shapes"]:
                                for a in s["attributes"]:
                                    labels[track["label"]].add(a)

                        tracks.append(track)
                        track = None

                # track tags
                elif track and elem.tag == "objectType":
                    track["label"] = elem.text
                elif track and elem.tag in {"h", "w", "l"}:
                    track["scale"][elem.tag] = float(elem.text)
                elif track and elem.tag == "first_frame":
                    track["start_frame"] = int(elem.text)
                elif track and elem.tag == "count" and track:
                    track["length"] = int(elem.text)

                # pose tags
                elif shape and elem.tag in point_tags:
                    shape["points"][elem.tag] = float(elem.text)
                elif shape and elem.tag == "occlusion":
                    shape["occluded"] = OcclusionStates(int(elem.text))
                elif shape and elem.tag == "occlusion_kf":
                    shape["occluded_kf"] = elem.text == "1"
                elif shape and elem.tag == "truncation":
                    shape["truncated"] = TruncationStates(int(elem.text))

                # common tags
                elif attr is not None and elem.tag == "name":
                    if not elem.text:
                        raise InvalidAnnotationError("Attribute name can't be empty")
                    attr["name"] = elem.text
                elif attr is not None and elem.tag == "value":
                    attr["value"] = elem.text or ""
                elif attr is not None and elem.tag == "attribute":
                    if shape:
                        shape["attributes"][attr["name"]] = attr["value"]
                    else:
                        track["attributes"][attr["name"]] = attr["value"]
                    attr = None

        if track is not None or shape is not None or attr is not None:
            raise InvalidAnnotationError("Failed to parse annotations from '%s'" % path)

        special_attrs = KittiRawPath.SPECIAL_ATTRS
        common_attrs = ["occluded"]

        if has_meta_file(path):
            categories = {
                AnnotationType.label: LabelCategories.from_iterable(parse_meta_file(path).keys())
            }
        else:
            label_cat = LabelCategories(attributes=common_attrs)
            for label, attrs in sorted(labels.items(), key=lambda e: e[0]):
                label_cat.add(label, attributes=set(attrs) - special_attrs)

            categories = {AnnotationType.label: label_cat}

        items = {}
        for idx, track in enumerate(tracks):
            track_id = idx + 1
            for i, ann in enumerate(cls._parse_track(track_id, track, categories)):
                frame_desc = items.setdefault(track["start_frame"] + i, {"annotations": []})
                frame_desc["annotations"].append(ann)

        return items, categories

    @classmethod
    def _parse_attr(cls, value):
        if value == "true":
            return True
        elif value == "false":
            return False
        elif str(cast(value, int, 0)) == value:
            return int(value)
        elif str(cast(value, float, 0)) == value:
            return float(value)
        else:
            return value

    @classmethod
    def _parse_track(cls, track_id, track, categories):
        common_attrs = {k: cls._parse_attr(v) for k, v in track["attributes"].items()}
        scale = [track["scale"][k] for k in ["w", "h", "l"]]
        label = categories[AnnotationType.label].find(track["label"])[0]

        kf_occluded = False
        for shape in track["shapes"]:
            occluded = shape["occluded"] in {OcclusionStates.FULLY, OcclusionStates.PARTLY}
            if shape["occluded_kf"]:
                kf_occluded = occluded
            elif shape["occluded"] == OcclusionStates.OCCLUSION_UNSET:
                occluded = kf_occluded

            if shape["truncated"] in {TruncationStates.OUT_IMAGE, TruncationStates.BEHIND_IMAGE}:
                # skip these frames
                continue

            local_attrs = {k: cls._parse_attr(v) for k, v in shape["attributes"].items()}
            local_attrs["occluded"] = occluded
            local_attrs["track_id"] = track_id
            attrs = dict(common_attrs)
            attrs.update(local_attrs)

            position = [shape["points"][k] for k in ["tx", "ty", "tz"]]
            rotation = [shape["points"][k] for k in ["rx", "ry", "rz"]]

            yield Cuboid3d(position, rotation, scale, label=label, attributes=attrs)

    @staticmethod
    def _parse_name_mapping(path):
        rootdir = osp.dirname(path)

        name_mapping = {}
        if osp.isfile(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    idx, path = line.split(maxsplit=1)
                    path = osp.abspath(osp.join(rootdir, path))
                    assert path.startswith(rootdir), path
                    path = osp.relpath(path, rootdir)
                    name_mapping[int(idx)] = path

        return name_mapping

    def _load_items(self, parsed):
        images = {}
        for d in os.listdir(self._rootdir):
            image_dir = osp.join(self._rootdir, d, "data")
            if not (d.lower().startswith(KittiRawPath.IMG_DIR_PREFIX) and osp.isdir(image_dir)):
                continue

            for p in find_images(image_dir, recursive=True):
                image_name = osp.splitext(osp.relpath(p, image_dir))[0]
                images.setdefault(image_name, []).append(p)

        name_mapping = self._parse_name_mapping(
            osp.join(self._rootdir, KittiRawPath.NAME_MAPPING_FILE)
        )

        items = {}
        for frame_id, item_desc in parsed.items():
            name = name_mapping.get(frame_id, "%010d" % int(frame_id))
            items[frame_id] = DatasetItem(
                id=name,
                subset=self._subset,
                media=PointCloud.from_file(
                    path=osp.join(self._rootdir, KittiRawPath.PCD_DIR, name + ".pcd"),
                    extra_images=[
                        Image.from_file(path=image) for image in sorted(images.get(name, []))
                    ],
                ),
                annotations=item_desc.get("annotations"),
                attributes={"frame": int(frame_id)},
            )
            for ann in item_desc.get("annotations"):
                self._ann_types.add(ann.type)

        for frame_id, name in name_mapping.items():
            if frame_id in items:
                continue

            items[frame_id] = DatasetItem(
                id=name,
                subset=self._subset,
                media=PointCloud.from_file(
                    path=osp.join(self._rootdir, KittiRawPath.PCD_DIR, name + ".pcd"),
                    extra_images=[
                        Image.from_file(path=image) for image in sorted(images.get(name, []))
                    ],
                ),
                attributes={"frame": int(frame_id)},
            )

        return items
