# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import os.path as osp
from typing import List, Optional, Type, TypeVar

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import InvalidAnnotationError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from datumaro.util.image import find_images

from .format import Kitti3dPath

T = TypeVar("T")


class Kitti3dBase(SubsetBase):
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
        assert osp.isdir(path), path
        super().__init__(subset=subset, ctx=ctx)

        self._path = path
        self._categories = {AnnotationType.label: LabelCategories()}
        self._items = self._load_items()

    def _load_items(self) -> List[DatasetItem]:
        items = []
        image_dir = osp.join(self._path, Kitti3dPath.IMAGE_DIR)
        image_path_by_id = {
            osp.splitext(osp.relpath(p, image_dir))[0]: p
            for p in find_images(image_dir, recursive=True)
        }

        ann_dir = osp.join(self._path, Kitti3dPath.LABEL_DIR)
        label_categories = self._categories[AnnotationType.label]
        for labels_path in sorted(glob.glob(osp.join(ann_dir, "*.txt"), recursive=True)):
            item_id = osp.splitext(osp.relpath(labels_path, ann_dir))[0]
            anns = []

            with open(labels_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines):
                line = line.split()
                assert len(line) == 15 or len(line) == 16

                label_name = line[0]
                label_id = label_categories.find(label_name)[0]
                if label_id is None:
                    label_id = label_categories.add(label_name)

                x1 = self._parse_field(line[4], float, "bbox left-top x")
                y1 = self._parse_field(line[5], float, "bbox left-top y")
                x2 = self._parse_field(line[6], float, "bbox right-bottom x")
                y2 = self._parse_field(line[7], float, "bbox right-bottom y")

                attributes = {}
                attributes["truncated"] = self._parse_field(line[1], float, "truncated")
                attributes["occluded"] = self._parse_field(line[2], int, "occluded")
                attributes["alpha"] = self._parse_field(line[3], float, "alpha")

                height_3d = self._parse_field(line[8], float, "height (in meters)")
                width_3d = self._parse_field(line[9], float, "width (in meters)")
                length_3d = self._parse_field(line[10], float, "length (in meters)")

                x_3d = self._parse_field(line[11], float, "x (in meters)")
                y_3d = self._parse_field(line[12], float, "y (in meters)")
                z_3d = self._parse_field(line[13], float, "z (in meters)")

                yaw_angle = self._parse_field(line[14], float, "rotation_y")

                attributes["dimensions"] = [height_3d, width_3d, length_3d]
                attributes["location"] = [x_3d, y_3d, z_3d]
                attributes["rotation_y"] = yaw_angle

                if len(line) == 16:
                    attributes["score"] = self._parse_field(line[15], float, "score")

                anns.append(
                    Bbox(
                        x=x1,
                        y=y1,
                        w=x2 - x1,
                        h=y2 - y1,
                        id=line_idx,
                        attributes=attributes,
                        label=label_id,
                    )
                )
                self._ann_types.add(AnnotationType.bbox)

            image = image_path_by_id.pop(item_id, None)
            if image:
                image = Image.from_file(path=image)

            items.append(
                DatasetItem(id=item_id, annotations=anns, media=image, subset=self._subset)
            )

        return items

    def _parse_field(self, value: str, desired_type: Type[T], field_name: str) -> T:
        try:
            return desired_type(value)
        except Exception as e:
            raise InvalidAnnotationError(
                f"Can't parse {field_name} from '{value}'. Expected {desired_type}"
            ) from e
