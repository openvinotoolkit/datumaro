# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import logging as log
import os
import os.path as osp
from itertools import chain
from typing import List, Union

from pycocotools import mask as mask_utils

from datumaro.components.annotation import AnnotationType, Ellipse, Polygon
from datumaro.components.errors import DatumaroError, MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image
from datumaro.util import NOTSET
from datumaro.util import annotation_util as anno_tools
from datumaro.util import dump_json_file, mask_tools


def replace(json_data, key, value_new):
    value_origin = json_data[key]
    if value_origin is NOTSET:
        json_data[key] = value_new
        return
    if value_origin != value_new:
        raise DatumaroError(f"The value for '{key}' is not same for item {value_new}")


class SegmentAnythingExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"

    _polygon_types = {AnnotationType.polygon, AnnotationType.ellipse}
    _allowed_types = {
        AnnotationType.bbox,
        AnnotationType.polygon,
        AnnotationType.mask,
        AnnotationType.ellipse,
    }

    def __init__(
        self,
        extractor,
        save_dir,
        **kwargs,
    ):
        super().__init__(extractor, save_dir, **kwargs)

    @staticmethod
    def find_instance_anns(annotations):
        return [a for a in annotations if a.type in SegmentAnythingExporter._allowed_types]

    @classmethod
    def find_instances(cls, annotations):
        return anno_tools.find_instances(cls.find_instance_anns(annotations))

    def get_annotation_info(self, group, img_width, img_height):
        boxes = [a for a in group if a.type == AnnotationType.bbox]
        polygons: List[Union[Polygon, Ellipse]] = [
            a for a in group if a.type in self._polygon_types
        ]
        masks = [a for a in group if a.type == AnnotationType.mask]

        anns = boxes + polygons + masks
        leader = anno_tools.find_group_leader(anns)
        if len(boxes) > 0:
            bbox = anno_tools.max_bbox(boxes)
        else:
            bbox = anno_tools.max_bbox(anns)
        polygons = [p.as_polygon() for p in polygons]

        mask = None
        if polygons:
            mask = mask_tools.rles_to_mask(polygons, img_width, img_height)
        if masks:
            masks = (m.image for m in masks)
            if mask is not None:
                masks = chain(masks, [mask])
            mask = mask_tools.merge_masks(masks)
        if mask is None:
            return None
        mask = mask_tools.mask_to_rle(mask)

        segmentation = {
            "counts": list(int(c) for c in mask["counts"]),
            "size": list(int(c) for c in mask["size"]),
        }
        rles = mask_utils.frPyObjects(segmentation, img_height, img_width)
        if isinstance(rles["counts"], bytes):
            rles["counts"] = rles["counts"].decode()
        area = mask_utils.area(rles)

        annotation_data = {
            "id": leader.group,
            "segmentation": rles,
            "bbox": bbox,
            "area": area,
            "predicted_iou": 0.0,
            "stability_score": 0.0,
            "crop_box": [],
            "point_coords": [],
        }
        for ann in anns:
            annotation_data["predicted_iou"] = max(
                annotation_data["predicted_iou"], ann.attributes.get("predicted_iou", 0.0)
            )
            annotation_data["stability_score"] = max(
                annotation_data["stability_score"], ann.attributes.get("stability_score", 0.0)
            )
            crop_box = ann.attributes.get("crop_box", [])
            if crop_box:
                crop_box = [crop_box]
                if annotation_data["crop_box"]:
                    crop_box.append(annotation_data["crop_box"])
                annotation_data["crop_box"] = anno_tools.max_bbox(crop_box)
            point_coords = ann.attributes.get("point_coords", [[]])
            if len(point_coords[0]) > 0:
                for point_coord in point_coords:
                    if point_coord not in annotation_data["point_coords"]:
                        annotation_data["point_coords"].append(point_coord)
        return annotation_data

    def apply(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)

        subsets = self._extractor.subsets()
        pbars = self._ctx.progress_reporter.split(len(subsets))

        max_image_id = 1
        for pbar, (subset_name, subset) in zip(pbars, subsets.items()):
            for item in pbar.iter(subset, desc=f"Exporting {subset_name}"):
                try:
                    # make sure file_name is flat
                    file_name = self._make_image_filename(item).replace("/", "__")
                    try:
                        image_id = int(item.attributes.get("id", max_image_id))
                    except ValueError:
                        image_id = max_image_id
                    max_image_id += 1

                    if not item.media or not item.media.size:
                        log.warning(
                            "Item '%s': skipping writing instances since no image info available"
                            % item.id
                        )
                        continue

                    height, width = item.media.size
                    json_data = {
                        "image": {
                            "image_id": image_id,
                            "file_name": file_name,
                            "height": height,
                            "width": width,
                        },
                        "annotations": [],
                    }

                    instances = self.find_instances(item.annotations)
                    annotations = [self.get_annotation_info(i, width, height) for i in instances]
                    annotations = [i for i in annotations if i is not None]
                    if not annotations:
                        log.warning(
                            "Item '%s': skipping writing instances since no annotation available"
                            % item.id
                        )
                        continue
                    json_data["annotations"] = annotations

                    dump_json_file(
                        os.path.join(self._save_dir, osp.splitext(file_name)[0] + ".json"),
                        json_data,
                    )

                    if self._save_media:
                        self._save_image(
                            item,
                            path=osp.abspath(osp.join(self._save_dir, file_name)),
                        )

                except Exception as e:
                    self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))
