# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import logging as log
import os
import os.path as osp
from collections import defaultdict

from pycocotools import mask as mask_utils

from datumaro.components.annotation import AnnotationType
from datumaro.components.errors import DatumaroError, MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image
from datumaro.util import NOTSET, dump_json_file


def replace(json_data, key, value_new):
    value_origin = json_data[key]
    if value_origin is NOTSET:
        json_data[key] = value_new
        return
    if value_origin != value_new:
        raise DatumaroError(f"The value for '{key}' is not same for item {value_new}")


class SegmentAnythingExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"

    def __init__(
        self,
        extractor,
        save_dir,
        **kwargs,
    ):
        super().__init__(extractor, save_dir, **kwargs)

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

                    annotations_grouped = defaultdict(list)
                    for annotation in item.annotations:
                        annotations_grouped[annotation.id].append(annotation)

                    for id, annotations in annotations_grouped.items():
                        annotation_data = {
                            "id": id,
                            "segmentation": NOTSET,
                            "bbox": NOTSET,
                            "area": NOTSET,
                            "predicted_iou": NOTSET,
                            "stability_score": NOTSET,
                            "crop_box": NOTSET,
                            "point_coords": NOTSET,
                        }

                        for annotation in annotations:
                            if annotation.type == AnnotationType.bbox:
                                replace(annotation_data, "bbox", annotation.get_bbox())
                            elif annotation.type == AnnotationType.mask:
                                if hasattr(annotation, "rle"):
                                    rle_encoded = annotation.rle
                                else:
                                    rle_encoded = mask_utils.encode(annotation.image)
                                if isinstance(rle_encoded["counts"], bytes):
                                    rle_encoded["counts"] = rle_encoded["counts"].decode()
                                replace(annotation_data, "segmentation", rle_encoded)
                                replace(annotation_data, "area", annotation.get_area())
                            else:
                                continue

                            replace(
                                annotation_data,
                                "predicted_iou",
                                annotation.attributes.get("predicted_iou", 0.0),
                            )
                            replace(
                                annotation_data,
                                "stability_score",
                                annotation.attributes.get("stability_score", 0.0),
                            )
                            replace(
                                annotation_data,
                                "crop_box",
                                annotation.attributes.get("crop_box", []),
                            )
                            replace(
                                annotation_data,
                                "point_coords",
                                annotation.attributes.get("point_coords", [[]]),
                            )

                        if annotation_data["segmentation"] is NOTSET:
                            continue
                        if (
                            annotation_data["bbox"] is NOTSET
                            and annotation_data["segmentation"] is not NOTSET
                        ):
                            annotation_data["bbox"] = (
                                annotation_data["segmentation"].get_bbox().tolist()
                            )
                        json_data["annotations"].append(annotation_data)

                    if not json_data["annotations"]:
                        continue

                    dump_json_file(
                        os.path.join(self._save_dir, osp.splitext(file_name)[0] + ".json"),
                        json_data,
                    )

                    if self._save_media:
                        if item.media:
                            self._save_image(
                                item,
                                path=osp.abspath(osp.join(self._save_dir, file_name)),
                            )
                        else:
                            log.debug("Item '%s' has no image info", item.id)

                except Exception as e:
                    self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))
