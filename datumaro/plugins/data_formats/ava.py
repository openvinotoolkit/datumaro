# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.annotation import Bbox
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.util.os_util import find_files


class VggFace2Path:
    IMAGE_DIR = "frames"
    IMAGE_EXT = ".jpg"
    ANNOTATION_DIR = "annotations"
    ANNOTATION_EXT = ".txt"


class AvaBase(DatasetBase):
    def __init__(self, path):
        assert osp.isdir(path), path
        self._path = path

        super().__init__()

        self._items = {}
        for ann_file in find_files(dirpath=path, exts="csv", recursive=True, max_depth=1):
            self._load_items(ann_file)

    def _load_items(self, ann_file):
        subset = osp.splitext(osp.basename(ann_file))[0]
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line_split = line.strip().split(",")

                video_id = line_split[0]
                timestamp = line_split[1]  # 6-digits?

                bbox = list(map(float, line_split[2:6]))
                label = int(line_split[6])
                entity_id = int(line_split[7])

                item_id = video_id + "/" + timestamp
                image_path = osp.join(self._path, "frames", video_id, timestamp, ".jpg")

                item = self._items.get(item_id)
                if item is None:
                    item = DatasetItem(
                        id=item_id,
                        subset=subset,
                        media=Image(path=image_path),
                    )
                    self._items[item_id] = item

                anns = item.annotations
                anns.append(
                    Bbox(
                        x=bbox[0],
                        y=bbox[2],
                        w=bbox[1],
                        h=bbox[3],
                        label=label,
                        attributes={"track_id": entity_id},
                    )
                )


class AvaImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if find_files(path, exts="csv", recursive=True, max_depth=1):
            return [{"url": path, "format": AvaBase.NAME}]


class AvaExporter(Exporter):
    def apply(self):
        def _get_name_id(item_parts, label_name):
            if 1 < len(item_parts) and item_parts[0] == label_name:
                return "/".join([label_name, *item_parts[1:]])
            else:
                return "/".join([label_name, *item_parts])

        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        save_dir = self._save_dir
        os.makedirs(save_dir, exist_ok=True)

        if self._save_dataset_meta:
            self._save_meta_file(save_dir)
        else:
            labels_path = osp.join(save_dir, VggFace2Path.LABELS_FILE)
            labels_file = ""
            for label in self._extractor.categories()[AnnotationType.label]:
                labels_file += "%s" % label.name
                if label.parent:
                    labels_file += " %s" % label.parent
                labels_file += "\n"
            with open(labels_path, "w", encoding="utf-8") as f:
                f.write(labels_file)

        label_categories = self._extractor.categories()[AnnotationType.label]

        for subset_name, subset in self._extractor.subsets().items():
            bboxes_table = []
            landmarks_table = []
            for item in subset:
                item_parts = item.id.split("/")
                if item.media and self._save_media:
                    labels = set(
                        p.label for p in item.annotations if getattr(p, "label") is not None
                    )
                    if labels:
                        for label in labels:
                            image_dir = label_categories[label].name
                            if 1 < len(item_parts) and image_dir == item_parts[0]:
                                image_dir = ""
                            self._save_image(item, subdir=osp.join(subset_name, image_dir))
                    else:
                        image_dir = VggFace2Path.IMAGES_DIR_NO_LABEL
                        if 1 < len(item_parts) and image_dir == item_parts[0]:
                            image_dir = ""
                        self._save_image(item, subdir=osp.join(subset_name, image_dir))

                landmarks = [a for a in item.annotations if a.type == AnnotationType.points]
                if 1 < len(landmarks):
                    raise Exception(
                        "Item (%s, %s): an image can have only one "
                        "set of landmarks" % (item.id, item.subset)
                    )
                if landmarks:
                    if landmarks[0].label is not None and label_categories[landmarks[0].label].name:
                        name_id = _get_name_id(
                            item_parts, label_categories[landmarks[0].label].name
                        )
                    else:
                        name_id = _get_name_id(item_parts, VggFace2Path.IMAGES_DIR_NO_LABEL)
                    points = landmarks[0].points
                    if len(points) != 10:
                        landmarks_table.append({"NAME_ID": name_id})
                    else:
                        landmarks_table.append(
                            {
                                "NAME_ID": name_id,
                                "P1X": points[0],
                                "P1Y": points[1],
                                "P2X": points[2],
                                "P2Y": points[3],
                                "P3X": points[4],
                                "P3Y": points[5],
                                "P4X": points[6],
                                "P4Y": points[7],
                                "P5X": points[8],
                                "P5Y": points[9],
                            }
                        )

                bboxes = [a for a in item.annotations if a.type == AnnotationType.bbox]
                if 1 < len(bboxes):
                    raise Exception(
                        "Item (%s, %s): an image can have only one " "bbox" % (item.id, item.subset)
                    )
                if bboxes:
                    if bboxes[0].label is not None and label_categories[bboxes[0].label].name:
                        name_id = _get_name_id(item_parts, label_categories[bboxes[0].label].name)
                    else:
                        name_id = _get_name_id(item_parts, VggFace2Path.IMAGES_DIR_NO_LABEL)
                    bboxes_table.append(
                        {
                            "NAME_ID": name_id,
                            "X": bboxes[0].x,
                            "Y": bboxes[0].y,
                            "W": bboxes[0].w,
                            "H": bboxes[0].h,
                        }
                    )

                labels = [a for a in item.annotations if a.type == AnnotationType.label]
                for label in labels:
                    if label.label is not None and label_categories[label.label].name:
                        name_id = _get_name_id(item_parts, label_categories[labels[0].label].name)
                    else:
                        name_id = _get_name_id(item_parts, VggFace2Path.IMAGES_DIR_NO_LABEL)
                    landmarks_table.append({"NAME_ID": name_id})

                if not landmarks and not bboxes and not labels:
                    landmarks_table.append(
                        {"NAME_ID": _get_name_id(item_parts, VggFace2Path.IMAGES_DIR_NO_LABEL)}
                    )

            landmarks_path = osp.join(
                save_dir,
                VggFace2Path.ANNOTATION_DIR,
                VggFace2Path.LANDMARKS_FILE + subset_name + ".csv",
            )
            os.makedirs(osp.dirname(landmarks_path), exist_ok=True)
            with open(landmarks_path, "w", encoding="utf-8", newline="") as file:
                columns = [
                    "NAME_ID",
                    "P1X",
                    "P1Y",
                    "P2X",
                    "P2Y",
                    "P3X",
                    "P3Y",
                    "P4X",
                    "P4Y",
                    "P5X",
                    "P5Y",
                ]
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(landmarks_table)

            if bboxes_table:
                bboxes_path = osp.join(
                    save_dir,
                    VggFace2Path.ANNOTATION_DIR,
                    VggFace2Path.BBOXES_FILE + subset_name + ".csv",
                )
                os.makedirs(osp.dirname(bboxes_path), exist_ok=True)
                with open(bboxes_path, "w", encoding="utf-8", newline="") as file:
                    columns = ["NAME_ID", "X", "Y", "W", "H"]
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(bboxes_table)
