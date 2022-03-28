# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from collections import OrderedDict

from datumaro.components.annotation import AnnotationType, Bbox
from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.errors import MediaTypeError
from datumaro.components.extractor import DEFAULT_SUBSET_NAME, DatasetItem, IExtractor
from datumaro.components.media import Image
from datumaro.util import str_to_bool

from .format import YoloPath


def _make_yolo_bbox(img_size, box):
    # https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
    # <x> <y> <width> <height> - values relative to width and height of image
    # <x> <y> - are center of rectangle
    x = (box[0] + box[2]) / 2 / img_size[0]
    y = (box[1] + box[3]) / 2 / img_size[1]
    w = (box[2] - box[0]) / img_size[0]
    h = (box[3] - box[1]) / img_size[1]
    return x, y, w, h


class YoloConverter(Converter):
    # https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
    DEFAULT_IMAGE_EXT = ".jpg"

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--add-path-prefix",
            default=True,
            type=str_to_bool,
            help="Add the 'data/' prefix for paths in the dataset info (default: %(default)s)",
        )
        return parser

    def __init__(
        self, extractor: IExtractor, save_dir: str, *, add_path_prefix: bool = True, **kwargs
    ) -> None:
        super().__init__(extractor, save_dir, **kwargs)

        self._prefix = "data" if add_path_prefix else ""

    def apply(self):
        extractor = self._extractor
        save_dir = self._save_dir

        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(save_dir, exist_ok=True)

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        label_categories = extractor.categories()[AnnotationType.label]
        label_ids = {label.name: idx for idx, label in enumerate(label_categories.items)}
        with open(osp.join(save_dir, "obj.names"), "w", encoding="utf-8") as f:
            f.writelines("%s\n" % l[0] for l in sorted(label_ids.items(), key=lambda x: x[1]))

        subset_lists = OrderedDict()

        subsets = self._extractor.subsets()
        pbars = self._ctx.progress_reporter.split(len(subsets))
        for (subset_name, subset), pbar in zip(subsets.items(), pbars):
            if not subset_name or subset_name == DEFAULT_SUBSET_NAME:
                subset_name = YoloPath.DEFAULT_SUBSET_NAME
            elif subset_name not in YoloPath.SUBSET_NAMES:
                log.warning(
                    "Skipping subset export '%s'. "
                    "If specified, the only valid names are %s"
                    % (subset_name, ", ".join("'%s'" % s for s in YoloPath.SUBSET_NAMES))
                )
                continue

            subset_dir = osp.join(save_dir, "obj_%s_data" % subset_name)
            os.makedirs(subset_dir, exist_ok=True)

            image_paths = OrderedDict()
            for item in pbar.iter(subset, desc=f"Exporting '{subset_name}'"):
                try:
                    if not item.media or not (item.media.has_data or item.media.has_size):
                        raise Exception(
                            "Failed to export item '%s': " "item has no image info" % item.id
                        )

                    image_name = self._make_image_filename(item)
                    if self._save_media:
                        if item.media:
                            self._save_image(item, osp.join(subset_dir, image_name))
                        else:
                            log.warning("Item '%s' has no image" % item.id)
                    image_paths[item.id] = osp.join(
                        self._prefix, osp.basename(subset_dir), image_name
                    )

                    yolo_annotation = self._export_item_annotation(item)
                    annotation_path = osp.join(subset_dir, "%s.txt" % item.id)
                    os.makedirs(osp.dirname(annotation_path), exist_ok=True)
                    with open(annotation_path, "w", encoding="utf-8") as f:
                        f.write(yolo_annotation)
                except Exception as e:
                    self._report_item_error(e, item_id=(item.id, item.subset))

            subset_list_name = f"{subset_name}.txt"
            subset_list_path = osp.join(save_dir, subset_list_name)
            if self._patch and subset_name in self._patch.updated_subsets and not image_paths:
                if osp.isfile(subset_list_path):
                    os.remove(subset_list_path)
                continue

            subset_lists[subset_name] = subset_list_name
            with open(subset_list_path, "w", encoding="utf-8") as f:
                f.writelines("%s\n" % s.replace("\\", "/") for s in image_paths.values())

        with open(osp.join(save_dir, "obj.data"), "w", encoding="utf-8") as f:
            f.write(f"classes = {len(label_ids)}\n")

            for subset_name, subset_list_name in subset_lists.items():
                f.write(
                    "%s = %s\n"
                    % (subset_name, osp.join(self._prefix, subset_list_name).replace("\\", "/"))
                )

            f.write("names = %s\n" % osp.join(self._prefix, "obj.names"))
            f.write("backup = backup/\n")

    def _export_item_annotation(self, item):
        height, width = item.media.size

        yolo_annotation = ""

        for bbox in item.annotations:
            if not isinstance(bbox, Bbox) or bbox.label is None:
                continue

            yolo_bb = _make_yolo_bbox((width, height), bbox.points)
            yolo_bb = " ".join("%.6f" % p for p in yolo_bb)
            yolo_annotation += "%s %s\n" % (bbox.label, yolo_bb)

        return yolo_annotation

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        conv = cls(dataset, save_dir=save_dir, **kwargs)
        conv._patch = patch
        conv.apply()

        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.media):
                continue

            if subset == DEFAULT_SUBSET_NAME:
                subset = YoloPath.DEFAULT_SUBSET_NAME
            subset_dir = osp.join(save_dir, "obj_%s_data" % subset)

            image_path = osp.join(subset_dir, conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.remove(image_path)

            ann_path = osp.join(subset_dir, "%s.txt" % item.id)
            if osp.isfile(ann_path):
                os.remove(ann_path)
