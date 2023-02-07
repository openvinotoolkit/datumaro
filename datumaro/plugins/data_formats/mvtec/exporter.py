# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from enum import Enum, auto

import cv2
import numpy as np

from datumaro.components.annotation import AnnotationType, CompiledMask, LabelCategories
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image
from datumaro.util import cast, parse_str_enum_value, str_to_bool
from datumaro.util.annotation_util import make_label_id_mapping
from datumaro.util.image import save_image
from datumaro.util.mask_tools import paint_mask
from datumaro.util.meta_file_util import is_meta_file, parse_meta_file

from .format import MvtecPath, MvtecTask


class LabelmapType(Enum):
    kitti = auto()
    source = auto()


class MvtecExporter(Exporter):
    DEFAULT_IMAGE_EXT = MvtecPath.IMAGE_EXT

    @staticmethod
    def _split_tasks_string(s):
        return [MvtecTask[i.strip().lower()] for i in s.split(",")]

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument(
            "--tasks",
            type=cls._split_tasks_string,
            help="MVTec task filter, comma-separated list of {%s} "
            "(default: all)" % ", ".join(t.name for t in MvtecTask),
        )
        return parser

    def __init__(
        self,
        extractor,
        save_dir,
        tasks=None,
        **kwargs,
    ):
        super().__init__(extractor, save_dir, **kwargs)

        assert tasks is None or isinstance(tasks, (MvtecTask, list, set))
        if tasks is None:
            tasks = set(MvtecTask)
        elif isinstance(tasks, MvtecTask):
            tasks = {tasks}
        else:
            tasks = set(parse_str_enum_value(t, MvtecTask) for t in tasks)
        self._tasks = tasks

    def apply(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)

        for subset_name, subset in self._extractor.subsets().items():
            if subset_name == MvtecPath.TEST_DIR and (
                MvtecTask.segmentation in self._tasks or MvtecTask.detection in self._tasks
            ):
                os.makedirs(
                    osp.join(self._save_dir, subset_name, MvtecPath.MASK_DIR), exist_ok=True
                )

            for item in subset:
                labels = []
                for ann in item.annotations:
                    if ann.type in [AnnotationType.label, AnnotationType.mask, AnnotationType.bbox]:
                        labels.append(self.get_label(ann.label))

                if self._save_media:
                    self._save_image(item, subdir=osp.join(subset_name, labels[0]))

                bboxes = [a for a in item.annotations if a.type == AnnotationType.bbox]
                if bboxes and MvtecTask.detection in self._tasks:
                    mask_path = osp.join(
                        MvtecPath.MASK_DIR, item.id + MvtecPath.POSTFIX + MvtecPath.MASK_EXT
                    )
                    x, y, h, w = bbox.get_bbox()
                    masks = np.zeros(item.media.size)
                    masks = cv2.rectangle(masks, (x, y), (x + w, y + h), (255, 255, 255), -1)

                    self.save_mask(
                        osp.join(self._save_dir, mask_path),
                        masks,
                        dtype=np.int32,
                    )

                masks = [a for a in item.annotations if a.type == AnnotationType.mask]
                if masks and MvtecTask.segmentation in self._tasks:
                    mask_path = osp.join(
                        MvtecPath.MASK_DIR, item.id + MvtecPath.POSTFIX + MvtecPath.MASK_EXT
                    )
                    self.save_mask(
                        osp.join(self._save_dir, mask_path),
                        masks,
                        dtype=np.int32,
                    )

    def get_label(self, label_id):
        return self._extractor.categories()[AnnotationType.label].items[label_id].name

    def save_mask(self, path, mask, colormap=None, apply_colormap=True, dtype=np.uint8):
        if self._apply_colormap and apply_colormap:
            if colormap is None:
                colormap = self._categories[AnnotationType.mask].colormap
            mask = paint_mask(mask, colormap)
        save_image(path, mask, create_dir=True, dtype=dtype)


class MvtecClassificationExporter(MvtecExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = MvtecTask.classification
        super().__init__(*args, **kwargs)


class MvtecSegmentationExporter(MvtecExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = MvtecTask.segmentation
        super().__init__(*args, **kwargs)


class MvtecDetectionExporter(MvtecExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = MvtecTask.detection
        super().__init__(*args, **kwargs)
