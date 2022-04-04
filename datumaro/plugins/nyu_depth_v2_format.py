# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import os.path as osp

import h5py
import numpy as np

from datumaro.components.annotation import ImageAnnotation
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.media import Image


class NyuDepthV2Extractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__(subset=subset)

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        anno_files = glob.glob(osp.join(path, "**", "*.h5"), recursive=True)
        for anno_file in anno_files:
            item_id = osp.splitext(osp.basename(anno_file))[0]
            with h5py.File(anno_file, "r") as f:
                image = np.transpose(f["rgb"], (1, 2, 0))
                depth = f["depth"][:].astype("float16")

            items[item_id] = DatasetItem(
                id=item_id,
                media=Image(data=image),
                annotations=[ImageAnnotation(image=Image(data=depth))],
            )

        return items


class NyuDepthV2Importer(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("*.h5")

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": "nyu_depth_v2"}]
