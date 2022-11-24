# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import os.path as osp

import h5py
import numpy as np

from datumaro.components.annotation import DepthAnnotation
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.media import Image


class NyuDepthV2Base(SubsetBase):
    def __init__(self, path, subset=None, save_hash=False):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__(subset=subset)

        self._save_hash = save_hash
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
                annotations=[DepthAnnotation(image=Image(data=depth))],
                save_hash=self._save_hash,
            )

        return items


class NyuDepthV2Importer(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("*.h5")

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": "nyu_depth_v2"}]
