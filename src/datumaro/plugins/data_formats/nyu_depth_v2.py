# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import glob
import os.path as osp
from typing import List, Optional

import h5py
import numpy as np

from datumaro.components.annotation import AnnotationType, DepthAnnotation
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image


class NyuDepthV2Base(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isdir(path):
            raise NotADirectoryError(errno.ENOTDIR, "Can't find dataset directory", path)

        super().__init__(subset=subset, ctx=ctx)

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
                media=Image.from_numpy(data=image),
                annotations=[DepthAnnotation(image=Image.from_numpy(data=depth))],
            )
            self._ann_types.add(AnnotationType.depth_annotation)

        return items


class NyuDepthV2Importer(Importer):
    _FORMAT_EXT = ".h5"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f"*{cls._FORMAT_EXT}")

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": "nyu_depth_v2"}]

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._FORMAT_EXT]
