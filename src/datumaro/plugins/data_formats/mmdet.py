# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from glob import glob
from typing import Optional

from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME
from datumaro.components.importer import ImportContext
from datumaro.plugins.data_formats.coco.base import _CocoBase
from datumaro.plugins.data_formats.coco.format import CocoImporterType, CocoTask
from datumaro.plugins.data_formats.coco.importer import CocoImporter


class MmdetCocoImporter(CocoImporter):
    def __call__(self, path, stream: bool = False, **extra_params):
        subset_paths = glob(osp.join(path, "**", "instances_*.json"), recursive=True)

        sources = []
        for subset_path in subset_paths:
            parts = osp.splitext(osp.basename(subset_path))[0].split("instances_", maxsplit=1)
            subset_name = parts[1] if len(parts) == 2 else DEFAULT_SUBSET_NAME

            options = dict(extra_params)
            options["subset"] = subset_name

            if stream:
                options["stream"] = True

            sources.append({"url": subset_path, "format": "mmdet_coco", "options": options})

        return sources


class MmdetCocoBase(_CocoBase):
    """
    Parses Roboflow COCO annotations written in the following format:
    https://cocodataset.org/#format-data
    """

    def __init__(
        self,
        path,
        *,
        subset: Optional[str] = None,
        stream: bool = False,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(
            path,
            task=CocoTask.instances,
            coco_importer_type=CocoImporterType.mmdet,
            subset=subset,
            stream=stream,
            ctx=ctx,
        )
