# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional
from glob import glob
import os.path as osp

from datumaro.components.importer import ImportContext
from datumaro.plugins.data_formats.coco.base import _CocoBase
from datumaro.plugins.data_formats.coco.format import CocoImporterType, CocoTask
from datumaro.plugins.data_formats.coco.importer import CocoImporter


class MmdetCocoImporter(CocoImporter):
    def __call__(self, path, stream: bool = False, **extra_params):
        subset_paths = glob(osp.join(path, "**", "instances_*.json"), recursive=True)

        sources = []
        for subset_path in subset_paths:
            subset_name = osp.basename(osp.dirname(subset_path))

            options = dict(extra_params)
            options["subset"] = subset_name

            if stream:
                options["stream"] = True

            sources.append(
                {"url": subset_path, "format": "mmdet_coco", "options": options}
            )

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
