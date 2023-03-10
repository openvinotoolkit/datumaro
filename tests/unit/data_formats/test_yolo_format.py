# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import pytest

from datumaro.components.annotation import Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.data_formats.common_semantic_segmentation import (
    CommonSemanticSegmentationImporter,
    CommonSemanticSegmentationWithSubsetDirsImporter,
    make_categories,
)
from datumaro.plugins.data_formats.yolo.base import YoloImporter

from .base import TestDataFormatBase

from tests.utils.assets import get_test_asset_path

STRICT_DIR = get_test_asset_path("yolo_dataset", "strict")
ANNOTATIONS_DIR = get_test_asset_path("yolo_dataset", "annotations")
LABELS_DIR = get_test_asset_path("yolo_dataset", "strict")


class YoloImporterTest(TestDataFormatBase):
    IMPORTER = YoloImporter

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [STRICT_DIR, ANNOTATIONS_DIR, LABELS_DIR],
        ids=["strict", "annotations", "labels"],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        return super().test_can_detect(fxt_dataset_dir)
