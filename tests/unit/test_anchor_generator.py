# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.anchor_generator import DataAwareAnchorGenerator

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path

try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True


@pytest.fixture
def fxt_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="train",
                annotations=[
                    Bbox(25, 5, 45, 22, label=0),
                    Bbox(57, 69, 45, 22, label=1),
                ],
                media=Image.from_numpy(data=np.ones((128, 128, 3))),
            ),
            DatasetItem(
                id="1",
                subset="train",
                annotations=[
                    Bbox(57, 5, 45, 22, label=1),
                    Bbox(25, 101, 45, 22, label=2),
                    Bbox(89, 101, 45, 22, label=4),
                ],
                media=Image.from_numpy(data=np.ones((128, 128, 3))),
            ),
            DatasetItem(
                id="2",
                subset="val",
                annotations=[
                    Bbox(51, 73, 91, 45, label=1),
                    Bbox(51, 9, 91, 45, label=2),
                    Bbox(89, 101, 45, 22, label=3),
                ],
                media=Image.from_numpy(data=np.ones((128, 128, 3))),
            ),
            DatasetItem(
                id="3",
                subset="val",
                annotations=[
                    Bbox(57, 101, 45, 22, label=3),
                ],
                media=Image.from_numpy(data=np.ones((128, 128, 3))),
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable([f"label_{i}" for i in range(4)])
        },
    )


@pytest.mark.new
@mark_requirement(Requirements.DATUM_GENERAL_REQ)
class DataAwareAnchorGeneratorTest:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_strides,fxt_scales,fxt_ratios",
        [
            (
                None,
                [32, 64],
                [[1.0], [1.0]],
                [[1.0], [1.0]],
            ),
            (
                "train",
                [32],
                [[1.0]],
                [[1.0]],
            ),
            (
                "val",
                [16],
                [[1.0]],
                [[1.0]],
            ),
        ],
    )
    def test_can_optimize_anchor_generator(
        self,
        fxt_dataset: Dataset,
        fxt_subset: str,
        fxt_strides: List[int],
        fxt_scales: List[List[float]],
        fxt_ratios: List[List[float]],
        request: pytest.FixtureRequest,
    ):
        prior_gen = DataAwareAnchorGenerator(
            img_size=(128, 128),
            strides=fxt_strides,
            scales=fxt_scales,
            ratios=fxt_ratios,
            pos_thr=0.7,
            neg_thr=0.3,
            device="cpu",
        )
        opt_scales, opt_ratios = prior_gen.optimize(
            dataset=fxt_dataset, subset=fxt_subset, num_iters=10
        )

        print(opt_scales, opt_ratios)
