# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import pytest

from datumaro import Dataset
from datumaro.components.algorithms.noisy_label_detection import (
    LossDynamicsAnalyzer,
    NoisyLabelCandidate,
)

from tests.utils.assets import get_test_asset_path


class LossDynamicsAnalyzerTestBase:
    def test_ema_dataframe(
        self, fxt_loss_dyns_dataset: Dataset, fxt_analyzer: LossDynamicsAnalyzer
    ):
        # Check all annotations are included
        n_anns = sum(len(item.annotations) for item in fxt_loss_dyns_dataset)
        assert n_anns == len(fxt_analyzer.ema_dataframe)

    def test_get_top_k_cands(self, fxt_analyzer: LossDynamicsAnalyzer):
        top_k = 5
        cands = fxt_analyzer.get_top_k_cands(top_k)

        assert len(cands) == top_k
        assert sorted(cands, reverse=True) == cands  # Should be sorted in the decreasing order

    def test_plot_ema_loss_dynamics(
        self, fxt_loss_dyns_dataset: Dataset, fxt_analyzer: LossDynamicsAnalyzer
    ):
        n_items = len(fxt_loss_dyns_dataset)
        cands = fxt_analyzer.get_top_k_cands(n_items)
        n_labels = len({cand.label_id for cand in cands})

        fig = fxt_analyzer.plot_ema_loss_dynamics(cands, mode="mean")
        assert len(fig.axes) == 1  # mode="mean" should have a single plot

        fig = fxt_analyzer.plot_ema_loss_dynamics(cands, mode="label_mean")
        assert (
            len(fig.axes) == n_labels
        )  # mode="label_mean" should have multiple subplots sized of n_labels


class LossDynamicsAnalyzerForClassificationTest(LossDynamicsAnalyzerTestBase):
    @pytest.fixture()
    def fxt_loss_dyns_dataset(self) -> Dataset:
        subdir = osp.join("algorithms", "noisy_label_detection_cls")
        dirpath = get_test_asset_path(subdir)
        return Dataset.import_from(dirpath, format="datumaro")

    @pytest.fixture()
    def fxt_analyzer(self, fxt_loss_dyns_dataset: Dataset) -> LossDynamicsAnalyzer:
        return LossDynamicsAnalyzer(fxt_loss_dyns_dataset)


class LossDynamicsAnalyzerForDetectionTest(LossDynamicsAnalyzerTestBase):
    @pytest.fixture()
    def fxt_loss_dyns_dataset(self) -> Dataset:
        subdir = osp.join("algorithms", "noisy_label_detection_det")
        dirpath = get_test_asset_path(subdir)
        return Dataset.import_from(dirpath, format="datumaro")

    @pytest.fixture(params=["cls", "bbox", "centerness"])
    def fxt_analyzer(
        self, fxt_loss_dyns_dataset: Dataset, request: pytest.FixtureRequest
    ) -> LossDynamicsAnalyzer:
        tracking_loss_type = request.param
        return LossDynamicsAnalyzer(fxt_loss_dyns_dataset, tracking_loss_type=tracking_loss_type)


class NoisyLabelCandidateTest:
    def test_eq(self):
        # Same item_id, subset, ann_id
        assert NoisyLabelCandidate("id0", "train", 0, 0, 1.0) == NoisyLabelCandidate(
            "id0", "train", 0, 1, 0.0
        )
        # Not equal ann_id
        assert NoisyLabelCandidate("id0", "train", 1, 0, 1.0) != NoisyLabelCandidate(
            "id0", "train", 0, 1, 0.0
        )

    def test_comp(self):
        # Only depend on the metric value
        assert NoisyLabelCandidate("id0", "train", 0, 0, 1.0) > NoisyLabelCandidate(
            "id11", "train", 1, 1, 0.0
        )
