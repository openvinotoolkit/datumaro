# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import Counter
from unittest import TestCase

import numpy as np
import pytest

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    Polygon,
)
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    AttributeDefinedButNotFound,
    FarFromAttrMean,
    FarFromLabelMean,
    FewSamplesInAttribute,
    FewSamplesInLabel,
    ImbalancedAttribute,
    ImbalancedDistInAttribute,
    ImbalancedDistInLabel,
    ImbalancedLabels,
    InvalidValue,
    LabelDefinedButNotFound,
    MissingAnnotation,
    MissingAttribute,
    MissingLabelCategories,
    MultiLabelAnnotations,
    NegativeLength,
    OnlyOneAttributeValue,
    OnlyOneLabel,
    UndefinedAttribute,
    UndefinedLabel,
)
from datumaro.components.media import Image
from datumaro.components.validator import TaskType
from datumaro.plugins.validators import (
    ClassificationValidator,
    DetectionValidator,
    SegmentationValidator,
    _TaskValidator,
)

from ..requirements import Requirements, mark_requirement


class _TestValidatorBase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Label(
                            1,
                            id=0,
                            attributes={
                                "a": 1,
                                "b": 7,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            id=1,
                            label=0,
                            attributes={
                                "a": 1,
                                "b": 2,
                            },
                        ),
                        Mask(
                            id=2,
                            label=0,
                            attributes={"a": 1, "b": 2},
                            image=np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                ]
                            ),
                        ),
                    ],
                ),
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    annotations=[
                        Label(
                            2,
                            id=0,
                            attributes={
                                "a": 2,
                                "b": 2,
                            },
                        ),
                        Bbox(
                            2,
                            3,
                            1,
                            4,
                            id=1,
                            label=0,
                            attributes={
                                "a": 1,
                                "b": 1,
                            },
                        ),
                        Mask(
                            id=2,
                            label=0,
                            attributes={"a": 1, "b": 1},
                            image=np.array([[1, 1, 1, 1], [0, 0, 0, 0]]),
                        ),
                    ],
                ),
                DatasetItem(id=3),
                DatasetItem(
                    id=4,
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    annotations=[
                        Label(
                            0,
                            id=0,
                            attributes={
                                "b": 4,
                            },
                        ),
                        Label(
                            1,
                            id=1,
                            attributes={
                                "a": 11,
                                "b": 7,
                            },
                        ),
                        Bbox(
                            1,
                            3,
                            2,
                            4,
                            id=2,
                            label=0,
                            attributes={
                                "a": 2,
                                "b": 1,
                            },
                        ),
                        Bbox(
                            3,
                            1,
                            4,
                            2,
                            id=3,
                            label=0,
                            attributes={
                                "a": 2,
                                "b": 2,
                            },
                        ),
                        Polygon(
                            [1, 3, 1, 5, 5, 5, 5, 3],
                            label=0,
                            id=4,
                            attributes={
                                "a": 2,
                                "b": 2,
                            },
                        ),
                        Polygon(
                            [3, 1, 3, 5, 5, 5, 5, 1],
                            label=1,
                            id=5,
                            attributes={
                                "a": 2,
                                "b": 1,
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id=5,
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    annotations=[
                        Label(
                            0,
                            id=0,
                            attributes={
                                "a": 20,
                                "b": 10,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            id=1,
                            label=1,
                            attributes={
                                "a": 1,
                                "b": 1,
                            },
                        ),
                        Polygon(
                            [1, 2, 1, 5, 5, 5, 5, 2],
                            label=1,
                            id=2,
                            attributes={
                                "a": 1,
                                "b": 1,
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id=6,
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    annotations=[
                        Label(
                            1,
                            id=0,
                            attributes={
                                "a": 11,
                                "b": 2,
                                "c": 3,
                            },
                        ),
                        Bbox(
                            2,
                            3,
                            4,
                            1,
                            id=1,
                            label=1,
                            attributes={
                                "a": 2,
                                "b": 2,
                            },
                        ),
                        Mask(
                            id=2,
                            label=1,
                            attributes={"a": 2, "b": 2},
                            image=np.array(
                                [
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                ]
                            ),
                        ),
                    ],
                ),
                DatasetItem(
                    id=7,
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    annotations=[
                        Label(
                            1,
                            id=0,
                            attributes={
                                "a": 1,
                                "b": 2,
                                "c": 5,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            id=1,
                            label=2,
                            attributes={
                                "a": 1,
                                "b": 2,
                            },
                        ),
                        Polygon(
                            [1, 2, 1, 5, 5, 5, 5, 2],
                            label=2,
                            id=2,
                            attributes={
                                "a": 1,
                                "b": 2,
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id=8,
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    annotations=[
                        Label(
                            2,
                            id=0,
                            attributes={
                                "a": 7,
                                "b": 9,
                                "c": 5,
                            },
                        ),
                        Bbox(
                            2,
                            1,
                            3,
                            4,
                            id=1,
                            label=2,
                            attributes={
                                "a": 2,
                                "b": 1,
                            },
                        ),
                        Mask(
                            id=2,
                            label=2,
                            attributes={"a": 2, "b": 1},
                            image=np.array(
                                [
                                    [1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1],
                                ]
                            ),
                        ),
                        Ellipse(
                            2,
                            1,
                            3,
                            4,
                            id=1,
                            label=2,
                            attributes={},
                        ),
                    ],
                ),
            ],
            categories=[
                [
                    f"label_{i}",
                    None,
                    {
                        "a",
                        "b",
                    },
                ]
                for i in range(2)
            ],
        )


class TestBaseValidator(_TestValidatorBase):
    @classmethod
    def setUpClass(cls):
        cls.validator = _TaskValidator(
            task_type=TaskType.classification,
            few_samples_thr=1,
            imbalance_ratio_thr=50,
            far_from_mean_thr=5.0,
            dominance_ratio_thr=0.8,
            topk_bins=0.1,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_has_enum_entries_in_environment(self):
        env = Environment()
        for key in TaskType.__members__:
            self.assertIn(key, env.validators._items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_generate_reports(self):
        with self.assertRaises(NotImplementedError):
            self.validator.generate_reports({})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_missing_label_categories(self):
        stats = {"label_distribution": {"defined_labels": {}}}

        actual_reports = self.validator._check_missing_label_categories(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MissingLabelCategories)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_missing_attribute(self):
        label_name = "unit"
        attr_name = "test"
        attr_dets = {"items_missing_attribute": [(1, "unittest")]}

        actual_reports = self.validator._check_missing_attribute(label_name, attr_name, attr_dets)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MissingAttribute)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_undefined_label(self):
        label_name = "cat0"
        item_id = 1
        item_subset = "unittest"
        label_stats = {label_name: {"items_with_undefined_label": [(item_id, item_subset)]}}
        stats = {"label_distribution": {"undefined_labels": label_stats}}

        actual_reports = self.validator._check_undefined_label(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], UndefinedLabel)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_undefined_attribute(self):
        label_name = "unit"
        attr_name = "test"
        attr_dets = {"items_with_undefined_attr": [(1, "unittest")]}

        actual_reports = self.validator._check_undefined_attribute(label_name, attr_name, attr_dets)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], UndefinedAttribute)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_label_defined_but_not_found(self):
        stats = {"label_distribution": {"defined_labels": {"unittest": 0}}}

        actual_reports = self.validator._check_label_defined_but_not_found(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], LabelDefinedButNotFound)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_attribute_defined_but_not_found(self):
        label_name = "unit"
        attr_stats = {"test": {"distribution": {}}}

        actual_reports = self.validator._check_attribute_defined_but_not_found(
            label_name, attr_stats
        )

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], AttributeDefinedButNotFound)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_only_one_label(self):
        stats = {"label_distribution": {"defined_labels": {"unit": 1, "test": 0}}}

        actual_reports = self.validator._check_only_one_label(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], OnlyOneLabel)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_only_one_attribute(self):
        label_name = "unit"
        attr_name = "test"
        attr_dets = {"distribution": {"mock": 1}}

        actual_reports = self.validator._check_only_one_attribute(label_name, attr_name, attr_dets)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], OnlyOneAttributeValue)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_few_samples_in_label(self):
        with self.subTest("Few Samples"):
            stats = {
                "label_distribution": {"defined_labels": {"unit": self.validator.few_samples_thr}}
            }

            actual_reports = self.validator._check_few_samples_in_label(stats)

            self.assertTrue(len(actual_reports) == 1)
            self.assertIsInstance(actual_reports[0], FewSamplesInLabel)

        with self.subTest("No Few Samples Warning"):
            stats = {
                "label_distribution": {
                    "defined_labels": {"unit": self.validator.few_samples_thr + 1}
                }
            }

            actual_reports = self.validator._check_few_samples_in_label(stats)

            self.assertTrue(len(actual_reports) == 0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_few_samples_in_attribute(self):
        label_name = "unit"
        attr_name = "test"

        with self.subTest("Few Samples"):
            attr_dets = {"distribution": {"mock": self.validator.few_samples_thr}}

            actual_reports = self.validator._check_few_samples_in_attribute(
                label_name, attr_name, attr_dets
            )

            self.assertTrue(len(actual_reports) == 1)
            self.assertIsInstance(actual_reports[0], FewSamplesInAttribute)

        with self.subTest("No Few Samples Warning"):
            attr_dets = {"distribution": {"mock": self.validator.few_samples_thr + 1}}

            actual_reports = self.validator._check_few_samples_in_attribute(
                label_name, attr_name, attr_dets
            )

            self.assertTrue(len(actual_reports) == 0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_imbalanced_labels(self):
        with self.subTest("Imbalance"):
            stats = {
                "label_distribution": {
                    "defined_labels": {"unit": self.validator.imbalance_ratio_thr, "test": 1}
                }
            }

            actual_reports = self.validator._check_imbalanced_labels(stats)

            self.assertTrue(len(actual_reports) == 1)
            self.assertIsInstance(actual_reports[0], ImbalancedLabels)

        with self.subTest("No Imbalance Warning"):
            stats = {
                "label_distribution": {
                    "defined_labels": {"unit": self.validator.imbalance_ratio_thr - 1, "test": 1}
                }
            }

            actual_reports = self.validator._check_imbalanced_labels(stats)

            self.assertTrue(len(actual_reports) == 0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_imbalanced_attribute(self):
        label_name = "unit"
        attr_name = "test"

        with self.subTest("Imbalance"):
            attr_dets = {"distribution": {"mock": self.validator.imbalance_ratio_thr, "mock_1": 1}}

            actual_reports = self.validator._check_imbalanced_attribute(
                label_name, attr_name, attr_dets
            )

            self.assertTrue(len(actual_reports) == 1)
            self.assertIsInstance(actual_reports[0], ImbalancedAttribute)

        with self.subTest("No Imbalance Warning"):
            attr_dets = {
                "distribution": {"mock": self.validator.imbalance_ratio_thr - 1, "mock_1": 1}
            }

            actual_reports = self.validator._check_imbalanced_attribute(
                label_name, attr_name, attr_dets
            )

            self.assertTrue(len(actual_reports) == 0)


class TestClassificationValidator(_TestValidatorBase):
    @classmethod
    def setUpClass(cls):
        cls.validator = ClassificationValidator(
            few_samples_thr=1,
            imbalance_ratio_thr=50,
            far_from_mean_thr=5.0,
            dominance_ratio_thr=0.8,
            topk_bins=0.1,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_missing_label_annotation(self):
        stats = {"items_missing_annotation": [(1, "unittest")]}

        actual_reports = self.validator._check_missing_annotation(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MissingAnnotation)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_multi_label_annotations(self):
        stats = {"items_with_multiple_labels": [(1, "unittest")]}

        actual_reports = self.validator._check_multi_label_annotations(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MultiLabelAnnotations)


class TestDetectionValidator(_TestValidatorBase):
    @classmethod
    def setUpClass(cls):
        cls.validator = DetectionValidator(
            few_samples_thr=1,
            imbalance_ratio_thr=50,
            far_from_mean_thr=5.0,
            dominance_ratio_thr=0.8,
            topk_bins=0.1,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_imbalanced_dist_in_label(self):
        label_name = "unittest"
        most = int(self.validator.dominance_thr * 100)
        rest = 100 - most

        with self.subTest("Imbalanced"):
            bbox_label_stats = {"x": {"histogram": {"counts": [most, rest]}}}
            reports = self.validator._check_imbalanced_dist_in_label(label_name, bbox_label_stats)

            self.assertTrue(len(reports) == 1)
            self.assertIsInstance(reports[0], ImbalancedDistInLabel)

        with self.subTest("No Imbalanced Warning"):
            bbox_label_stats = {"x": {"histogram": {"counts": [most - 1, rest]}}}
            reports = self.validator._check_imbalanced_dist_in_label(label_name, bbox_label_stats)

            self.assertTrue(len(reports) == 0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_imbalanced_dist_in_attr(self):
        label_name = "unit"
        attr_name = "test"
        most = int(self.validator.dominance_thr * 100)
        rest = 100 - most

        with self.subTest("Imbalanced"):
            bbox_attr_stats = {"mock": {"x": {"histogram": {"counts": [most, rest]}}}}

            reports = self.validator._check_imbalanced_dist_in_attr(
                label_name, attr_name, bbox_attr_stats
            )

            self.assertTrue(len(reports) == 1)
            self.assertIsInstance(reports[0], ImbalancedDistInAttribute)

        with self.subTest("No Imbalanced Warning"):
            bbox_attr_stats = {"mock": {"x": {"histogram": {"counts": [most - 1, rest]}}}}

            reports = self.validator._check_imbalanced_dist_in_attr(
                label_name, attr_name, bbox_attr_stats
            )

            self.assertTrue(len(reports) == 0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_missing_bbox_annotation(self):
        stats = {"items_missing_annotation": [(1, "unittest")]}

        actual_reports = self.validator._check_missing_annotation(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MissingAnnotation)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_negative_length(self):
        stats = {"items_with_negative_length": {("1", "unittest"): {1: {"x": -1}}}}

        actual_reports = self.validator._check_negative_length(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], NegativeLength)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_invalid_value(self):
        stats = {"items_with_invalid_value": {("1", "unittest"): {1: ["x"]}}}

        actual_reports = self.validator._check_invalid_value(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], InvalidValue)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_far_from_label_mean(self):
        label_name = "unittest"
        bbox_label_stats = {
            "w": {
                "items_far_from_mean": {("1", "unittest"): {1: 100}},
                "mean": 0,
            }
        }

        actual_reports = self.validator._check_far_from_label_mean(label_name, bbox_label_stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], FarFromLabelMean)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_far_from_attr_mean(self):
        label_name = "unit"
        attr_name = "test"
        bbox_attr_stats = {
            "mock": {
                "w": {
                    "items_far_from_mean": {("1", "unittest"): {1: 100}},
                    "mean": 0,
                }
            }
        }

        actual_reports = self.validator._check_far_from_attr_mean(
            label_name, attr_name, bbox_attr_stats
        )

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], FarFromAttrMean)


class TestSegmentationValidator(_TestValidatorBase):
    @classmethod
    def setUpClass(cls):
        cls.validator = SegmentationValidator(
            few_samples_thr=1,
            imbalance_ratio_thr=50,
            far_from_mean_thr=5.0,
            dominance_ratio_thr=0.8,
            topk_bins=0.1,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_imbalanced_dist_in_label(self):
        label_name = "unittest"
        most = int(self.validator.dominance_thr * 100)
        rest = 100 - most

        with self.subTest("Imbalanced"):
            mask_label_stats = {"area": {"histogram": {"counts": [most, rest]}}}
            reports = self.validator._check_imbalanced_dist_in_label(label_name, mask_label_stats)

            self.assertTrue(len(reports) == 1)
            self.assertIsInstance(reports[0], ImbalancedDistInLabel)

        with self.subTest("No Imbalanced Warning"):
            mask_label_stats = {"area": {"histogram": {"counts": [most - 1, rest]}}}
            reports = self.validator._check_imbalanced_dist_in_label(label_name, mask_label_stats)

            self.assertTrue(len(reports) == 0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_imbalanced_dist_in_attr(self):
        label_name = "unit"
        attr_name = "test"
        most = int(self.validator.dominance_thr * 100)
        rest = 100 - most

        with self.subTest("Imbalanced"):
            mask_attr_stats = {"mock": {"x": {"histogram": {"counts": [most, rest]}}}}

            reports = self.validator._check_imbalanced_dist_in_attr(
                label_name, attr_name, mask_attr_stats
            )

            self.assertTrue(len(reports) == 1)
            self.assertIsInstance(reports[0], ImbalancedDistInAttribute)

        with self.subTest("No Imbalanced Warning"):
            mask_attr_stats = {"mock": {"x": {"histogram": {"counts": [most - 1, rest]}}}}

            reports = self.validator._check_imbalanced_dist_in_attr(
                label_name, attr_name, mask_attr_stats
            )

            self.assertTrue(len(reports) == 0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_missing_mask_annotation(self):
        stats = {"items_missing_annotation": [(1, "unittest")]}

        actual_reports = self.validator._check_missing_annotation(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MissingAnnotation)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_invalid_value(self):
        stats = {"items_with_invalid_value": {("1", "unittest"): {1: ["x"]}}}

        actual_reports = self.validator._check_invalid_value(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], InvalidValue)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_far_from_label_mean(self):
        label_name = "unittest"
        mask_label_stats = {
            "w": {
                "items_far_from_mean": {("1", "unittest"): {1: 100}},
                "mean": 0,
            }
        }

        actual_reports = self.validator._check_far_from_label_mean(label_name, mask_label_stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], FarFromLabelMean)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_check_far_from_attr_mean(self):
        label_name = "unit"
        attr_name = "test"
        mask_attr_stats = {
            "mock": {
                "w": {
                    "items_far_from_mean": {("1", "unittest"): {1: 100}},
                    "mean": 0,
                }
            }
        }

        actual_reports = self.validator._check_far_from_attr_mean(
            label_name, attr_name, mask_attr_stats
        )

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], FarFromAttrMean)


class TestValidateAnnotations(_TestValidatorBase):
    extra_args = {
        "few_samples_thr": 1,
        "imbalance_ratio_thr": 50,
        "far_from_mean_thr": 5.0,
        "dominance_ratio_thr": 0.8,
        "topk_bins": 0.1,
    }

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_validate_annotations_classification(self):
        validator = ClassificationValidator(**self.extra_args)
        actual_results = validator.validate(self.dataset)

        with self.subTest("Test of statistics", i=0):
            actual_stats = actual_results["statistics"]
            self.assertEqual(actual_stats["total_ann_count"], 8)
            self.assertEqual(len(actual_stats["items_missing_annotation"]), 1)
            self.assertEqual(len(actual_stats["items_with_multiple_labels"]), 1)

            label_dist = actual_stats["label_distribution"]
            defined_label_dist = label_dist["defined_labels"]
            self.assertEqual(len(defined_label_dist), 2)
            self.assertEqual(sum(defined_label_dist.values()), 6)

            undefined_label_dist = label_dist["undefined_labels"]
            undefined_label_stats = undefined_label_dist[2]
            self.assertEqual(len(undefined_label_dist), 1)
            self.assertEqual(undefined_label_stats["count"], 2)
            self.assertEqual(len(undefined_label_stats["items_with_undefined_label"]), 2)

            attr_stats = actual_stats["attribute_distribution"]
            defined_attr_dets = attr_stats["defined_attributes"]["label_0"]["a"]
            self.assertEqual(len(defined_attr_dets["items_missing_attribute"]), 1)
            self.assertEqual(defined_attr_dets["distribution"], {"20": 1})

            undefined_attr_dets = attr_stats["undefined_attributes"][2]["c"]
            self.assertEqual(len(undefined_attr_dets["items_with_undefined_attr"]), 1)
            self.assertEqual(undefined_attr_dets["distribution"], {"5": 1})

        with self.subTest("Test of validation reports", i=1):
            actual_reports = actual_results["validation_reports"]
            report_types = [r["anomaly_type"] for r in actual_reports]
            report_count_by_type = Counter(report_types)

            self.assertEqual(len(actual_reports), 16)
            self.assertEqual(report_count_by_type["UndefinedAttribute"], 7)
            self.assertEqual(report_count_by_type["FewSamplesInAttribute"], 3)
            self.assertEqual(report_count_by_type["UndefinedLabel"], 2)
            self.assertEqual(report_count_by_type["MissingAnnotation"], 1)
            self.assertEqual(report_count_by_type["MultiLabelAnnotations"], 1)
            self.assertEqual(report_count_by_type["OnlyOneAttributeValue"], 1)
            self.assertEqual(report_count_by_type["MissingAttribute"], 1)

        with self.subTest("Test of summary", i=2):
            actual_summary = actual_results["summary"]
            expected_summary = {"errors": 10, "infos": 4, "warnings": 2}

            self.assertEqual(actual_summary, expected_summary)

    @pytest.mark.new
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_validate_multilabel_annotations_classification(self):
        label_cat = LabelCategories.from_iterable(["car", "bicycle", "dog", "cat", "plate", "pan"])
        label_cat.add_label_group("vehicle", ["car", "bicycle"], group_type=0)
        label_cat.add_label_group("animal", ["dog", "cat"], group_type=0)
        label_cat.add_label_group("kithen", ["plate", "pan"], group_type=1)

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    annotations=[
                        Label(
                            id=0,
                            label=0,
                        ),
                        Label(
                            id=1,
                            label=1,
                        ),
                    ],
                ),
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(
                            id=0,
                            label=2,
                        ),
                        Label(
                            id=1,
                            label=3,
                        ),
                    ],
                ),
                DatasetItem(
                    id=2,
                    annotations=[
                        Label(
                            id=0,
                            label=0,
                        ),
                        Label(
                            id=1,
                            label=2,
                        ),
                    ],
                ),
                DatasetItem(
                    id=3,
                    annotations=[
                        Label(
                            id=0,
                            label=1,
                        ),
                        Label(
                            id=1,
                            label=3,
                        ),
                    ],
                ),
                DatasetItem(
                    id=4,
                    annotations=[
                        Label(
                            id=0,
                            label=4,
                        ),
                        Label(
                            id=1,
                            label=5,
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: label_cat,
            },
        )

        validator = ClassificationValidator(**self.extra_args)
        actual_results = validator.validate(dataset)

        with self.subTest("Test of validation reports", i=1):
            actual_reports = actual_results["validation_reports"]
            multilabel_ids = []
            for r in actual_reports:
                if r["anomaly_type"] == "MultiLabelAnnotations":
                    multilabel_ids.append(r["item_id"])

            self.assertEqual(multilabel_ids, ["0", "1"])

        with self.subTest("Test of summary", i=2):
            actual_summary = actual_results["summary"]
            expected_summary = {"errors": 2, "warnings": 0, "infos": 2}

            self.assertEqual(actual_summary, expected_summary)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_validate_annotations_detection(self):
        validator = DetectionValidator(**self.extra_args)
        actual_results = validator.validate(self.dataset)

        with self.subTest("Test of statistics", i=0):
            actual_stats = actual_results["statistics"]
            self.assertEqual(actual_stats["total_ann_count"], 8)
            self.assertEqual(len(actual_stats["items_missing_annotation"]), 1)
            self.assertEqual(actual_stats["items_with_negative_length"], {})
            self.assertEqual(actual_stats["items_with_invalid_value"], {})

            bbox_dist_by_label = actual_stats["point_distribution_in_label"]
            label_prop_stats = bbox_dist_by_label["label_1"]["width"]
            self.assertEqual(label_prop_stats["items_far_from_mean"], {})
            self.assertEqual(label_prop_stats["mean"], 3.5)
            self.assertEqual(label_prop_stats["stdev"], 0.5)
            self.assertEqual(label_prop_stats["min"], 3.0)
            self.assertEqual(label_prop_stats["max"], 4.0)
            self.assertEqual(label_prop_stats["median"], 3.5)

            bbox_dist_by_attr = actual_stats["point_distribution_in_attribute"]
            attr_prop_stats = bbox_dist_by_attr["label_0"]["a"]["1"]["width"]
            self.assertEqual(attr_prop_stats["items_far_from_mean"], {})
            self.assertEqual(attr_prop_stats["mean"], 2.0)
            self.assertEqual(attr_prop_stats["stdev"], 1.0)
            self.assertEqual(attr_prop_stats["min"], 1.0)
            self.assertEqual(attr_prop_stats["max"], 3.0)
            self.assertEqual(attr_prop_stats["median"], 2.0)

            bbox_dist_item = actual_stats["point_distribution_in_dataset_item"]
            self.assertEqual(sum(bbox_dist_item.values()), 8)

        with self.subTest("Test of validation reports", i=1):
            actual_reports = actual_results["validation_reports"]
            report_types = [r["anomaly_type"] for r in actual_reports]
            count_by_type = Counter(report_types)

            self.assertEqual(len(actual_reports), 45)
            self.assertEqual(count_by_type["ImbalancedDistInAttribute"], 32)
            self.assertEqual(count_by_type["FewSamplesInAttribute"], 4)
            self.assertEqual(count_by_type["UndefinedAttribute"], 4)
            self.assertEqual(count_by_type["ImbalancedDistInLabel"], 2)
            self.assertEqual(count_by_type["UndefinedLabel"], 2)
            self.assertEqual(count_by_type["MissingAnnotation"], 1)

        with self.subTest("Test of summary", i=2):
            actual_summary = actual_results["summary"]
            expected_summary = {"errors": 6, "infos": 38, "warnings": 1}

            self.assertEqual(actual_summary, expected_summary)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_validate_annotations_segmentation(self):
        validator = SegmentationValidator(**self.extra_args)
        actual_results = validator.validate(self.dataset)

        with self.subTest("Test of statistics", i=0):
            actual_stats = actual_results["statistics"]
            self.assertEqual(actual_stats["total_ann_count"], 9)
            self.assertEqual(len(actual_stats["items_missing_annotation"]), 1)
            self.assertEqual(actual_stats["items_with_invalid_value"], {})

            mask_dist_by_label = actual_stats["point_distribution_in_label"]
            label_prop_stats = mask_dist_by_label["label_1"]["area"]
            self.assertEqual(label_prop_stats["items_far_from_mean"], {})
            areas = [12, 4, 8]
            self.assertEqual(label_prop_stats["mean"], np.mean(areas))
            self.assertEqual(label_prop_stats["stdev"], np.std(areas))
            self.assertEqual(label_prop_stats["min"], np.min(areas))
            self.assertEqual(label_prop_stats["max"], np.max(areas))
            self.assertEqual(label_prop_stats["median"], np.median(areas))

            mask_dist_by_attr = actual_stats["point_distribution_in_attribute"]
            attr_prop_stats = mask_dist_by_attr["label_0"]["a"]["1"]["area"]
            areas = [12, 4]
            self.assertEqual(attr_prop_stats["items_far_from_mean"], {})
            self.assertEqual(attr_prop_stats["mean"], np.mean(areas))
            self.assertEqual(attr_prop_stats["stdev"], np.std(areas))
            self.assertEqual(attr_prop_stats["min"], np.min(areas))
            self.assertEqual(attr_prop_stats["max"], np.max(areas))
            self.assertEqual(attr_prop_stats["median"], np.median(areas))

            mask_dist_item = actual_stats["point_distribution_in_dataset_item"]
            self.assertEqual(sum(mask_dist_item.values()), 9)

        with self.subTest("Test of validation reports", i=1):
            actual_reports = actual_results["validation_reports"]
            report_types = [r["anomaly_type"] for r in actual_reports]
            count_by_type = Counter(report_types)

            self.assertEqual(len(actual_reports), 25)
            self.assertEqual(count_by_type["ImbalancedDistInLabel"], 0)
            self.assertEqual(count_by_type["ImbalancedDistInAttribute"], 13)
            self.assertEqual(count_by_type["MissingAnnotation"], 1)
            self.assertEqual(count_by_type["UndefinedLabel"], 3)
            self.assertEqual(count_by_type["FewSamplesInAttribute"], 4)
            self.assertEqual(count_by_type["UndefinedAttribute"], 4)

        with self.subTest("Test of summary", i=2):
            actual_summary = actual_results["summary"]
            expected_summary = {"errors": 7, "infos": 17, "warnings": 1}

            self.assertEqual(actual_summary, expected_summary)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_validate_invalid_dataset_type(self):
        with self.assertRaises(TypeError):
            validator = ClassificationValidator(**self.extra_args)
            validator.validate(object())
