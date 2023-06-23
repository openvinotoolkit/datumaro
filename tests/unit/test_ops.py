import os.path as osp
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pytest

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetItem
from datumaro.components.media import Image, MultiframeImage, PointCloud
from datumaro.components.merge.exact_merge import ExactMerge
from datumaro.components.merge.intersect_merge import IntersectMerge
from datumaro.components.merge.union_merge import UnionMerge
from datumaro.components.operations import compute_ann_statistics, find_unique_images, mean_std
from datumaro.errors import (
    ConflictingCategoriesError,
    FailedAttrVotingError,
    MismatchingMediaPathError,
    NoMatchingAnnError,
    NoMatchingItemError,
    WrongGroupError,
)

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets


class TestOperations(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_mean_std(self):
        expected_mean = [100, 50, 150]
        expected_std = [20, 50, 10]

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=i,
                    media=Image.from_numpy(
                        data=np.random.normal(expected_mean, expected_std, size=(h, w, 3))
                    ),
                )
                for i, (w, h) in enumerate([(3000, 100), (800, 600), (400, 200), (700, 300)])
            ]
        )

        actual_mean, actual_std = mean_std(dataset)

        for em, am in zip(expected_mean, actual_mean):
            self.assertAlmostEqual(em, am, places=0)
        for estd, astd in zip(expected_std, actual_std):
            self.assertAlmostEqual(estd, astd, places=0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_stats(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Caption("hello"),
                        Caption("world"),
                        Label(
                            2,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            2,
                            2,
                            label=2,
                            attributes={
                                "score": 0.5,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            2,
                            attributes={
                                "x": 1,
                                "y": "3",
                                "occluded": True,
                            },
                        ),
                        Points([1, 2, 2, 0, 1, 1], label=0),
                        Mask(
                            label=3,
                            image=np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
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
                            attributes={
                                "x": 2,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            2,
                            2,
                            label=3,
                            attributes={
                                "score": 0.5,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            2,
                            attributes={
                                "x": 2,
                                "y": "3",
                                "occluded": False,
                            },
                        ),
                        Ellipse(
                            5,
                            6,
                            2,
                            2,
                            attributes={
                                "x": 2,
                                "y": "3",
                                "occluded": False,
                            },
                        ),
                    ],
                ),
                DatasetItem(id=3),
                DatasetItem(id="2.2", media=Image.from_numpy(data=np.ones((2, 4, 3)))),
            ],
            categories=["label_%s" % i for i in range(4)],
        )

        expected = {
            "images count": 4,
            "annotations count": 11,
            "unannotated images count": 2,
            "unannotated images": ["3", "2.2"],
            "annotations by type": {
                "label": {
                    "count": 2,
                },
                "polygon": {
                    "count": 0,
                },
                "polyline": {
                    "count": 0,
                },
                "bbox": {
                    "count": 4,
                },
                "mask": {
                    "count": 1,
                },
                "points": {
                    "count": 1,
                },
                "caption": {
                    "count": 2,
                },
                "cuboid_3d": {"count": 0},
                "super_resolution_annotation": {"count": 0},
                "depth_annotation": {"count": 0},
                "ellipse": {"count": 1},
                "hash_key": {"count": 0},
                "feature_vector": {"count": 0},
                "unknown": {"count": 0},
            },
            "annotations": {
                "labels": {
                    "count": 6,
                    "distribution": {
                        "label_0": [1, 1 / 6],
                        "label_1": [0, 0.0],
                        "label_2": [3, 3 / 6],
                        "label_3": [2, 2 / 6],
                    },
                    "attributes": {
                        "x": {
                            "count": 2,  # annotations with no label are skipped
                            "values count": 2,
                            "values present": ["1", "2"],
                            "distribution": {
                                "1": [1, 1 / 2],
                                "2": [1, 1 / 2],
                            },
                        },
                        "y": {
                            "count": 2,  # annotations with no label are skipped
                            "values count": 1,
                            "values present": ["2"],
                            "distribution": {
                                "2": [2, 2 / 2],
                            },
                        },
                        # must not include "special" attributes like "occluded"
                    },
                },
                "segments": {
                    "avg. area": (4 * 2 + 9 * 1) / 3,
                    "area distribution": [
                        {"min": 4.0, "max": 4.5, "count": 2, "percent": 2 / 3},
                        {"min": 4.5, "max": 5.0, "count": 0, "percent": 0.0},
                        {"min": 5.0, "max": 5.5, "count": 0, "percent": 0.0},
                        {"min": 5.5, "max": 6.0, "count": 0, "percent": 0.0},
                        {"min": 6.0, "max": 6.5, "count": 0, "percent": 0.0},
                        {"min": 6.5, "max": 7.0, "count": 0, "percent": 0.0},
                        {"min": 7.0, "max": 7.5, "count": 0, "percent": 0.0},
                        {"min": 7.5, "max": 8.0, "count": 0, "percent": 0.0},
                        {"min": 8.0, "max": 8.5, "count": 0, "percent": 0.0},
                        {"min": 8.5, "max": 9.0, "count": 1, "percent": 1 / 3},
                    ],
                    "pixel distribution": {
                        "label_0": [0, 0.0],
                        "label_1": [0, 0.0],
                        "label_2": [4, 4 / 17],
                        "label_3": [13, 13 / 17],
                    },
                },
            },
        }

        actual = compute_ann_statistics(dataset)

        self.assertEqual(expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_stats_with_empty_dataset(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1),
                DatasetItem(id=3),
            ],
            categories=["label_%s" % i for i in range(4)],
        )

        expected = {
            "images count": 2,
            "annotations count": 0,
            "unannotated images count": 2,
            "unannotated images": ["1", "3"],
            "annotations by type": {
                "label": {
                    "count": 0,
                },
                "polygon": {
                    "count": 0,
                },
                "polyline": {
                    "count": 0,
                },
                "bbox": {
                    "count": 0,
                },
                "mask": {
                    "count": 0,
                },
                "points": {
                    "count": 0,
                },
                "caption": {
                    "count": 0,
                },
                "cuboid_3d": {"count": 0},
                "super_resolution_annotation": {"count": 0},
                "depth_annotation": {"count": 0},
                "ellipse": {"count": 0},
                "hash_key": {"count": 0},
                "feature_vector": {"count": 0},
                "unknown": {"count": 0},
            },
            "annotations": {
                "labels": {
                    "count": 0,
                    "distribution": {
                        "label_0": [0, 0.0],
                        "label_1": [0, 0.0],
                        "label_2": [0, 0.0],
                        "label_3": [0, 0.0],
                    },
                    "attributes": {},
                },
                "segments": {
                    "avg. area": 0.0,
                    "area distribution": [],
                    "pixel distribution": {
                        "label_0": [0, 0.0],
                        "label_1": [0, 0.0],
                        "label_2": [0, 0.0],
                        "label_3": [0, 0.0],
                    },
                },
            },
        }

        actual = compute_ann_statistics(dataset)

        self.assertEqual(expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_unique_image_count(self):
        expected = {
            frozenset([("1", "a"), ("1", "b")]),
            frozenset([("2", DEFAULT_SUBSET_NAME), ("3", DEFAULT_SUBSET_NAME)]),
            frozenset([("4", DEFAULT_SUBSET_NAME)]),
        }

        dataset = Dataset.from_iterable(
            [
                # no image data, but the same path
                DatasetItem(1, subset="a", media=Image.from_file(path="1.jpg")),
                DatasetItem(1, subset="b", media=Image.from_file(path="1.jpg")),
                # same images
                DatasetItem(2, media=Image.from_numpy(data=np.ones((5, 5, 3)))),
                DatasetItem(3, media=Image.from_numpy(data=np.ones((5, 5, 3)))),
                # no image is always a unique image
                DatasetItem(4),
            ]
        )

        groups = find_unique_images(dataset)

        self.assertEqual(expected, set(frozenset(s) for s in groups.values()))


class TestMultimerge(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_match_items(self):
        # items 1 and 3 are unique, item 2 is common and should be merged

        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                    ],
                ),
                DatasetItem(
                    2,
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    2,
                    annotations=[
                        Label(1),
                    ],
                ),
                DatasetItem(
                    3,
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        source2 = Dataset.from_iterable(
            [
                DatasetItem(2, annotations=[Label(0), Bbox(1, 2, 3, 4)]),
            ],
            categories=["a", "b"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0, attributes={"score": 1 / 3}),
                    ],
                ),
                DatasetItem(
                    2,
                    annotations=[
                        Label(0, attributes={"score": 2 / 3}),
                        Label(1, attributes={"score": 1 / 3}),
                        Bbox(1, 2, 3, 4, attributes={"score": 1.0}),
                    ],
                ),
                DatasetItem(
                    3,
                    annotations=[
                        Label(0, attributes={"score": 1 / 3}),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        merger = IntersectMerge()
        merged = merger(source0, source1, source2)

        compare_datasets(self, expected, merged)
        self.assertEqual(
            [
                NoMatchingItemError(item_id=("1", DEFAULT_SUBSET_NAME), sources={1, 2}),
                NoMatchingItemError(item_id=("3", DEFAULT_SUBSET_NAME), sources={0, 2}),
            ],
            sorted(
                (e for e in merger.errors if isinstance(e, NoMatchingItemError)),
                key=lambda e: e.item_id,
            ),
        )
        self.assertEqual(
            [
                NoMatchingAnnError(
                    item_id=("2", DEFAULT_SUBSET_NAME),
                    sources={0, 1},
                    ann=source2.get("2").annotations[1],
                ),
            ],
            sorted(
                (e for e in merger.errors if isinstance(e, NoMatchingAnnError)),
                key=lambda e: e.item_id,
            ),
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_match_shapes(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        # unique
                        Bbox(1, 2, 3, 4, label=1),
                        # common
                        Mask(
                            label=2,
                            z_order=2,
                            image=np.array(
                                [
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [1, 1, 1, 0],
                                    [1, 1, 1, 0],
                                ]
                            ),
                        ),
                        Polygon([1, 0, 3, 2, 1, 2]),
                        # an instance with keypoints
                        Bbox(4, 5, 2, 4, label=2, z_order=1, group=1),
                        Points([5, 6], label=0, group=1),
                        Points([6, 8], label=1, group=1),
                        PolyLine([1, 1, 2, 1, 3, 1]),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        # common
                        Mask(
                            label=2,
                            image=np.array(
                                [
                                    [0, 0, 0, 0],
                                    [0, 1, 1, 1],
                                    [0, 1, 1, 1],
                                    [0, 1, 1, 1],
                                ]
                            ),
                        ),
                        Polygon([0, 2, 2, 0, 2, 1]),
                        # an instance with keypoints
                        Bbox(4, 4, 2, 5, label=2, z_order=1, group=2),
                        Points([5.5, 6.5], label=0, group=2),
                        Points([6, 8], label=1, group=2),
                        PolyLine([1, 1.5, 2, 1.5]),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        source2 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        # common
                        Mask(
                            label=2,
                            z_order=3,
                            image=np.array(
                                [
                                    [0, 0, 1, 1],
                                    [0, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 0],
                                ]
                            ),
                        ),
                        Polygon([3, 1, 2, 2, 0, 1]),
                        # an instance with keypoints, one is missing
                        Bbox(3, 6, 2, 3, label=2, z_order=4, group=3),
                        Points([4.5, 5.5], label=0, group=3),
                        PolyLine([1, 1.25, 3, 1, 4, 2]),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        # unique
                        Bbox(1, 2, 3, 4, label=1),
                        # common
                        # nearest to mean bbox
                        Mask(
                            label=2,
                            z_order=3,
                            image=np.array(
                                [
                                    [0, 0, 0, 0],
                                    [0, 1, 1, 1],
                                    [0, 1, 1, 1],
                                    [0, 1, 1, 1],
                                ]
                            ),
                        ),
                        Polygon([1, 0, 3, 2, 1, 2]),
                        # an instance with keypoints
                        Bbox(4, 5, 2, 4, label=2, z_order=4, group=1),
                        Points([5, 6], label=0, group=1),
                        Points([6, 8], label=1, group=1),
                        PolyLine([1, 1.25, 3, 1, 4, 2]),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        merger = IntersectMerge(conf={"quorum": 1, "pairwise_dist": 0.1})
        merged = merger(source0, source1, source2)

        compare_datasets(self, expected, merged, ignored_attrs={"score"})
        self.assertEqual(
            [
                NoMatchingAnnError(
                    item_id=("1", DEFAULT_SUBSET_NAME),
                    sources={2},
                    ann=source0.get("1").annotations[5],
                ),
                NoMatchingAnnError(
                    item_id=("1", DEFAULT_SUBSET_NAME),
                    sources={1, 2},
                    ann=source0.get("1").annotations[0],
                ),
            ],
            sorted(
                (e for e in merger.errors if isinstance(e, NoMatchingAnnError)),
                key=lambda e: len(e.sources),
            ),
        )

    @mark_requirement(Requirements.DATUM_BUG_219)
    def test_can_match_lines_when_line_not_approximated(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        PolyLine([1, 1, 2, 1, 3, 5, 5, 5, 8, 3]),
                    ],
                ),
            ]
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        PolyLine([1, 1, 8, 3]),
                    ],
                ),
            ]
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        PolyLine([1, 1, 2, 1, 3, 5, 5, 5, 8, 3]),
                    ],
                ),
            ],
            categories=[],
        )

        merger = IntersectMerge(conf={"quorum": 1, "pairwise_dist": 0.1})
        merged = merger(source0, source1)

        compare_datasets(self, expected, merged, ignored_attrs={"score"})
        self.assertEqual(0, len(merger.errors))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_attributes(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(
                            2,
                            attributes={
                                "unique": 1,
                                "common_under_quorum": 2,
                                "common_over_quorum": 3,
                                "ignored": "q",
                            },
                        ),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(
                            2,
                            attributes={
                                "common_under_quorum": 2,
                                "common_over_quorum": 3,
                                "ignored": "q",
                            },
                        ),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        source2 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(
                            2,
                            attributes={
                                "common_over_quorum": 3,
                                "ignored": "q",
                            },
                        ),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(2, attributes={"common_over_quorum": 3}),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        merger = IntersectMerge(conf={"quorum": 3, "ignored_attributes": {"ignored"}})
        merged = merger(source0, source1, source2)

        compare_datasets(self, expected, merged, ignored_attrs={"score"})
        self.assertEqual(2, len([e for e in merger.errors if isinstance(e, FailedAttrVotingError)]))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_group_checks(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Bbox(0, 0, 0, 0, label=0, group=1),  # misses an optional label
                        Bbox(0, 0, 0, 0, label=1, group=1),
                        Bbox(0, 0, 0, 0, label=2, group=2),  # misses a mandatory label - error
                        Bbox(0, 0, 0, 0, label=2, group=2),
                        Bbox(0, 0, 0, 0, label=4),  # misses an optional label
                        Bbox(0, 0, 0, 0, label=5),  # misses a mandatory label - error
                        Bbox(0, 0, 0, 0, label=0),  # misses a mandatory label - error
                        Bbox(0, 0, 0, 0, label=3),  # not listed - not checked
                    ],
                ),
            ],
            categories=["a", "a_g1", "a_g2_opt", "b", "c", "c_g1_opt"],
        )

        merger = IntersectMerge(conf={"groups": [["a", "a_g1", "a_g2_opt?"], ["c", "c_g1_opt?"]]})
        merger(dataset, dataset)

        self.assertEqual(
            3, len([e for e in merger.errors if isinstance(e, WrongGroupError)]), merger.errors
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_classes(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                        Label(1),
                        Bbox(0, 0, 1, 1, label=1),
                        Ellipse(0, 0, 1, 1, label=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                        Label(1),
                        Bbox(0, 0, 1, 1, label=0),
                        Bbox(0, 0, 1, 1, label=1),
                        Ellipse(0, 0, 1, 1, label=0),
                        Ellipse(0, 0, 1, 1, label=1),
                    ],
                ),
            ],
            categories=["b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                        Label(1),
                        Label(2),
                        Bbox(0, 0, 1, 1, label=1),
                        Bbox(0, 0, 1, 1, label=2),
                        Ellipse(0, 0, 1, 1, label=1),
                        Ellipse(0, 0, 1, 1, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        merger = IntersectMerge()
        merged = merger(source0, source1)

        compare_datasets(self, expected, merged, ignored_attrs={"score"})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_categories(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b"]),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (0, ["l0", "l1"]),
                        (1, ["l2", "l3"]),
                    ]
                ),
                AnnotationType.mask: MaskCategories(
                    {
                        0: (0, 1, 2),
                        1: (1, 2, 3),
                    }
                ),
            },
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["c", "b"]),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (0, []),
                        (1, ["l2", "l3"]),
                    ]
                ),
                AnnotationType.mask: MaskCategories(
                    {
                        0: (0, 2, 4),
                        1: (1, 2, 3),
                    }
                ),
            },
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                        Label(2),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b", "c"]),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (0, ["l0", "l1"]),
                        (1, ["l2", "l3"]),
                        (2, []),
                    ]
                ),
                AnnotationType.mask: MaskCategories(
                    {
                        0: (0, 1, 2),
                        1: (1, 2, 3),
                        2: (0, 2, 4),
                    }
                ),
            },
        )

        merger = IntersectMerge()
        merged = merger(source0, source1)

        compare_datasets(self, expected, merged, ignored_attrs={"score"})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_point_clouds(self):
        dataset_dir = get_test_asset_path("sly_pointcloud_dataset")
        pcd1 = osp.join(dataset_dir, "ds0", "pointcloud", "frame1.pcd")
        pcd2 = osp.join(dataset_dir, "ds0", "pointcloud", "frame2.pcd")

        image1 = Image.from_file(
            path=osp.join(dataset_dir, "ds0", "related_images", "frame1_pcd", "img2.png")
        )
        image2 = Image.from_file(
            path=osp.join(dataset_dir, "ds0", "related_images", "frame2_pcd", "img1.png")
        )

        source0 = Dataset.from_iterable(
            [
                DatasetItem(1, media=PointCloud.from_file(path=pcd1, extra_images=[image1])),
                DatasetItem(2, media=PointCloud.from_file(path=pcd1, extra_images=[image1])),
                DatasetItem(3, media=PointCloud.from_file(path=pcd2)),
                DatasetItem(4),
                DatasetItem(5, media=PointCloud.from_file(path=pcd2)),
            ],
            categories=[],
            media_type=PointCloud,
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(1, media=PointCloud.from_file(path=pcd1, extra_images=[image1])),
                DatasetItem(2, media=PointCloud.from_file(path=pcd1, extra_images=[image2])),
                DatasetItem(3),
                DatasetItem(4, media=PointCloud.from_file(path=pcd2)),
                DatasetItem(5, media=PointCloud.from_file(path=pcd2, extra_images=[image2])),
            ],
            categories=[],
            media_type=PointCloud,
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(1, media=PointCloud.from_file(path=pcd1, extra_images=[image1])),
                DatasetItem(
                    2, media=PointCloud.from_file(path=pcd1, extra_images=[image1, image2])
                ),
                DatasetItem(3, media=PointCloud.from_file(path=pcd2)),
                DatasetItem(4, media=PointCloud.from_file(path=pcd2)),
                DatasetItem(5, media=PointCloud.from_file(path=pcd2, extra_images=[image2])),
            ],
            categories=[],
            media_type=PointCloud,
        )

        merger = IntersectMerge()
        merged = merger(source0, source1)

        compare_datasets(self, expected, merged)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_multiframe_images(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(1, media=MultiframeImage([np.ones((1, 5, 3))] * 2)),
                DatasetItem(2, media=MultiframeImage([np.ones((3, 5, 3))] * 2)),
                DatasetItem(3, media=MultiframeImage([np.zeros((1, 5, 3))] * 2)),
                DatasetItem(4),
                DatasetItem(5, media=MultiframeImage([np.ones((1, 5, 3))] * 4)),
            ],
            categories=[],
            media_type=MultiframeImage,
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(1, media=MultiframeImage([np.ones((1, 5, 3))] * 2)),
                DatasetItem(2, media=MultiframeImage([np.ones((3, 5, 3))] * 3)),
                DatasetItem(3),
                DatasetItem(4, media=MultiframeImage([np.ones((4, 5, 3))] * 2)),
                DatasetItem(5, media=MultiframeImage([np.ones((1, 5, 3))] * 2)),
            ],
            categories=[],
            media_type=MultiframeImage,
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(1, media=MultiframeImage([np.ones((1, 5, 3))] * 2)),
                DatasetItem(2, media=MultiframeImage([np.ones((3, 5, 3))] * 3)),
                DatasetItem(3, media=MultiframeImage([np.zeros((1, 5, 3))] * 2)),
                DatasetItem(4, media=MultiframeImage([np.ones((4, 5, 3))] * 2)),
                DatasetItem(5, media=MultiframeImage([np.ones((1, 5, 3))] * 4)),
            ],
            categories=[],
            media_type=MultiframeImage,
        )

        merger = IntersectMerge()
        merged = merger(source0, source1)

        compare_datasets(self, expected, merged)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_union(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    "0",
                    annotations=[
                        Label(0),
                    ],
                ),
                DatasetItem(
                    "1",
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b"]),
            },
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    "1",
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=0)],
                ),
                DatasetItem(
                    "2",
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["c", "b"]),
            },
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    "0",
                    annotations=[
                        Label(0),
                    ],
                ),
                DatasetItem(
                    "1-0",
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
                DatasetItem(
                    "1-1",
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=2)],
                ),
                DatasetItem(
                    "2",
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b", "c"]),
            },
        )

        merger = UnionMerge()
        source = merger(source0, source1)
        merged = Dataset(source=source)

        compare_datasets(self, expected, merged, ignored_attrs={"score"})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_raises_error_exact_merge_different_categories(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    annotations=[
                        Label(0),
                    ],
                ),
                DatasetItem(
                    1,
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b"]),
            },
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    2,
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=0)],
                ),
                DatasetItem(
                    3,
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["c", "b"]),
            },
        )

        merger = ExactMerge()
        with pytest.raises(ConflictingCategoriesError):
            _ = merger(source0, source1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_exact_image(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    media=Image.from_numpy(data=np.ones([5, 5, 3])),
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a"]),
            },
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    media=Image.from_numpy(data=np.zeros([5, 5, 3])),
                    annotations=[Mask(image=np.ones((2, 2), dtype=np.uint8), label=0)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a"]),
            },
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    media=Image.from_numpy(data=np.ones([5, 5, 3])),
                    annotations=[
                        Label(0),
                        Mask(image=np.ones((2, 2), dtype=np.uint8), label=0),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a"]),
            },
        )

        merger = ExactMerge()
        source = merger(source0, source1)
        merged = Dataset(source=source)

        compare_datasets(self, expected, merged, ignored_attrs={"score"})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_raises_error_exact_image_merge(self):
        with TemporaryDirectory() as tmp_dir:
            Image.from_numpy(data=np.ones([5, 5, 3])).save(osp.join(tmp_dir, "ones.png"))
            Image.from_numpy(data=np.zeros([5, 5, 3])).save(osp.join(tmp_dir, "zeros.png"))

            source0 = Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=Image.from_file(path=osp.join(tmp_dir, "ones.png")),
                        annotations=[
                            Label(0),
                        ],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a"]),
                },
            )

            source1 = Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=Image.from_file(path=osp.join(tmp_dir, "zeros.png")),
                        annotations=[Mask(image=np.ones((2, 2), dtype=np.uint8), label=0)],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a"]),
                },
            )

            merger = ExactMerge()
            with pytest.raises(MismatchingMediaPathError):
                _ = merger(source0, source1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_exact_pcd(self):
        with TemporaryDirectory() as tmp_dir:
            Image.from_numpy(data=np.ones([5, 5, 3])).save(osp.join(tmp_dir, "ones.png"))
            Image.from_numpy(data=np.zeros([5, 5, 3])).save(osp.join(tmp_dir, "zeros.png"))

            source0 = Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=Image.from_file(path=osp.join(tmp_dir, "ones.png")),
                        annotations=[
                            Label(0),
                        ],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a"]),
                },
            )

            source1 = Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=Image.from_file(path=osp.join(tmp_dir, "zeros.png")),
                        annotations=[Mask(image=np.ones((2, 2), dtype=np.uint8), label=0)],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a"]),
                },
            )

            merger = ExactMerge()
            with pytest.raises(MismatchingMediaPathError):
                _ = merger(source0, source1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_raises_error_exact_pcd_merge(self):
        with TemporaryDirectory() as tmp_dir:
            with open(osp.join(tmp_dir, "ones.pcd"), "wb") as f:
                f.write(b"1111")
            with open(osp.join(tmp_dir, "zeros.pcd"), "wb") as f:
                f.write(b"0000")

            source0 = Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=PointCloud.from_file(path=osp.join(tmp_dir, "ones.pcd")),
                        annotations=[
                            Label(0),
                        ],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a"]),
                },
                media_type=PointCloud,
            )

            source1 = Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=PointCloud.from_file(path=osp.join(tmp_dir, "zeros.pcd")),
                        annotations=[Mask(image=np.ones((2, 2), dtype=np.uint8), label=0)],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a"]),
                },
                media_type=PointCloud,
            )

            merger = ExactMerge()
            with pytest.raises(MismatchingMediaPathError):
                _ = merger(source0, source1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_exact_image_with_different_path(self):
        with TemporaryDirectory() as tmp_dir:
            Image.from_numpy(data=np.ones([5, 5, 3])).save(osp.join(tmp_dir, "ones.png"))

            source0 = Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=Image.from_file(path=osp.join(tmp_dir, "ones.png")),
                        annotations=[
                            Label(0),
                        ],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a"]),
                },
            )

            source1 = Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=Image.from_file(path="dummy/path.png"),
                        annotations=[Mask(image=np.ones((2, 2), dtype=np.uint8), label=0)],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a"]),
                },
            )

            expected = Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=Image.from_numpy(data=np.ones([5, 5, 3])),
                        annotations=[
                            Label(0),
                            Mask(image=np.ones((2, 2), dtype=np.uint8), label=0),
                        ],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a"]),
                },
            )

            merger = ExactMerge()
            source = merger(source0, source1)
            merged = Dataset(source=source)

            compare_datasets(self, expected, merged, ignored_attrs={"score"})
