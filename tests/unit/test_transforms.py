from __future__ import annotations

import logging as log
import random
from unittest import TestCase

import numpy as np
import pycocotools.mask as mask_utils
import pytest

import datumaro.plugins.transforms as transforms
import datumaro.util.mask_tools as mask_tools
from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RleMask,
    Tabular,
    TabularCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image

from ..requirements import Requirements, mark_bug, mark_requirement

from tests.utils.test_utils import compare_datasets, compare_datasets_strict


class TransformsTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(id=10),
                DatasetItem(id=10, subset="train"),
                DatasetItem(id="a", subset="val"),
            ]
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(id=5),
                DatasetItem(id=6, subset="train"),
                DatasetItem(id=7, subset="val"),
            ]
        )

        actual = transforms.Reindex(source, start=5)
        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_sort(self):
        items = []
        for i in range(10):
            items.append(DatasetItem(id=i))

        expected = Dataset.from_iterable(items)

        items = items.copy()
        random.shuffle(items)
        source = Dataset.from_iterable(items)

        actual = transforms.Sort(source, lambda item: int(item.id))
        compare_datasets_strict(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_mask_to_polygons(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                        ),
                    ],
                ),
            ]
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon([1, 0, 3, 2, 3, 0, 1, 0]),
                        Polygon([5, 0, 5, 3, 8, 0, 5, 0]),
                    ],
                ),
            ]
        )

        actual = transforms.MasksToPolygons(source)
        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_mask_to_polygons_small_polygons_message(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0],
                                ]
                            ),
                        ),
                    ],
                ),
            ]
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, media=Image.from_numpy(data=np.zeros((5, 10, 3)))),
            ]
        )

        with self.assertLogs(level=log.DEBUG) as logs:
            actual = transforms.MasksToPolygons(source_dataset)

            compare_datasets(self, target_dataset, actual)
            self.assertRegex("\n".join(logs.output), "too small polygons")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_polygons_to_masks(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon([0, 0, 4, 0, 4, 4]),
                        Polygon([5, 0, 9, 0, 5, 5]),
                        Ellipse(0, 1, 5, 4),
                        Ellipse(6, 0, 9, 5),
                    ],
                ),
            ]
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                        ),
                        Mask(
                            np.array(
                                [
                                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                        ),
                        Mask(
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                        ),
                        Mask(
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                ]
                            ),
                        ),
                    ],
                ),
            ]
        )

        actual = transforms.PolygonsToMasks(source_dataset)
        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_crop_covered_segments(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        # The mask is partially covered by the polygon
                        Mask(
                            np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 0, 0],
                                    [1, 1, 1, 0, 0],
                                ],
                            ),
                            z_order=0,
                        ),
                        Polygon([1, 1, 4, 1, 4, 4, 1, 4], z_order=1),
                    ],
                ),
            ]
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0],
                                ],
                            ),
                            z_order=0,
                        ),
                        Polygon([1, 1, 4, 1, 4, 4, 1, 4], z_order=1),
                    ],
                ),
            ]
        )

        actual = transforms.CropCoveredSegments(source_dataset)
        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_merge_instance_segments(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0],
                                ],
                            ),
                            z_order=0,
                            group=1,
                        ),
                        Polygon([1, 1, 4, 1, 4, 4, 1, 4], z_order=1, group=1),
                        Polygon([0, 0, 0, 2, 2, 2, 2, 0], z_order=1),
                    ],
                ),
            ]
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 0],
                                    [1, 1, 1, 0, 0],
                                ],
                            ),
                            z_order=0,
                            group=1,
                        ),
                        Mask(
                            np.array(
                                [
                                    [1, 1, 0, 0, 0],
                                    [1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ],
                            ),
                            z_order=1,
                        ),
                    ],
                ),
            ]
        )

        actual = transforms.MergeInstanceSegments(source_dataset, include_polygons=True)
        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_map_subsets(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, subset="a"),
                DatasetItem(id=2, subset="b"),
                DatasetItem(id=3, subset="c"),
            ]
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, subset=""),
                DatasetItem(id=2, subset="a"),
                DatasetItem(id=3, subset="c"),
            ]
        )

        actual = transforms.MapSubsets(source_dataset, {"a": "", "b": "a"})
        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_shapes_to_boxes(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0],
                                ],
                            ),
                            id=1,
                        ),
                        Polygon([1, 1, 4, 1, 4, 4, 1, 4], id=2),
                        PolyLine([1, 1, 2, 1, 2, 2, 1, 2], id=3),
                        Points([2, 2, 4, 2, 4, 4, 2, 4], id=4),
                        Ellipse(0, 1, 5, 4, id=5),
                    ],
                ),
            ]
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Bbox(0, 0, 4, 4, id=1),
                        Bbox(1, 1, 3, 3, id=2),
                        Bbox(1, 1, 1, 1, id=3),
                        Bbox(2, 2, 2, 2, id=4),
                        Bbox(0, 1, 5, 3, id=5),
                    ],
                ),
            ]
        )

        actual = transforms.ShapesToBoxes(source_dataset)
        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_id_from_image(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, media=Image.from_file(path="path.jpg")),
                DatasetItem(id=2),
                DatasetItem(id=3, media=Image.from_numpy(data=np.ones([5, 5, 3]))),
            ]
        )
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="path", media=Image.from_file(path="path.jpg")),
                DatasetItem(id=2),
                DatasetItem(id=3, media=Image.from_numpy(data=np.ones([5, 5, 3]))),
            ]
        )

        actual = transforms.IdFromImageName(source_dataset)
        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_boxes_to_masks(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Bbox(0, 0, 3, 3, z_order=1),
                        Bbox(0, 0, 3, 1, z_order=2),
                        Bbox(0, 2, 3, 1, z_order=3),
                    ],
                ),
            ]
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [1, 1, 1, 0, 0],
                                    [1, 1, 1, 0, 0],
                                    [1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ],
                            ),
                            z_order=1,
                        ),
                        Mask(
                            np.array(
                                [
                                    [1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ],
                            ),
                            z_order=2,
                        ),
                        Mask(
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ],
                            ),
                            z_order=3,
                        ),
                    ],
                ),
            ]
        )

        actual = transforms.BoxesToMasks(source_dataset)
        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_boxes_to_polygons(self):
        x, y, w, h = 0, 0, 3, 3
        points = (0, 0, 3, 0, 3, 3, 0, 3)
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Bbox(x, y, w, h, id=1),
                        Bbox(x, y, w, h, attributes={"a": 1}),
                        Bbox(x, y, w, h, group=0),
                        Bbox(x, y, w, h, label=2),
                        Bbox(x, y, w, h, z_order=3),
                    ],
                ),
            ]
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Polygon(points=points, id=1),
                        Polygon(points=points, attributes={"a": 1}),
                        Polygon(points=points, group=0),
                        Polygon(points=points, label=2),
                        Polygon(points=points, z_order=3),
                    ],
                ),
            ]
        )

        actual = transforms.BoxesToPolygons(source_dataset)
        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_random_split(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, subset="a"),
                DatasetItem(id=2, subset="a"),
                DatasetItem(id=3, subset="b"),
                DatasetItem(id=4, subset="b"),
                DatasetItem(id=5, subset="b"),
                DatasetItem(id=6, subset=""),
                DatasetItem(id=7, subset=""),
            ]
        )

        actual = transforms.RandomSplit(
            source_dataset,
            splits=[
                ("train", 4.0 / 7.0),
                ("test", 3.0 / 7.0),
            ],
        )

        self.assertEqual(4, len(actual.get_subset("train")))
        self.assertEqual(3, len(actual.get_subset("test")))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_random_split_gives_error_on_wrong_ratios(self):
        source_dataset = Dataset.from_iterable([DatasetItem(id=1)])

        with self.assertRaises(Exception):
            transforms.RandomSplit(
                source_dataset,
                splits=[
                    ("train", 0.5),
                    ("test", 0.7),
                ],
            )

        with self.assertRaises(Exception):
            transforms.RandomSplit(source_dataset, splits=[])

        with self.assertRaises(Exception):
            transforms.RandomSplit(
                source_dataset,
                splits=[
                    ("train", -0.5),
                    ("test", 1.5),
                ],
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_rename_item(self):
        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="frame_1"),
                DatasetItem(id="frame_2"),
                DatasetItem(id="frame_3"),
            ]
        )
        expected = Dataset.from_iterable(
            [
                DatasetItem(id="1"),
                DatasetItem(id="2"),
                DatasetItem(id="3"),
            ]
        )
        actual = transforms.Rename(src_dataset, "|^frame_|")
        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_remap_labels(self):
        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        # Should be remapped
                        Label(1),
                        Bbox(1, 2, 3, 4, label=2),
                        Mask(image=np.array([1]), label=3),
                        # Should be deleted
                        Polygon([1, 1, 2, 2, 3, 4], label=4),
                        # Should be kept
                        PolyLine([1, 3, 4, 2, 5, 6]),
                        Bbox(4, 3, 2, 1, label=5),
                    ],
                )
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(f"label{i}" for i in range(6)),
                AnnotationType.mask: MaskCategories(colormap=mask_tools.generate_colormap(6)),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(i, [str(i)]) for i in range(6)]
                ),
            },
        )

        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(1),
                        Bbox(1, 2, 3, 4, label=0),
                        Mask(image=np.array([1]), label=1),
                        PolyLine([1, 3, 4, 2, 5, 6], label=None),
                        Bbox(4, 3, 2, 1, label=2),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["label0", "label9", "label5"]),
                AnnotationType.mask: MaskCategories(
                    colormap={
                        i: v
                        for i, v in enumerate(
                            {
                                k: v
                                for k, v in mask_tools.generate_colormap(6).items()
                                if k in {0, 1, 5}
                            }.values()
                        )
                    }
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(0, ["0"]), (1, ["1"]), (2, ["5"])]
                ),
            },
        )

        actual = transforms.RemapLabels(
            src_dataset,
            mapping={
                "label1": "label9",  # rename & join with new label9 (from label3)
                "label2": "label0",  # rename & join with existing label0
                "label3": "label9",  # rename & join with new label9 (from label1)
                "label4": "",  # delete the label and associated annotations
                # 'label5' - unchanged
            },
            default="keep",
        )

        compare_datasets(self, dst_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_remap_labels_delete_unspecified(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(0, id=0),  # will be removed
                        Label(1, id=1),
                        Bbox(1, 2, 3, 4, label=None),
                    ],
                )
            ],
            categories=["label0", "label1"],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(0, id=1),
                    ],
                ),
            ],
            categories=["label1"],
        )

        actual = transforms.RemapLabels(
            source_dataset, mapping={"label1": "label1"}, default="delete"
        )

        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_BUG_314)
    def test_remap_labels_ignore_missing_labels_in_secondary_categories(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(0),
                    ],
                )
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b", "c"]),
                AnnotationType.points: PointsCategories.from_iterable([]),  # all missing
                AnnotationType.mask: MaskCategories.generate(2),  # no c color
            },
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["d", "e", "f"]),
                AnnotationType.points: PointsCategories.from_iterable([]),
                AnnotationType.mask: MaskCategories.generate(2),
            },
        )

        actual = transforms.RemapLabels(
            source_dataset, mapping={"a": "d", "b": "e", "c": "f"}, default="delete"
        )

        compare_datasets(self, target_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_project_labels(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(1),  # Label must be remapped
                        Label(3),  # Must be removed (extra label)
                        Bbox(1, 2, 3, 4, label=None),  # Must be kept (no label)
                    ],
                )
            ],
            categories=["a", "b", "c", "d"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(2),
                        Bbox(1, 2, 3, 4, label=None),
                    ],
                ),
            ],
            categories=["c", "a", "b"],
        )

        actual = transforms.ProjectLabels(source, dst_labels=["c", "a", "b"])

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_project_labels_maps_secondary_categories(self):
        source = Dataset.from_iterable(
            [],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    ["a", "b", ("c", "a"), ("d", "b")]  # no parents  # have parents
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(0, ["a"]), (1, ["b"]), (2, ["c"])]
                ),
                AnnotationType.mask: MaskCategories.generate(4),
            },
        )

        expected = Dataset.from_iterable(
            [],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        ("c", "a"),  # must keep parent
                        "a",
                        "d",  # must drop parent because it was removed
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable([(0, ["c"]), (1, ["a"])]),
                AnnotationType.mask: MaskCategories(
                    colormap={
                        i: v
                        for i, v in {
                            {2: 0, 0: 1, 3: 2}.get(k): v
                            for k, v in mask_tools.generate_colormap(4).items()
                        }.items()
                        if i is not None
                    }
                ),
            },
        )

        actual = transforms.ProjectLabels(source, dst_labels=["c", "a", "d"])

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_project_labels_generates_colors_for_added_labels(self):
        source = Dataset.from_iterable(
            [],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b", "c"]),
                AnnotationType.mask: MaskCategories.generate(2),
            },
        )

        actual = transforms.ProjectLabels(source, dst_labels=["a", "c", "d"])

        self.assertEqual((0, 0, 0), actual.categories()[AnnotationType.mask][0])
        self.assertNotIn(1, actual.categories()[AnnotationType.mask])
        self.assertIn(2, actual.categories()[AnnotationType.mask])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_to_labels(self):
        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(1),
                        Bbox(1, 2, 3, 4, label=2),
                        Bbox(1, 3, 3, 3),
                        Mask(image=np.array([1]), label=3),
                        Polygon([1, 1, 2, 2, 3, 4], label=4),
                        PolyLine([1, 3, 4, 2, 5, 6], label=5),
                    ],
                )
            ],
            categories=["label%s" % i for i in range(6)],
        )

        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Label(1), Label(2), Label(3), Label(4), Label(5)]),
            ],
            categories=["label%s" % i for i in range(6)],
        )

        actual = transforms.AnnsToLabels(src_dataset)

        compare_datasets(self, dst_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_bboxes_values_decrement_transform(self):
        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[Label(1), Bbox(2, 3, 3, 4, label=2), Bbox(1.3, 3.5, 3.33, 3.12)],
                )
            ],
            categories=["label%s" % i for i in range(6)],
        )

        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[Label(1), Bbox(1, 2, 3, 4, label=2), Bbox(0.3, 2.5, 3.33, 3.12)],
                ),
            ],
            categories=["label%s" % i for i in range(6)],
        )

        actual = transforms.BboxValuesDecrement(src_dataset)

        compare_datasets(self, dst_dataset, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @mark_bug(Requirements.DATUM_BUG_618)
    def test_can_resize(self):
        small_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=i,
                    media=Image.from_numpy(data=np.ones((4, 4)) * i),
                    annotations=[
                        Label(1),
                        Bbox(1, 1, 2, 2, label=2),
                        Polygon([1, 1, 1, 2, 2, 2, 2, 1], label=1),
                        PolyLine([1, 1, 1, 2, 2, 2, 2, 1], label=2),
                        Points([1, 1, 1, 2, 2, 2, 2, 1], label=2),
                        Mask(
                            np.array(
                                [
                                    [0, 0, 1, 1],
                                    [1, 0, 0, 1],
                                    [0, 1, 1, 0],
                                    [1, 1, 0, 0],
                                ]
                            )
                        ),
                        RleMask(
                            mask_utils.encode(
                                np.asfortranarray(
                                    np.array(
                                        [
                                            [0, 0, 1, 1],
                                            [1, 0, 0, 1],
                                            [0, 1, 1, 0],
                                            [1, 1, 0, 0],
                                        ]
                                    ).astype(np.uint8)
                                )
                            )
                        ),
                    ],
                )
                for i in range(3)
            ],
            categories=["a", "b", "c"],
        )

        big_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=i,
                    media=Image.from_numpy(data=np.ones((8, 8)) * i),
                    annotations=[
                        Label(1),
                        Bbox(2, 2, 4, 4, label=2),
                        Polygon([2, 2, 2, 4, 4, 4, 4, 2], label=1),
                        PolyLine([2, 2, 2, 4, 4, 4, 4, 2], label=2),
                        Points([2, 2, 2, 4, 4, 4, 4, 2], label=2),
                        Mask(
                            np.array(
                                [
                                    [0, 0, 0, 0, 1, 1, 1, 1],
                                    [0, 0, 0, 0, 1, 1, 1, 1],
                                    [1, 1, 0, 0, 0, 0, 1, 1],
                                    [1, 1, 0, 0, 0, 0, 1, 1],
                                    [0, 0, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 0, 0],
                                    [1, 1, 1, 1, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 0, 0, 0, 0],
                                ]
                            )
                        ),
                        RleMask(
                            mask_utils.encode(
                                np.asfortranarray(
                                    np.array(
                                        [
                                            [0, 0, 0, 0, 1, 1, 1, 1],
                                            [0, 0, 0, 0, 1, 1, 1, 1],
                                            [1, 1, 0, 0, 0, 0, 1, 1],
                                            [1, 1, 0, 0, 0, 0, 1, 1],
                                            [0, 0, 1, 1, 1, 1, 0, 0],
                                            [0, 0, 1, 1, 1, 1, 0, 0],
                                            [1, 1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 0, 0, 0, 0],
                                        ]
                                    ).astype(np.uint8)
                                )
                            )
                        ),
                    ],
                )
                for i in range(3)
            ],
            categories=["a", "b", "c"],
        )

        with self.subTest("upscale"):
            actual = transforms.ResizeTransform(small_dataset, width=8, height=8)
            compare_datasets(self, big_dataset, actual)

        with self.subTest("downscale"):
            actual = transforms.ResizeTransform(big_dataset, width=4, height=4)
            compare_datasets(self, small_dataset, actual)

    @mark_bug(Requirements.DATUM_BUG_606)
    def test_can_keep_image_ext_on_resize(self):
        expected = Image.from_numpy(data=np.ones((8, 4)), ext="jpg")

        dataset = Dataset.from_iterable(
            [DatasetItem(id=1, media=Image.from_numpy(data=np.ones((4, 2)), ext="jpg"))]
        )

        dataset.transform("resize", width=4, height=8)

        actual = dataset.get("1").media
        self.assertEqual(actual.ext, expected.ext)
        self.assertTrue(np.array_equal(actual.data, expected.data))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_remove_items_by_ids(self):
        expected = Dataset.from_iterable([DatasetItem(id="1", subset="train")])

        dataset = Dataset.from_iterable(
            [DatasetItem(id="1", subset="train"), DatasetItem(id="2", subset="train")]
        )

        actual = transforms.RemoveItems(dataset, ids=[("2", "train")])

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_remove_annotations_by_item_id(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="test"),
                DatasetItem(id="2", subset="test", annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="test", annotations=[Label(0)]),
                DatasetItem(id="2", subset="test", annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )

        actual = transforms.RemoveAnnotations(dataset, ids=[("1", "test")])

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_remove_annotations_in_dataset(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train"),
                DatasetItem(id="1", subset="test", annotations=[Label(1, id=1)]),
            ],
            categories=["a", "b"],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    annotations=[
                        Label(0, id=0),
                        Label(0, id=1),
                        Label(1, id=2),
                    ],
                ),
                DatasetItem(
                    id="1",
                    subset="test",
                    annotations=[
                        Label(0, id=0),
                        Label(1, id=1),
                        Label(1, id=2),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        actual = transforms.RemoveAnnotations(
            dataset, ids=[("1", "train"), ("1", "test", 0), ("1", "test", 2)]
        )

        compare_datasets(self, expected, actual)


class RemoveAttributesTest(TestCase):
    def setUp(self):
        self.source = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="val",
                    attributes={"qq": 1, "x": 2},
                    annotations=[Label(0, attributes={"x": 1, "y": 2})],
                ),
                DatasetItem(
                    id="2",
                    subset="val",
                    attributes={"qq": 2},
                    annotations=[Label(0, attributes={"x": 1, "y": 2})],
                ),
            ],
            categories=["a"],
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_remove_all_attrs_by_item_id(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="val", annotations=[Label(0)]),
                DatasetItem(
                    id="2",
                    subset="val",
                    attributes={"qq": 2},
                    annotations=[Label(0, attributes={"x": 1, "y": 2})],
                ),
            ],
            categories=["a"],
        )

        actual = transforms.RemoveAttributes(self.source, ids=[("1", "val")])

        compare_datasets(self, expected, actual)

    def test_can_remove_attrs_by_name_from_all_items(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="val",
                    attributes={"qq": 1},
                    annotations=[Label(0, attributes={"y": 2})],
                ),
                DatasetItem(
                    id="2",
                    subset="val",
                    attributes={"qq": 2},
                    annotations=[Label(0, attributes={"y": 2})],
                ),
            ],
            categories=["a"],
        )

        actual = transforms.RemoveAttributes(self.source, attributes=["x"])

        compare_datasets(self, expected, actual)

    def test_can_remove_attrs_by_name_from_specific_items(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="val",
                    attributes={"qq": 1},
                    annotations=[Label(0, attributes={"y": 2})],
                ),
                DatasetItem(
                    id="2",
                    subset="val",
                    attributes={"qq": 2},
                    annotations=[Label(0, attributes={"x": 1, "y": 2})],
                ),
            ],
            categories=["a"],
        )

        actual = transforms.RemoveAttributes(self.source, ids=[("1", "val")], attributes=["x"])

        compare_datasets(self, expected, actual)


class ReindexAnnotationsTest:
    @pytest.fixture
    def fxt_dataset(self, n_labels=3, n_anns=5, n_items=7) -> Dataset:
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=f"item_{item_idx}",
                    annotations=[
                        Label(label=random.randint(0, n_labels - 1)) for _ in range(n_anns)
                    ],
                )
                for item_idx in range(n_items)
            ],
            LabelCategories.from_iterable([f"label_{label_id}" for label_id in range(n_labels)]),
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("reindex_each_item", [True, False])
    def test_annotation_reindex(self, fxt_dataset: Dataset, reindex_each_item: bool):
        start = random.randint(0, 10)
        n_anns = sum([len(item.annotations) for item in fxt_dataset])

        # n_anns != n_unique_ids
        assert n_anns != len(
            {(item.id, ann.id) for item in fxt_dataset for ann in item.annotations}
        )

        fxt_dataset.transform(
            "reindex_annotations",
            start=start,
            reindex_each_item=reindex_each_item,
        )
        # n_anns == n_unique_ids
        assert n_anns == len(
            {(item.id, ann.id) for item in fxt_dataset for ann in item.annotations}
        )


from pandas.api.types import CategoricalDtype


class AstypeAnnotationsTest(TestCase):
    def setUp(self):
        self.table = Table.from_list(
            [{"nswprice": 0.076108, "class": "DOWN"}, {"nswprice": 0.060376, "class": "UP"}]
        )
        self.dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0",
                    subset="train",
                    media=TableRow(table=self.table, index=0),
                    annotations=[Tabular(values={"class": "DOWN"})],
                ),
                DatasetItem(
                    id="1",
                    subset="train",
                    media=TableRow(table=self.table, index=1),
                    annotations=[Tabular(values={"class": "UP"})],
                ),
            ],
            categories={
                AnnotationType.tabular: TabularCategories.from_iterable(
                    [("class", CategoricalDtype(), {"DOWN", "UP"})]
                )
            },
            media_type=TableRow,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_annotation_type(self):
        table = self.table
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0",
                    subset="train",
                    media=TableRow(table=table, index=0),
                    annotations=[Label(label=0)],
                ),
                DatasetItem(
                    id="1",
                    subset="train",
                    media=TableRow(table=table, index=1),
                    annotations=[Label(label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [("DOWN", "class"), ("UP", "class")]
                )
            },
            media_type=TableRow,
        )

        dataset = self.dataset
        result = transforms.AstypeAnnotations(dataset)

        categories = result._categories.get(AnnotationType.label, None)
        assert categories

        # Check label_groups of categories
        assert categories.label_groups[0].name == "class"
        assert sorted(categories.label_groups[0].labels) == ["DOWN", "UP"]

        compare_datasets(self, expected, result)
