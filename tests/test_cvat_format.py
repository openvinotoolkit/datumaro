import os
import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    Skeleton,
)
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.cvat_format.converter import CvatConverter
from datumaro.plugins.cvat_format.extractor import CvatImporter
from datumaro.util.test_utils import TestDir, check_save_and_load, compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_IMAGE_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "cvat_dataset", "for_images")

DUMMY_VIDEO_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "cvat_dataset", "for_video")


class CvatImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_image(self):
        detected_formats = Environment().detect_dataset(DUMMY_IMAGE_DATASET_DIR)
        self.assertEqual([CvatImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_video(self):
        detected_formats = Environment().detect_dataset(DUMMY_VIDEO_DATASET_DIR)
        self.assertEqual([CvatImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_image(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="img0",
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(
                            0,
                            2,
                            4,
                            2,
                            label=0,
                            z_order=1,
                            attributes={
                                "occluded": True,
                                "a1": True,
                                "a2": "v3",
                                "a3": "0003",
                                "a4": 2.4,
                            },
                        ),
                        PolyLine([1, 2, 3, 4, 5, 6, 7, 8], attributes={"occluded": False}),
                        Skeleton(
                            [
                                Points(
                                    [1, 1], label=3, attributes={"occluded": False, "outside": True}
                                ),
                                Points(
                                    [2, 2],
                                    label=4,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [3, 3],
                                    label=5,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [4, 4],
                                    label=6,
                                    attributes={"occluded": False, "outside": False},
                                ),
                            ],
                            label=2,
                            attributes={"occluded": False},
                        ),
                    ],
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id="img1",
                    subset="train",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[
                        Polygon([1, 2, 3, 4, 6, 5], z_order=1, attributes={"occluded": False}),
                        Points(
                            [1, 2, 3, 4, 5, 6], label=1, z_order=2, attributes={"occluded": False}
                        ),
                        Skeleton(
                            [
                                Points(
                                    [3, 3],
                                    label=3,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [4, 4],
                                    label=4,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [5, 5], label=5, attributes={"occluded": False, "outside": True}
                                ),
                                Points(
                                    [6, 6],
                                    label=6,
                                    attributes={"occluded": False, "outside": False},
                                ),
                            ],
                            label=2,
                            attributes={"occluded": False},
                        ),
                    ],
                    attributes={"frame": 1},
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        ["label1", "", {"a1", "a2", "a3", "a4"}],
                        ["label2"],
                        ["skeleton", ""],
                        ["1", "skeleton"],
                        ["2", "skeleton"],
                        ["3", "skeleton"],
                        ["4", "skeleton"],
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(2, ["1", "2", "3", "4"], [[2, 1], [0, 3]])]
                ),
            },
        )

        parsed_dataset = Dataset.import_from(DUMMY_IMAGE_DATASET_DIR, "cvat")

        compare_datasets(self, expected_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_video(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="frame_000010",
                    subset="annotations",
                    media=Image(data=255 * np.ones((20, 25, 3))),
                    annotations=[
                        Bbox(
                            3,
                            4,
                            7,
                            1,
                            label=2,
                            id=0,
                            attributes={
                                "occluded": True,
                                "outside": False,
                                "keyframe": True,
                                "track_id": 0,
                            },
                        ),
                        Points(
                            [21.95, 8.00, 2.55, 15.09, 2.23, 3.16],
                            label=0,
                            id=1,
                            attributes={
                                "occluded": False,
                                "outside": False,
                                "keyframe": True,
                                "track_id": 1,
                                "hgl": "hgkf",
                            },
                        ),
                        Skeleton(
                            [
                                Points(
                                    [48.80, 111.77],
                                    id=0,
                                    label=4,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": True,
                                    },
                                ),
                                Points(
                                    [48.80, 156.28],
                                    id=1,
                                    label=5,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": True,
                                    },
                                ),
                                Points(
                                    [83.27, 106.61],
                                    id=2,
                                    label=6,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": True,
                                    },
                                ),
                                Points(
                                    [89.11, 159.91],
                                    id=3,
                                    label=7,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": True,
                                    },
                                ),
                            ],
                            id=3,
                            label=3,
                            attributes={
                                "track_id": 3,
                                "occluded": False,
                                "outside": False,
                                "keyframe": True,
                            },
                        ),
                    ],
                    attributes={"frame": 10},
                ),
                DatasetItem(
                    id="frame_000013",
                    subset="annotations",
                    media=Image(data=255 * np.ones((20, 25, 3))),
                    annotations=[
                        Bbox(
                            7,
                            6,
                            7,
                            2,
                            label=2,
                            id=0,
                            attributes={
                                "occluded": False,
                                "outside": True,
                                "keyframe": True,
                                "track_id": 0,
                            },
                        ),
                        Points(
                            [21.95, 8.00, 9.55, 15.09, 5.23, 1.16],
                            label=0,
                            id=1,
                            attributes={
                                "occluded": False,
                                "outside": True,
                                "keyframe": True,
                                "track_id": 1,
                                "hgl": "jk",
                            },
                        ),
                        PolyLine(
                            [7.85, 13.88, 3.50, 6.67, 15.90, 2.00, 13.31, 7.21],
                            label=2,
                            id=2,
                            attributes={
                                "occluded": False,
                                "outside": False,
                                "keyframe": True,
                                "track_id": 2,
                            },
                        ),
                        Skeleton(
                            [
                                Points(
                                    [48.80, 111.77],
                                    id=0,
                                    label=4,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": False,
                                    },
                                ),
                                Points(
                                    [48.80, 156.28],
                                    id=1,
                                    label=5,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": False,
                                    },
                                ),
                                Points(
                                    [83.27, 106.61],
                                    id=2,
                                    label=6,
                                    attributes={
                                        "occluded": False,
                                        "outside": True,
                                        "keyframe": True,
                                    },
                                ),
                                Points(
                                    [89.11, 159.91],
                                    id=3,
                                    label=7,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": False,
                                    },
                                ),
                            ],
                            label=3,
                            id=3,
                            attributes={
                                "track_id": 3,
                                "occluded": False,
                                "outside": False,
                                "keyframe": False,
                            },
                        ),
                    ],
                    attributes={"frame": 13},
                ),
                DatasetItem(
                    id="frame_000016",
                    subset="annotations",
                    media=Image(path="frame_0000016.png", size=(20, 25)),
                    annotations=[
                        Bbox(
                            8,
                            7,
                            6,
                            10,
                            label=2,
                            id=0,
                            attributes={
                                "occluded": False,
                                "outside": True,
                                "keyframe": True,
                                "track_id": 0,
                            },
                        ),
                        PolyLine(
                            [7.85, 13.88, 3.50, 6.67, 15.90, 2.00, 13.31, 7.21],
                            label=2,
                            id=2,
                            attributes={
                                "occluded": False,
                                "outside": True,
                                "keyframe": True,
                                "track_id": 2,
                            },
                        ),
                        Skeleton(
                            [
                                Points(
                                    [48.80, 111.77],
                                    id=0,
                                    label=4,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": False,
                                    },
                                ),
                                Points(
                                    [48.80, 156.28],
                                    id=1,
                                    label=5,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": False,
                                    },
                                ),
                                Points(
                                    [89.11, 159.91],
                                    id=3,
                                    label=7,
                                    attributes={
                                        "occluded": False,
                                        "outside": False,
                                        "keyframe": False,
                                    },
                                ),
                            ],
                            label=3,
                            id=3,
                            attributes={
                                "track_id": 3,
                                "occluded": False,
                                "outside": False,
                                "keyframe": False,
                            },
                        ),
                    ],
                    attributes={"frame": 16},
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        ["klhg", "", {"hgl"}],
                        ["z U k"],
                        ["II"],
                        ["skeleton", ""],
                        ["1", "skeleton"],
                        ["2", "skeleton"],
                        ["3", "skeleton"],
                        ["4", "skeleton"],
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(3, ["1", "2", "3", "4"], [[2, 1], [0, 3]])]
                ),
            },
        )

        parsed_dataset = Dataset.import_from(DUMMY_VIDEO_DATASET_DIR, "cvat")

        compare_datasets(self, expected_dataset, parsed_dataset)


class CvatConverterTest(TestCase):
    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="cvat",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        src_label_cat = LabelCategories(attributes={"occluded", "common"})
        for i in range(10):
            src_label_cat.add(str(i))
        src_label_cat.items[2].attributes.update(["a1", "a2", "empty"])
        src_label_cat.add("skeleton")
        src_label_cat.add("s_1", "skeleton")
        src_label_cat.add("s_2", "skeleton")
        src_label_cat.add("s_3", "skeleton")

        src_points_cat = PointsCategories()
        src_points_cat.add(10, ["s_1", "s_2", "s_3"], [[0, 1], [0, 2]])

        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    subset="s1",
                    media=Image(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=1,
                            group=4,
                            attributes={"occluded": True, "common": "t"},
                        ),
                        Points(
                            [1, 1, 3, 2, 2, 3],
                            label=2,
                            attributes={"a1": "x", "a2": 42, "empty": "", "unknown": "bar"},
                        ),
                        Label(1),
                        Label(2, attributes={"a1": "y", "a2": 44}),
                        Skeleton(
                            [
                                Points(
                                    [1, 1],
                                    label=11,
                                    attributes={"occluded": True, "outside": False},
                                ),
                                Points(
                                    [2, 2],
                                    label=12,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [3, 3],
                                    label=13,
                                    attributes={"occluded": False, "outside": False},
                                ),
                            ],
                            label=10,
                            attributes={"occluded": False},
                        ),
                    ],
                ),
                DatasetItem(
                    id=1,
                    subset="s1",
                    annotations=[
                        PolyLine([0, 0, 4, 0, 4, 4], label=3, id=4, group=4),
                        Bbox(5, 0, 1, 9, label=3, id=4, group=4),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="s2",
                    media=Image(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            z_order=1,
                            label=3,
                            group=4,
                            attributes={"occluded": False},
                        ),
                        PolyLine([5, 0, 9, 0, 5, 5]),  # will be skipped as no label
                        Skeleton(
                            [
                                Points(
                                    [5, 5],
                                    label=11,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [6, 6],
                                    label=12,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [7, 7],
                                    label=13,
                                    attributes={"occluded": True, "outside": False},
                                ),
                            ],
                            label=10,
                            attributes={"occluded": False},
                        ),
                    ],
                ),
                DatasetItem(id=3, subset="s3", media=Image(path="3.jpg", size=(2, 4))),
            ],
            categories={AnnotationType.label: src_label_cat, AnnotationType.points: src_points_cat},
        )

        target_label_cat = LabelCategories(
            attributes={"occluded"}
        )  # unable to represent a common attribute
        for i in range(10):
            target_label_cat.add(str(i), attributes={"common"})
        target_label_cat.items[2].attributes.update(["a1", "a2", "empty", "common"])
        target_label_cat.add("skeleton", attributes={"common"})
        target_label_cat.add("s_1", "skeleton", attributes={"common"})
        target_label_cat.add("s_2", "skeleton", attributes={"common"})
        target_label_cat.add("s_3", "skeleton", attributes={"common"})

        target_points_cat = PointsCategories()
        target_points_cat.add(10, ["s_1", "s_2", "s_3"], [[0, 1], [0, 2]])
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    subset="s1",
                    media=Image(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=1,
                            group=4,
                            attributes={"occluded": True, "common": "t"},
                        ),
                        Points(
                            [1, 1, 3, 2, 2, 3],
                            label=2,
                            attributes={"occluded": False, "empty": "", "a1": "x", "a2": "42"},
                        ),
                        Label(1),
                        Label(2, attributes={"a1": "y", "a2": "44"}),
                        Skeleton(
                            [
                                Points(
                                    [1, 1],
                                    label=11,
                                    attributes={"occluded": True, "outside": False},
                                ),
                                Points(
                                    [2, 2],
                                    label=12,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [3, 3],
                                    label=13,
                                    attributes={"occluded": False, "outside": False},
                                ),
                            ],
                            label=10,
                            attributes={"occluded": False},
                        ),
                    ],
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id=1,
                    subset="s1",
                    annotations=[
                        PolyLine(
                            [0, 0, 4, 0, 4, 4], label=3, group=4, attributes={"occluded": False}
                        ),
                        Bbox(5, 0, 1, 9, label=3, group=4, attributes={"occluded": False}),
                    ],
                    attributes={"frame": 1},
                ),
                DatasetItem(
                    id=2,
                    subset="s2",
                    media=Image(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            z_order=1,
                            label=3,
                            group=4,
                            attributes={"occluded": False},
                        ),
                        Skeleton(
                            [
                                Points(
                                    [5, 5],
                                    label=11,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [6, 6],
                                    label=12,
                                    attributes={"occluded": False, "outside": False},
                                ),
                                Points(
                                    [7, 7],
                                    label=13,
                                    attributes={"occluded": True, "outside": False},
                                ),
                            ],
                            label=10,
                            attributes={"occluded": False},
                        ),
                    ],
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id=3,
                    subset="s3",
                    media=Image(path="3.jpg", size=(2, 4)),
                    attributes={"frame": 0},
                ),
            ],
            categories={
                AnnotationType.label: target_label_cat,
                AnnotationType.points: target_points_cat,
            },
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatConverter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_allow_undeclared_attrs(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    annotations=[
                        Label(0, attributes={"x": 4, "y": 2}),
                        Bbox(1, 2, 3, 4, label=0, attributes={"x": 1, "y": 1}),
                    ],
                ),
            ],
            categories=[("a", "", {"x"})],
        )

        target_label_cat = LabelCategories(attributes={"occluded"})
        target_label_cat.add("a", attributes={"x"})
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    annotations=[
                        Label(0, attributes={"x": "4", "y": "2"}),
                        Bbox(
                            1, 2, 3, 4, label=0, attributes={"x": "1", "y": "1", "occluded": False}
                        ),
                    ],
                    attributes={"frame": 0},
                ),
            ],
            categories={AnnotationType.label: target_label_cat},
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatConverter.convert, allow_undeclared_attrs=True),
                test_dir,
                target_dataset=target_dataset,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_relative_paths(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", media=Image(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", media=Image(data=np.ones((5, 4, 3)))),
            ]
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image(data=np.ones((4, 2, 3))), attributes={"frame": 0}),
                DatasetItem(
                    id="subdir1/1", media=Image(data=np.ones((2, 6, 3))), attributes={"frame": 1}
                ),
                DatasetItem(
                    id="subdir2/1", media=Image(data=np.ones((5, 4, 3))), attributes={"frame": 2}
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatConverter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_media=True,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        label_categories = LabelCategories(attributes={"occluded"})
        for i in range(10):
            label_categories.add(str(i))
        label_categories.items[2].attributes.update(["a1", "a2", "empty"])

        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="s1",
                    media=Image(data=np.ones((5, 10, 3))),
                    annotations=[
                        Label(1),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: label_categories,
            },
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="s1",
                    media=Image(data=np.ones((5, 10, 3))),
                    annotations=[
                        Label(1),
                    ],
                    attributes={"frame": 0},
                ),
            ],
            categories={
                AnnotationType.label: label_categories,
            },
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatConverter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_media=True,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    "q/1",
                    media=Image(path="q/1.JPEG", data=np.zeros((4, 3, 3))),
                    attributes={"frame": 1},
                ),
                DatasetItem(
                    "a/b/c/2",
                    media=Image(path="a/b/c/2.bmp", data=np.zeros((3, 4, 3))),
                    attributes={"frame": 2},
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected,
                partial(CvatConverter.convert, save_media=True),
                test_dir,
                require_media=True,
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "images", "q", "1.JPEG")))
            self.assertTrue(osp.isfile(osp.join(test_dir, "images", "a", "b", "c", "2.bmp")))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preserve_frame_ids(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="some/name1", media=Image(data=np.ones((4, 2, 3))), attributes={"frame": 40}
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset, CvatConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="some/name1", media=Image(data=np.ones((4, 2, 3))), attributes={"frame": 40}
                ),
            ]
        )

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="some/name1", media=Image(data=np.ones((4, 2, 3))), attributes={"frame": 0}
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatConverter.convert, reindex=True),
                test_dir,
                target_dataset=expected_dataset,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="a"),
                DatasetItem(2, subset="a", media=Image(data=np.ones((3, 2, 3)))),
                DatasetItem(2, subset="b"),
            ],
            categories=[],
        )

        with TestDir() as path:
            # generate initial dataset
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(1, subset="a"),
                    DatasetItem(2, subset="b"),
                    DatasetItem(3, subset="c", media=Image(data=np.ones((3, 2, 3)))),
                ]
            )
            dataset.export(path, "cvat", save_media=True)

            dataset.put(DatasetItem(2, subset="a", media=Image(data=np.ones((3, 2, 3)))))
            dataset.remove(3, "c")
            dataset.save(save_media=True)

            self.assertEqual({"a.xml", "b.xml", "images"}, set(os.listdir(path)))
            self.assertEqual({"2.jpg"}, set(os.listdir(osp.join(path, "images"))))
            compare_datasets(
                self,
                expected,
                Dataset.import_from(path, "cvat"),
                require_media=True,
                ignored_attrs={"frame"},
            )
