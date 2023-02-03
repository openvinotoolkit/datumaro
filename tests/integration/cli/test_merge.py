import os.path as osp
from unittest import TestCase

import numpy as np

import datumaro.plugins.data_formats.voc.format as VOC
from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories, MaskCategories
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.project import Dataset, Project
from datumaro.util.test_utils import TestDir, compare_datasets
from datumaro.util.test_utils import run_datum as run

from ...requirements import Requirements, mark_requirement


class MergeTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_self_merge(self):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1),
                        Bbox(5, 6, 2, 3, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=2,
                            id=1,
                            group=1,
                            attributes={
                                "score": 0.5,
                                "occluded": False,
                                "difficult": False,
                                "truncated": False,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            3,
                            label=3,
                            id=2,
                            group=2,
                            attributes={
                                "score": 0.5,
                                "occluded": False,
                                "difficult": False,
                                "truncated": False,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            3,
                            label=1,
                            id=1,
                            group=1,
                            attributes={"score": 0.5, "is_crowd": False},
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["background", "a", "b", "c"]),
                AnnotationType.mask: MaskCategories(VOC.generate_colormap(4)),
            },
        )

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, "dataset1")
            dataset2_url = osp.join(test_dir, "dataset2")

            dataset1.export(dataset1_url, "coco", save_media=True)
            dataset2.export(dataset2_url, "voc", save_media=True)

            proj_dir = osp.join(test_dir, "proj")
            with Project.init(proj_dir) as project:
                project.import_source("source", dataset2_url, "voc")

            result_dir = osp.join(test_dir, "result")
            run(self, "merge", "-o", result_dir, "-p", proj_dir, dataset1_url + ":coco")

            compare_datasets(self, expected, Dataset.load(result_dir), require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_multimerge(self):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1),
                        Bbox(5, 6, 2, 3, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=2,
                            id=1,
                            group=1,
                            attributes={
                                "score": 0.5,
                                "occluded": False,
                                "difficult": False,
                                "truncated": False,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            3,
                            label=3,
                            id=2,
                            group=2,
                            attributes={
                                "score": 0.5,
                                "occluded": False,
                                "difficult": False,
                                "truncated": False,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            3,
                            label=1,
                            id=1,
                            group=1,
                            attributes={"score": 0.5, "is_crowd": False},
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["background", "a", "b", "c"]),
                AnnotationType.mask: MaskCategories(VOC.generate_colormap(4)),
            },
        )

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, "dataset1")
            dataset2_url = osp.join(test_dir, "dataset2")

            dataset1.export(dataset1_url, "coco", save_media=True)
            dataset2.export(dataset2_url, "voc", save_media=True)

            result_dir = osp.join(test_dir, "result")
            run(self, "merge", "-o", result_dir, dataset2_url + ":voc", dataset1_url + ":coco")

            compare_datasets(self, expected, Dataset.load(result_dir), require_media=True)

    @mark_requirement(Requirements.DATUM_542)
    def test_can_save_in_another_format(self):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1),
                        Bbox(5, 6, 2, 3, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=2),
                        Bbox(5, 6, 2, 3, label=3),
                        Bbox(1, 2, 3, 3, label=1),
                    ],
                ),
            ],
            categories=["background", "a", "b", "c"],
        )

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, "dataset1")
            dataset2_url = osp.join(test_dir, "dataset2")

            dataset1.export(dataset1_url, "coco", save_media=True)
            dataset2.export(dataset2_url, "voc", save_media=True)

            result_dir = osp.join(test_dir, "result")
            run(
                self,
                "merge",
                "-o",
                result_dir,
                "-f",
                "yolo",
                dataset2_url + ":voc",
                dataset1_url + ":coco",
                "--",
                "--save-media",
            )

            compare_datasets(
                self, expected, Dataset.import_from(result_dir, "yolo"), require_media=True
            )
