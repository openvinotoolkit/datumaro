import os.path as osp
from unittest import TestCase

from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import ReadonlyDatasetError
from datumaro.components.project import Project
from datumaro.util.scope import scope_add, scoped
from datumaro.util.test_utils import TestDir, compare_datasets
from datumaro.util.test_utils import run_datum as run

from ...requirements import Requirements, mark_requirement


class FilterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_filter_dataset_inplace(self):
        test_dir = scope_add(TestDir())
        Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        ).export(test_dir, "coco")

        run(self, "filter", "-e", '/item[id = "1"]', "--overwrite", test_dir + ":coco")

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0, id=1, group=1)]),
            ],
            categories=["a", "b"],
        )
        compare_datasets(
            self, expected_dataset, Dataset.import_from(test_dir, "coco"), ignored_attrs="*"
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_filter_fails_on_inplace_update_without_overwrite(self):
        with TestDir() as test_dir:
            Dataset.from_iterable(
                [
                    DatasetItem(id=1, annotations=[Bbox(1, 2, 3, 4, label=1)]),
                ],
                categories=["a", "b"],
            ).export(test_dir, "coco")

            run(self, "filter", "-e", "/item", test_dir + ":coco", expected_code=1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_filter_fails_on_inplace_update_of_stage(self):
        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, "dataset")
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(id=1, annotations=[Bbox(1, 2, 3, 4, label=1)]),
                ],
                categories=["a", "b"],
            )
            dataset.export(dataset_url, "coco", save_media=True)

            project_dir = osp.join(test_dir, "proj")
            with Project.init(project_dir) as project:
                project.import_source("source-1", dataset_url, "coco", no_cache=True)
                project.commit("first commit")

            with self.subTest("without overwrite"):
                run(
                    self,
                    "filter",
                    "-p",
                    project_dir,
                    "-e",
                    "/item",
                    "HEAD:source-1",
                    expected_code=1,
                )

            with self.subTest("with overwrite"):
                with self.assertRaises(ReadonlyDatasetError):
                    run(
                        self,
                        "filter",
                        "-p",
                        project_dir,
                        "--overwrite",
                        "-e",
                        "/item",
                        "HEAD:source-1",
                    )
