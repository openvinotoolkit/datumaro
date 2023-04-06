import os.path as osp
import shutil
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import DEFAULT_FORMAT, Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.project import Project
from datumaro.util.scope import scope_add, scoped

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets, compare_dirs
from tests.utils.test_utils import run_datum as run


class ProjectIntegrationScenarios(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_voc_as_coco(self):
        voc_dir = get_test_asset_path(
            "voc_dataset",
            "voc_dataset1",
        )

        with TestDir() as test_dir:
            result_dir = osp.join(test_dir, "coco_export")

            run(
                self,
                "convert",
                "-if",
                "voc",
                "-i",
                voc_dir,
                "-f",
                "coco",
                "-o",
                result_dir,
                "--",
                "--save-images",
                "--reindex",
                "1",
            )

            self.assertTrue(osp.isdir(result_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_export_coco_as_voc(self):
        # TODO: use subformats once importers are removed
        coco_dir = get_test_asset_path(
            "coco_dataset",
            "coco_instances",
        )

        with TestDir() as test_dir:
            run(self, "project", "create", "-o", test_dir)
            run(self, "project", "import", "-f", "coco", "-p", test_dir, coco_dir)

            result_dir = osp.join(test_dir, "voc_export")
            run(
                self,
                "project",
                "export",
                "-f",
                "voc",
                "-p",
                test_dir,
                "-o",
                result_dir,
                "--",
                "--save-images",
            )

            self.assertTrue(osp.isdir(result_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_list_project_info(self):
        coco_dir = get_test_asset_path(
            "coco_dataset",
            "coco_instances",
        )

        with TestDir() as test_dir:
            run(self, "project", "create", "-o", test_dir)
            run(self, "project", "import", "-f", "coco", "-p", test_dir, coco_dir)

            with self.subTest("on project"):
                run(self, "project", "pinfo", "-p", test_dir)

            with self.subTest("on project revision"):
                run(self, "project", "pinfo", "-p", test_dir, "HEAD")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_list_dataset_info(self):
        coco_dir = get_test_asset_path(
            "coco_dataset",
            "coco_instances",
        )

        with TestDir() as test_dir:
            run(self, "project", "create", "-o", test_dir)
            run(self, "project", "import", "-f", "coco", "-p", test_dir, coco_dir)
            run(self, "project", "commit", "-m", "first", "-p", test_dir)

            with self.subTest("on current project"):
                run(self, "project", "dinfo", "-p", test_dir)

            with self.subTest("on current project revision"):
                run(self, "project", "dinfo", "-p", test_dir, "HEAD")

            with self.subTest("on other project"):
                run(self, "project", "dinfo", test_dir)

            with self.subTest("on other project revision"):
                run(self, "project", "dinfo", test_dir + "@HEAD")

            with self.subTest("on dataset"):
                run(self, "project", "dinfo", coco_dir + ":coco")

    def test_can_use_vcs(self):
        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, "dataset")
            project_dir = osp.join(test_dir, "proj")
            result_dir = osp.join(project_dir, "result")

            Dataset.from_iterable(
                [
                    DatasetItem(
                        0,
                        media=Image.from_data(data=np.ones((1, 2, 3))),
                        annotations=[
                            Bbox(1, 1, 1, 1, label=0),
                            Bbox(2, 2, 2, 2, label=1),
                        ],
                    )
                ],
                categories=["a", "b"],
            ).save(dataset_dir, save_media=True)

            run(self, "project", "create", "-o", project_dir)
            run(self, "project", "import", "-p", project_dir, "-f", "datumaro", dataset_dir)
            run(self, "project", "commit", "-p", project_dir, "-m", "Add data")

            run(
                self,
                "project",
                "transform",
                "-p",
                project_dir,
                "-t",
                "remap_labels",
                "source-1",
                "--",
                "-l",
                "b:cat",
            )
            run(self, "project", "commit", "-p", project_dir, "-m", "Add transform")

            run(
                self,
                "filter",
                "-p",
                project_dir,
                "-e",
                '/item/annotation[label="cat"]',
                "-m",
                "i+a",
            )
            run(self, "project", "commit", "-p", project_dir, "-m", "Add filter")

            run(
                self,
                "project",
                "export",
                "-p",
                project_dir,
                "-f",
                "coco",
                "-o",
                result_dir,
                "source-1",
                "--",
                "--save-images",
            )
            parsed = Dataset.import_from(result_dir, "coco")
            compare_datasets(
                self,
                Dataset.from_iterable(
                    [
                        DatasetItem(
                            0,
                            media=Image.from_data(data=np.ones((1, 2, 3))),
                            annotations=[
                                Bbox(
                                    2,
                                    2,
                                    2,
                                    2,
                                    label=1,
                                    group=1,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                            ],
                            attributes={"id": 1},
                        )
                    ],
                    categories=["a", "cat"],
                ),
                parsed,
                require_media=True,
            )

            shutil.rmtree(result_dir, ignore_errors=True)
            run(self, "project", "checkout", "-p", project_dir, "HEAD~1")
            run(
                self,
                "project",
                "export",
                "-p",
                project_dir,
                "-f",
                "coco",
                "-o",
                result_dir,
                "--",
                "--save-images",
            )
            parsed = Dataset.import_from(result_dir, "coco")
            compare_datasets(
                self,
                Dataset.from_iterable(
                    [
                        DatasetItem(
                            0,
                            media=Image.from_data(data=np.ones((1, 2, 3))),
                            annotations=[
                                Bbox(
                                    1,
                                    1,
                                    1,
                                    1,
                                    label=0,
                                    group=1,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                                Bbox(
                                    2,
                                    2,
                                    2,
                                    2,
                                    label=1,
                                    group=2,
                                    id=2,
                                    attributes={"is_crowd": False},
                                ),
                            ],
                            attributes={"id": 1},
                        )
                    ],
                    categories=["a", "cat"],
                ),
                parsed,
                require_media=True,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_chain_transforms_in_working_tree_without_hashing(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project_dir = osp.join(test_dir, "proj")
        run(self, "project", "create", "-o", project_dir)
        run(
            self,
            "project",
            "import",
            "-p",
            project_dir,
            "-n",
            "source1",
            "--format",
            DEFAULT_FORMAT,
            source_url,
        )
        run(self, "filter", "-p", project_dir, "-e", '/item/annotation[label="b"]')
        run(self, "project", "transform", "-p", project_dir, "-t", "rename", "--", "-e", "|2|qq|")
        run(
            self,
            "project",
            "transform",
            "-p",
            project_dir,
            "-t",
            "remap_labels",
            "--",
            "-l",
            "a:cat",
            "-l",
            "b:dog",
        )

        project = scope_add(Project(project_dir))
        built_dataset = project.working_tree.make_dataset()

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem("qq", annotations=[Label(1)]),
            ],
            categories=["cat", "dog"],
        )
        compare_datasets(self, expected_dataset, built_dataset)

        with self.assertRaises(Exception):
            compare_dirs(self, source_url, project.source_data_dir("source1"))

        source1_target = project.working_tree.build_targets["source1"]
        self.assertEqual(4, len(source1_target.stages))
        self.assertEqual("", source1_target.stages[0].hash)
        self.assertEqual("", source1_target.stages[1].hash)
        self.assertEqual("", source1_target.stages[2].hash)
