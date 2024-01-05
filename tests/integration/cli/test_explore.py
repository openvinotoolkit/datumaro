import os.path as osp
from glob import glob
from unittest import TestCase

import numpy as np

from datumaro.cli.util.project import load_project
from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.util.meta_file_util import has_hashkey_file, parse_hashkey_file
from datumaro.util.scope import scope_add, scoped

from ...requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir
from tests.utils.test_utils import run_datum as run


class ExploreTest(TestCase):
    @property
    def test_dataset(self):
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        train_img[2, :] = 0
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)
        test_img[2, :] = 255

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(0), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(0), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image.from_numpy(data=test_img),
                    annotations=[Label(1), Caption("dog")],
                ),
            ]
        )
        return dataset

    @property
    def test_dataset2(self):
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        train_img[2, :] = 0
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)
        test_img[2, :] = 255

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=4,
                    subset="train",
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(0), Caption("cat")],
                ),
                DatasetItem(
                    id=5,
                    subset="train",
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(0), Caption("cat")],
                ),
                DatasetItem(
                    id=6,
                    subset="test",
                    media=Image.from_numpy(data=test_img),
                    annotations=[Label(1), Caption("dog")],
                ),
            ]
        )
        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_explore_dataset_w_query_img_path(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset_url = osp.join(test_dir, "dataset")
        train_image_path = osp.join(proj_dir, "source-1", "images", "train", "1.jpg")

        self.test_dataset.export(dataset_url, "datumaro", save_media=True)

        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset_url)
        run(
            self,
            "explore",
            "source-1",
            "--query-img-path",
            train_image_path,
            "-topk",
            "2",
            "-p",
            proj_dir,
            "-s",
        )

        saved_result_path = osp.join(proj_dir, "explore_result")
        results = glob(osp.join(saved_result_path, "**", "*"), recursive=True)

        self.assertIn(osp.join(saved_result_path, "train", "1.jpg"), results)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_explore_dataset_w_query_item_id(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset_url = osp.join(test_dir, "dataset")

        self.test_dataset.export(dataset_url, "datumaro", save_media=True)

        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset_url)
        run(
            self,
            "explore",
            "source-1",
            "--query-item-id",
            "1",
            "-topk",
            "2",
            "-p",
            proj_dir,
            "-s",
        )

        saved_result_path = osp.join(proj_dir, "explore_result")
        results = glob(osp.join(saved_result_path, "**", "*"), recursive=True)

        self.assertIn(osp.join(saved_result_path, "train", "1.jpg"), results)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_explore_dataset_w_query_str(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset_url = osp.join(test_dir, "dataset")

        self.test_dataset.export(dataset_url, "datumaro", save_media=True)

        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset_url)
        run(
            self,
            "explore",
            "source-1",
            "--query-str",
            "a photo of a upper white and bottom black background",
            "-topk",
            "2",
            "-p",
            proj_dir,
            "-s",
        )

        saved_result_path = osp.join(proj_dir, "explore_result")
        results = glob(osp.join(saved_result_path, "**", "*"), recursive=True)

        self.assertIn(osp.join(saved_result_path, "train", "1.jpg"), results)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_explore_dataset_wo_target(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset_url = osp.join(test_dir, "dataset")
        train_image_path = osp.join(proj_dir, "source-1", "images", "train", "1.jpg")

        self.test_dataset.export(dataset_url, "datumaro", save_media=True)

        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset_url)
        run(
            self,
            "explore",
            "--query-img-path",
            train_image_path,
            "-topk",
            "2",
            "-p",
            proj_dir,
            "-s",
        )

        saved_result_path = osp.join(proj_dir, "explore_result")
        results = glob(osp.join(saved_result_path, "**", "*"), recursive=True)

        self.assertIn(osp.join(saved_result_path, "train", "1.jpg"), results)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_explore_added_dataset_w_target(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset1_url = osp.join(test_dir, "dataset1")
        train_image_path = osp.join(proj_dir, "source-1", "images", "train", "1.jpg")

        self.test_dataset.export(dataset1_url, "datumaro", save_media=True)
        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset1_url)

        run(
            self,
            "explore",
            "source-1",
            "--query-img-path",
            train_image_path,
            "-topk",
            "2",
            "-p",
            proj_dir,
        )

        dataset2_url = osp.join(proj_dir, "dataset2")
        self.test_dataset2.save(dataset2_url, save_media=True)
        run(self, "project", "add", "-p", proj_dir, "-f", "datumaro", dataset2_url)
        run(
            self,
            "explore",
            "source-1",
            "--query-img-path",
            train_image_path,
            "-topk",
            "2",
            "-p",
            proj_dir,
            "-s",
        )

        saved_result_path = osp.join(proj_dir, "explore_result")
        results = glob(osp.join(saved_result_path, "**", "*"), recursive=True)

        self.assertIn(osp.join(saved_result_path, "train", "1.jpg"), results)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_explore_added_dataset_wo_target(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset1_url = osp.join(test_dir, "dataset1")
        train_image_path = osp.join(proj_dir, "source-1", "images", "train", "1.jpg")

        self.test_dataset.export(dataset1_url, "datumaro", save_media=True)
        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset1_url)

        run(
            self,
            "explore",
            "--query-img-path",
            train_image_path,
            "-topk",
            "2",
            "-p",
            proj_dir,
        )

        dataset2_url = osp.join(proj_dir, "source-2")
        self.test_dataset2.save(dataset2_url, save_media=True)
        run(self, "project", "add", "-p", proj_dir, "-f", "datumaro", dataset2_url)
        run(
            self,
            "explore",
            "--query-img-path",
            train_image_path,
            "-topk",
            "4",
            "-p",
            proj_dir,
            "-s",
        )

        saved_result_path = osp.join(proj_dir, "explore_result")
        results = glob(osp.join(saved_result_path, "**", "*"), recursive=True)

        self.assertIn(osp.join(saved_result_path, "train", "1.jpg"), results)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_explore_merged_dataset_w_target(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset1_url = osp.join(test_dir, "dataset1")
        train_image_path = osp.join(proj_dir, "source-1", "images", "train", "1.jpg")

        self.test_dataset.export(dataset1_url, "datumaro", save_media=True)

        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset1_url)
        run(
            self,
            "explore",
            "--query-img-path",
            train_image_path,
            "-topk",
            "2",
            "-p",
            proj_dir,
            "source-1",
        )

        dataset2_url = osp.join(test_dir, "dataset2")
        self.test_dataset2.save(dataset2_url, save_media=True)
        result_dir = osp.join(test_dir, "result")

        run(
            self,
            "merge",
            "-f",
            "datumaro",
            "-o",
            result_dir,
            dataset1_url,
            dataset2_url,
            "--",
            "--save-media",
        )
        run(self, "project", "import", "-n", "result", "-p", proj_dir, "-f", "datumaro", result_dir)
        train_image_path = osp.join(proj_dir, "result", "images", "train", "1.jpg")
        run(
            self,
            "explore",
            "result",
            "--query-img-path",
            train_image_path,
            "-topk",
            "4",
            "-p",
            proj_dir,
            "-s",
        )

        saved_result_path = osp.join(proj_dir, "explore_result")
        results = glob(osp.join(saved_result_path, "**", "*"), recursive=True)

        self.assertIn(osp.join(saved_result_path, "train", "1.jpg"), results)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_explore_merged_dataset_wo_target(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset1_url = osp.join(test_dir, "dataset1")
        self.test_dataset.export(dataset1_url, "datumaro", save_media=True)

        train_image_path = osp.join(proj_dir, "source-1", "images", "train", "1.jpg")
        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset1_url)
        run(self, "explore", "--query-img-path", train_image_path, "-topk", "2", "-p", proj_dir)

        dataset2_url = osp.join(test_dir, "dataset2")
        self.test_dataset2.save(dataset2_url, save_media=True)
        result_dir = osp.join(test_dir, "result")

        run(
            self,
            "merge",
            "-f",
            "datumaro",
            "-o",
            result_dir,
            dataset1_url,
            dataset2_url,
        )
        run(self, "project", "import", "-n", "result", "-p", proj_dir, "-f", "datumaro", result_dir)
        run(
            self,
            "explore",
            "--query-img-path",
            train_image_path,
            "-topk",
            "4",
            "-p",
            proj_dir,
            "-s",
        )

        saved_result_path = osp.join(proj_dir, "explore_result")
        results = glob(osp.join(saved_result_path, "**", "*"), recursive=True)

        self.assertIn(osp.join(saved_result_path, "train", "1.jpg"), results)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_checkout_load_hashkey(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset1_url = osp.join(test_dir, "dataset1")
        train_image_path = osp.join(dataset1_url, "images", "train", "1.jpg")

        self.test_dataset.export(dataset1_url, "datumaro", save_media=True)
        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset1_url)
        run(
            self,
            "explore",
            "source-1",
            "--query-img-path",
            train_image_path,
            "-topk",
            "2",
            "-p",
            proj_dir,
        )
        run(self, "project", "commit", "-p", proj_dir, "-m", "commit1")

        commit1_proj = load_project(proj_dir)
        commit1_srcs = list(commit1_proj.working_tree.config.sources.keys())
        self.assertTrue(len(commit1_srcs), 1)
        src_dir = commit1_proj.source_data_dir(commit1_srcs[0])
        self.assertTrue(has_hashkey_file(src_dir))
        commit1_hashkey = parse_hashkey_file(src_dir)

        # check stage added
        new_tree = commit1_proj.working_tree.clone()
        stage = new_tree.build_targets.add_explore_stage("source-1", params={"save_hashkey": True})
        self.assertTrue(stage in new_tree.build_targets)

        dataset2_url = osp.join(test_dir, "dataset2")
        self.test_dataset2.save(dataset2_url, save_media=True)
        result_dir = osp.join(test_dir, "result")

        run(
            self,
            "merge",
            "-f",
            "datumaro",
            "-o",
            result_dir,
            "-p",
            proj_dir,
            dataset2_url + ":datumaro",
        )
        run(self, "project", "import", "-n", "result", "-p", proj_dir, "-f", "datumaro", result_dir)
        run(
            self,
            "explore",
            "result",
            "--query-img-path",
            train_image_path,
            "-topk",
            "2",
            "-p",
            proj_dir,
        )

        run(self, "project", "commit", "-p", proj_dir, "-m", "commit2")
        commit2_proj = load_project(proj_dir)
        commit2_srcs = list(commit2_proj.working_tree.config.sources.keys())
        self.assertTrue(len(commit2_srcs), 2)

        src_dir = commit2_proj.source_data_dir("result")
        self.assertTrue(has_hashkey_file(src_dir))
        commit2_hashkey = parse_hashkey_file(src_dir)

        self.assertTrue(len(commit2_hashkey) > len(commit1_hashkey))

        run(self, "project", "checkout", "-p", proj_dir, "HEAD~1")
        checkout_proj = load_project(proj_dir)
        checkout_srcs = list(checkout_proj.working_tree.config.sources.keys())
        self.assertTrue(len(checkout_srcs), 1)
        src_dir = checkout_proj.source_data_dir(checkout_srcs[0])
        checkout_hashkey = parse_hashkey_file(src_dir)

        self.assertEqual(len(checkout_hashkey), len(commit1_hashkey))
        self.assertEqual(checkout_hashkey["train/1"], commit1_hashkey["train/1"])
        self.assertEqual(checkout_hashkey["train/2"], commit1_hashkey["train/2"])
        self.assertEqual(checkout_hashkey["test/3"], commit1_hashkey["test/3"])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_parse_hashkey(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset1_url = osp.join(test_dir, "dataset1")
        train_image_path = osp.join(dataset1_url, "images", "train", "1.jpg")

        self.test_dataset.export(dataset1_url, "datumaro", save_media=True)
        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset1_url)
        run(self, "explore", "--query-img-path", train_image_path, "-topk", "2", "-p", proj_dir)

        proj = load_project(proj_dir)
        srcs = list(proj.working_tree.config.sources.keys())
        src_dir = proj.source_data_dir(srcs[0])
        hashkey = parse_hashkey_file(src_dir)

        for item in self.test_dataset:
            item_id = item.id
            item_subset = item.subset
            self.assertIsNotNone(hashkey[item_subset + "/" + item_id])
