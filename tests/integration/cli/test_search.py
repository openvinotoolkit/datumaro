import os.path as osp
import platform
from unittest import TestCase, skipIf

import numpy as np

from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.project import Project
from datumaro.util.scope import scope_add, scoped

from ...requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir
from tests.utils.test_utils import run_datum as run


class SearchTest(TestCase):
    @property
    def test_dataset(self):
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        train_img[2, :] = 0
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)
        test_img[2, :] = 255
        train_Image = Image.from_numpy(data=train_img)
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=train_Image,
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=train_Image,
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image.from_numpy(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ]
        )
        return dataset

    @property
    def test_dataset_black_white(self):
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)
        train_Image = Image.from_numpy(data=train_img)

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=4,
                    subset="train",
                    media=train_Image,
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=5,
                    subset="train",
                    media=train_Image,
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=6,
                    subset="test",
                    media=Image.from_numpy(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ]
        )
        return dataset

    @skipIf(
        platform.system() == "Darwin",
        "Segmentation fault only occurs on MacOS: "
        "https://github.com/openvinotoolkit/datumaro/actions/runs/4202399957/jobs/7324077250",
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_search_dataset(self):
        test_dir = scope_add(TestDir())
        dataset_url = osp.join(test_dir, "dataset")

        self.test_dataset.export(dataset_url, "datumaro", save_media=True)

        train_image_path = osp.join(test_dir, "train", "1.jpg")
        proj_dir = osp.join(test_dir, "proj")
        with Project.init(proj_dir) as project:
            project.import_source("source-1", dataset_url, "datumaro", no_cache=True)
            project.commit("first commit")

        run(
            self,
            "search",
            "-q",
            train_image_path,
            "-topk",
            "2",
            proj_dir,
        )

    @skipIf(
        platform.system() == "Darwin",
        "Segmentation fault only occurs on MacOS: "
        "https://github.com/openvinotoolkit/datumaro/actions/runs/4202399957/jobs/7324077250",
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_search_added_dataset(self):
        test_dir = scope_add(TestDir())
        dataset1_url = osp.join(test_dir, "dataset1")

        self.test_dataset.export(dataset1_url, "datumaro", save_media=True)

        train_image_path = osp.join(test_dir, "train", "1.jpg")
        proj_dir = osp.join(test_dir, "proj")
        run(self, "project", "create", "-o", proj_dir)
        run(self, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset1_url)

        run(
            self,
            "search",
            "-q",
            train_image_path,
            "-topk",
            "2",
            proj_dir,
        )

        dataset2_url = osp.join(proj_dir, "dataset2")
        self.test_dataset_black_white.save(dataset2_url, save_media=True)

        run(
            self,
            "project",
            "add",
            "-f",
            "datumaro",
            "-p",
            proj_dir,
            dataset2_url,
        )

        run(
            self,
            "search",
            "-q",
            train_image_path,
            "-topk",
            "2",
            proj_dir,
        )

        result_dir = osp.join(proj_dir, "result")
        run(
            self,
            "project",
            "export",
            "-p",
            proj_dir,
            "-f",
            "datumaro",
            "-o",
            result_dir,
        )
        parsed = Dataset.import_from(result_dir, "datumaro")
