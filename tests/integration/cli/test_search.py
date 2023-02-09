import os.path as osp
import unittest
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.project import Project
from datumaro.util.scope import scope_add, scoped
from datumaro.util.test_utils import TestDir
from datumaro.util.test_utils import run_datum as run

from ...requirements import Requirements, mark_requirement


class SearchTest(TestCase):
    @unittest.skip(reason="Searcher model is not uploaded in public storage")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_search_dataset(self):
        test_dir = scope_add(TestDir())

        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        train_img[2, :] = 0
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)
        test_img[2, :] = 255
        train_Image = Image(data=train_img)

        dataset_url = osp.join(test_dir, "dataset")
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
                    media=Image(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ]
        )
        dataset.export(dataset_url, "datumaro", save_media=True)

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
            "-p",
            proj_dir,
        )
