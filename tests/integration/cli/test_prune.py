import os.path as osp
from collections import Counter

import numpy as np
import pytest

from datumaro.cli.util.project import parse_dataset_pathspec
from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.util.scope import scope_add, scoped

from ...requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestCaseHelper, TestDir
from tests.utils.test_utils import run_datum as run


class PruneTest:
    @pytest.fixture
    def fxt_dataset(self) -> Dataset:
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image.from_numpy(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
                DatasetItem(
                    id=4,
                    subset="test",
                    media=Image.from_numpy(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ],
            categories=["1", "2"],
        )
        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_prune_dataset_w_target(self, helper_tc: TestCaseHelper, fxt_dataset):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset_url = osp.join(test_dir, "dataset")

        fxt_dataset.export(dataset_url, "datumaro", save_media=True)

        run(helper_tc, "project", "create", "-o", proj_dir)
        run(helper_tc, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset_url)
        run(
            helper_tc,
            "prune",
            "source-1",
            "-m",
            "random",
            "-r",
            "0.5",
            "-p",
            proj_dir,
        )

        parsed_dataset = parse_dataset_pathspec(proj_dir)
        result_subsets = [item.subset for item in parsed_dataset]
        assert Counter(result_subsets) == {"test": 1, "train": 1}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_prune_dataset_wo_target(self, helper_tc: TestCaseHelper, fxt_dataset):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset_url = osp.join(test_dir, "dataset")

        fxt_dataset.export(dataset_url, "datumaro", save_media=True)

        run(helper_tc, "project", "create", "-o", proj_dir)
        run(helper_tc, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset_url)
        run(helper_tc, "prune", "-m", "random", "-r", "0.5", "-p", proj_dir)

        parsed_dataset = parse_dataset_pathspec(proj_dir)
        result_subsets = [item.subset for item in parsed_dataset]
        assert Counter(result_subsets) == {"test": 1, "train": 1}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_prune_wo_overwrite(self, helper_tc: TestCaseHelper, fxt_dataset):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")
        dataset_url = osp.join(test_dir, "dataset")
        dst_dir = osp.join(test_dir, "result")

        fxt_dataset.export(dataset_url, "datumaro", save_media=True)

        run(helper_tc, "project", "create", "-o", proj_dir)
        run(helper_tc, "project", "import", "-p", proj_dir, "-f", "datumaro", dataset_url)
        run(helper_tc, "prune", "-m", "random", "-r", "0.5", "-p", proj_dir, "-o", dst_dir)

        parsed_dataset = parse_dataset_pathspec(dst_dir)
        result_subsets = [item.subset for item in parsed_dataset]
        assert Counter(result_subsets) == {"test": 1, "train": 1}
