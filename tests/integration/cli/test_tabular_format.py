# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import pytest

from datumaro.components.dataset import Dataset
from datumaro.plugins.data_formats.tabular import *

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestCaseHelper, TestDir, compare_datasets
from tests.utils.test_utils import run_datum as run


@pytest.fixture()
def fxt_tabular_root():
    yield get_test_asset_path("tabular_dataset")


@pytest.fixture()
def fxt_electricity_path(fxt_tabular_root):
    yield osp.join(fxt_tabular_root, "electricity.csv")


@pytest.fixture()
def txf_electricity(fxt_electricity_path):
    yield Dataset.import_from(fxt_electricity_path, "tabular")


@pytest.fixture()
def fxt_buddy_path(fxt_tabular_root):
    yield osp.join(fxt_tabular_root, "adopt-a-buddy")


@pytest.fixture()
def fxt_buddy_target():
    yield {"input": "length(m)", "output": ["breed_category", "pet_category"]}


@pytest.fixture()
def fxt_buddy(fxt_buddy_path, fxt_buddy_target):
    yield Dataset.import_from(fxt_buddy_path, "tabular", target=fxt_buddy_target)


@pytest.mark.new
class TabularIntegrationTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "input_string,expected_result",
        [
            ("input:date,output:class", {"input": ["date"], "output": ["class"]}),
            (
                "input:length(m),output:breed_category,pet_category",
                {"input": ["length(m)"], "output": ["breed_category", "pet_category"]},
            ),
            ("input:age,color,output:size", {"input": ["age", "color"], "output": ["size"]}),
            ("input:height", {"input": ["height"]}),
            ("output:breed_category", {"output": ["breed_category"]}),
        ],
    )
    def test_string_to_dict(self, input_string, expected_result):
        assert string_to_dict(input_string) == expected_result

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "fxt_dataset, fxt_path, fxt_target",
        [
            ("txf_electricity", "fxt_electricity_path", None),
            ("fxt_buddy", "fxt_buddy_path", "fxt_buddy_target"),
        ],
    )
    def test_can_import_and_export_tabular_dataset(
        self, helper_tc: TestCaseHelper, fxt_dataset, fxt_path, fxt_target, request
    ):
        """
        <b>Description:</b>
        Ensure that the tabular dataset can be imported and exported.

        <b>Expected results:</b>
        A tabular dataset can be imported and exported.

        <b>Steps:</b>
        1. Get path to the source dataset from assets.
        2. Create a datumaro project and add source dataset to it.
        3. Export the project to a tabular dataset with `export` command.
        4. Verify that the resulting dataset is equal to the expected result.
        """

        dataset = request.getfixturevalue(fxt_dataset)
        path = request.getfixturevalue(fxt_path)
        target = request.getfixturevalue(fxt_target) if isinstance(fxt_target, str) else None
        string_target = "input:length(m),output:breed_category,pet_category"

        with TestDir() as test_dir:
            run(helper_tc, "project", "create", "-o", test_dir)
            args = ["project", "import", "-p", test_dir, "-f", "tabular", path]
            if target:
                args.extend(["--", "--target", string_target])
            run(helper_tc, *args)

            export_dir = osp.join(test_dir, "export_dir")
            run(
                helper_tc,
                "project",
                "export",
                "-p",
                test_dir,
                "-o",
                export_dir,
                "-f",
                "tabular",
            )
            exported = Dataset.import_from(export_dir, format="tabular", target=target)
            compare_datasets(helper_tc, dataset, exported)
