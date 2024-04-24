# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest import TestCase

import pytest

from datumaro.components.annotation import AnnotationType, TabularCategories
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.plugins.data_formats.tabular import *

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets


@pytest.fixture()
def fxt_tabular_root():
    yield get_test_asset_path("tabular_dataset")


@pytest.fixture()
def fxt_electricity(fxt_tabular_root):
    path = osp.join(fxt_tabular_root, "electricity.csv")
    yield Dataset.import_from(path, "tabular")


@pytest.fixture()
def fxt_buddy_target():
    yield {"input": "length(m)", "output": ["breed_category", "pet_category"]}


@pytest.fixture()
def fxt_buddy(fxt_tabular_root, fxt_buddy_target):
    path = osp.join(fxt_tabular_root, "adopt-a-buddy")
    yield Dataset.import_from(path, "tabular", target=fxt_buddy_target)


@pytest.mark.new
class TabularImporterTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_tabular_file(self, fxt_electricity) -> None:
        dataset: Type[Dataset] = fxt_electricity
        expected_categories = {AnnotationType.tabular: TabularCategories.from_iterable([])}
        expected_subset = "electricity"

        assert dataset.categories() == expected_categories
        assert len(dataset) == 100
        assert set(dataset.subsets()) == {expected_subset}

        for idx, item in enumerate(dataset):
            assert idx == item.media.index
            assert len(item.annotations) == 0

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_tabular_folder(self, fxt_buddy) -> None:
        dataset: Type[Dataset] = fxt_buddy
        expected_categories_keys = [("breed_category", float), ("pet_category", int)]

        assert [
            (cat.name, cat.dtype) for cat in dataset.categories()[AnnotationType.tabular].items
        ] == expected_categories_keys
        assert len(dataset) == 200
        assert set(dataset.subsets()) == {"train", "test"}

        train = dataset.get_subset("train")
        test = dataset.get_subset("test")
        assert len(train) == 100 and len(test) == 100

        for idx, item in enumerate(train):
            assert idx == item.media.index
            assert len(item.annotations) == 1
            assert (
                item.media.data()["breed_category"] == item.annotations[0].values["breed_category"]
            )
            assert item.media.data()["pet_category"] == item.annotations[0].values["pet_category"]

        for idx, item in enumerate(test):
            assert idx == item.media.index
            assert len(item.annotations) == 0  # buddy dataset has no annotations in the test set.

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_tabular(self, fxt_tabular_root: str) -> None:
        detected_formats = Environment().detect_dataset(fxt_tabular_root)
        assert [TabularDataImporter.NAME] == detected_formats
        with TestDir() as test_dir:
            detected_formats = Environment().detect_dataset(test_dir)
            assert TabularDataImporter.NAME not in detected_formats

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "fxt,target", [("fxt_electricity", None), ("fxt_buddy", "fxt_buddy_target")]
    )
    def test_can_export_tabular(self, fxt: str, target, request) -> None:
        dataset: Type[Dataset] = request.getfixturevalue(fxt)
        if isinstance(target, str):
            target = request.getfixturevalue(target)

        with TestDir() as test_dir:
            dataset.export(test_dir, "tabular")
            back_dataset = Dataset.import_from(test_dir, "tabular", target=target)
            compare_datasets(TestCase(), dataset, back_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "target, expected_media_data_keys, expected_categories_keys",
        [
            (
                {"input": "length(m)", "output": "breed_category"},
                ["length(m)", "breed_category"],
                [("breed_category", float)],
            ),
            (
                {"input": "length", "output": "breed_category"},
                ["breed_category"],
                [("breed_category", float)],
            ),
            ({"input": "length(m)", "output": "breed"}, ["length(m)"], []),
        ],
    )
    def test_target_check_in_table(
        self, fxt_tabular_root, target, expected_media_data_keys, expected_categories_keys
    ) -> None:
        path = osp.join(fxt_tabular_root, "adopt-a-buddy")
        dataset = Dataset.import_from(path, "tabular", target=target)

        assert list(dataset.__getitem__(100).media.data().keys()) == expected_media_data_keys
        assert [
            (cat.name, cat.dtype) for cat in dataset.categories()[AnnotationType.tabular].items
        ] == expected_categories_keys

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "target,expected_included_labels",
        [
            ({"input": "length(m)", "output": "breed_category"}, [True]),
            ({"input": "length(m)", "output": ["color_type", "breed_category"]}, [False, True]),
        ],
    )
    def test_target_dtype(self, fxt_tabular_root, target, expected_included_labels) -> None:
        path = osp.join(fxt_tabular_root, "adopt-a-buddy")
        dataset = Dataset.import_from(path, "tabular", target=target)

        included_lables_result = [
            False if len(cat.labels) == 0 else True
            for cat in dataset.categories()[AnnotationType.tabular].items
        ]

        assert included_lables_result == expected_included_labels
