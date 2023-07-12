import pytest

from datumaro.components.annotation import AnnotationType, TabularCategories
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.plugins.data_formats.tabular import *

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path


@pytest.fixture()
def fxt_tabular_root():
    yield get_test_asset_path("tabular_dataset")


@pytest.fixture()
def fxt_tabular_buddy(fxt_tabular_root):
    yield osp.join(fxt_tabular_root, "adopt-a-buddy")


@pytest.fixture()
def fxt_tabular_electricity(fxt_tabular_root):
    yield osp.join(fxt_tabular_root, "electricity.csv")


class TabularImporterTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_tabular_file(self, fxt_tabular_electricity: str) -> None:
        dataset = Dataset.import_from(fxt_tabular_electricity, "tabular")

        expected_categories = {
            AnnotationType.tabular: TabularCategories.from_iterable(
                [("class", str, {"UP", "DOWN"})]
            )
        }
        expected_subset = "electricity"

        assert dataset.categories() == expected_categories
        assert len(dataset) == 100
        assert set(dataset.subsets()) == {expected_subset}

        for idx, item in enumerate(dataset):
            assert idx == item.media.index
            assert len(item.annotations) == 1
            assert item.media.data()["class"] == item.annotations[0].values["class"]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_tabular_folder(self, fxt_tabular_buddy: str) -> None:
        dataset = Dataset.import_from(
            fxt_tabular_buddy, "tabular", target=["breed_category", "pet_category"]
        )

        expected_categories = {
            AnnotationType.tabular: TabularCategories.from_iterable(
                [("breed_category", float), ("pet_category", int)]
            )
        }

        assert dataset.categories() == expected_categories
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
    def test_can_detect_tabular(self, fxt_tabular_buddy: str) -> None:
        detected_formats = Environment().detect_dataset(fxt_tabular_buddy)
        assert TabularDataImporter.NAME in detected_formats
