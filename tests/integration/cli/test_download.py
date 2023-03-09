import os.path as osp
from unittest import TestCase
from unittest.case import skipIf

from datumaro.components.dataset import Dataset
from datumaro.components.extractor_tfds import AVAILABLE_TFDS_DATASETS, TFDS_EXTRACTOR_AVAILABLE

from ...requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir, compare_datasets, mock_tfds_data
from tests.utils.test_utils import run_datum as run


@skipIf(not TFDS_EXTRACTOR_AVAILABLE, reason="TFDS is not installed")
class DownloadTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_download(self):
        with TestDir() as test_dir, mock_tfds_data(subsets=("train", "val")):
            expected_dataset = Dataset(AVAILABLE_TFDS_DATASETS["mnist"].make_extractor())

            run(
                self,
                "download",
                "--dataset-id=tfds:mnist",
                f"--output-dir={test_dir}",
                "--",
                "--save-media",
            )

            actual_dataset = Dataset.import_from(test_dir, "mnist")
            compare_datasets(self, expected_dataset, actual_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_download_custom_format(self):
        with TestDir() as test_dir, mock_tfds_data():
            expected_dataset = Dataset(AVAILABLE_TFDS_DATASETS["mnist"].make_extractor())

            run(
                self,
                "download",
                "--dataset-id=tfds:mnist",
                "--output-format=datumaro",
                f"--output-dir={test_dir}",
                "--",
                "--save-media",
            )

            actual_dataset = Dataset.load(test_dir)
            compare_datasets(self, expected_dataset, actual_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_download_fails_on_existing_dir_without_overwrite(self):
        with TestDir() as test_dir:
            with open(osp.join(test_dir, "text.txt"), "w"):
                pass

            run(
                self,
                "download",
                "--dataset-id=tfds:mnist",
                "--output-format=datumaro",
                f"--output-dir={test_dir}",
                expected_code=1,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_download_works_on_existing_dir_without_overwrite(self):
        with TestDir() as test_dir, mock_tfds_data():
            with open(osp.join(test_dir, "text.txt"), "w"):
                pass

            run(
                self,
                "download",
                "--dataset-id=tfds:mnist",
                "--output-format=datumaro",
                f"--output-dir={test_dir}",
                "--overwrite",
                "--",
                "--save-media",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_download_subset(self):
        with TestDir() as test_dir, mock_tfds_data(subsets=("train", "val")):
            expected_dataset = Dataset(
                AVAILABLE_TFDS_DATASETS["mnist"].make_extractor().get_subset("train")
            )

            run(
                self,
                "download",
                "--dataset-id=tfds:mnist",
                "--output-format=datumaro",
                f"--output-dir={test_dir}",
                "--subset=train",
                "--",
                "--save-media",
            )

            actual_dataset = Dataset.load(test_dir)
            compare_datasets(self, expected_dataset, actual_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_download_invalid_subset(self):
        with TestDir() as test_dir, mock_tfds_data(subsets=("train", "val")):
            run(
                self,
                "download",
                "--dataset-id=tfds:mnist",
                "--output-format=datumaro",
                f"--output-dir={test_dir}",
                "--subset=test",
                expected_code=1,
            )
