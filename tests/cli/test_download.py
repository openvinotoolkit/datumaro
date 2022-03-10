import os.path as osp
from unittest import TestCase
from unittest.case import skipIf

from datumaro.components.dataset import Dataset
from datumaro.components.extractor_tfds import TFDS_EXTRACTOR_AVAILABLE, make_tfds_extractor
from datumaro.util.test_utils import TestDir, compare_datasets, mock_tfds_data
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement


@skipIf(not TFDS_EXTRACTOR_AVAILABLE, reason="TFDS is not installed")
class DownloadTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_download(self):
        with TestDir() as test_dir, mock_tfds_data():
            expected_dataset = Dataset(make_tfds_extractor("mnist"))

            run(self, "download", "-i", "tfds:mnist", "-o", test_dir, "--", "--save-media")

            actual_dataset = Dataset.import_from(test_dir, "mnist")
            compare_datasets(self, expected_dataset, actual_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_download_custom_format(self):
        with TestDir() as test_dir, mock_tfds_data():
            expected_dataset = Dataset(make_tfds_extractor("mnist"))

            run(
                self,
                "download",
                "-i",
                "tfds:mnist",
                "-f",
                "datumaro",
                "-o",
                test_dir,
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
                "-i",
                "tfds:mnist",
                "-f",
                "datumaro",
                "-o",
                test_dir,
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
                "-i",
                "tfds:mnist",
                "-f",
                "datumaro",
                "-o",
                test_dir,
                "--overwrite",
                "--",
                "--save-media",
            )
