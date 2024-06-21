from unittest import TestCase

import numpy as np

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.image_dir import ImageDirExporter

from ..requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir, check_save_and_load


class ImageDirFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, media=Image.from_numpy(data=np.ones((10, 6, 3)))),
                DatasetItem(id=2, media=Image.from_numpy(data=np.ones((5, 4, 3)))),
            ]
        )

        with TestDir() as test_dir:
            check_save_and_load(
                self,
                dataset,
                ImageDirExporter.convert,
                test_dir,
                importer="image_dir",
                require_media=True,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом", media=Image.from_numpy(data=np.ones((4, 2, 3)))
                ),
            ]
        )

        with TestDir() as test_dir:
            check_save_and_load(
                self, dataset, ImageDirExporter.convert, test_dir, importer="image_dir"
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")),
                DatasetItem(id="2", media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp")),
            ]
        )

        with TestDir() as test_dir:
            check_save_and_load(
                self,
                dataset,
                ImageDirExporter.convert,
                test_dir,
                importer="image_dir",
                require_media=True,
            )
