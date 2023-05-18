import os
from unittest import TestCase

import numpy as np

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.image_dir import ImageDirExporter
from datumaro.util import dump_json_file
from datumaro.util.meta_file_util import get_hashkey_file

from ..requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir, check_save_and_load, compare_hashkey_meta


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
    def test_relative_paths(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", media=Image.from_numpy(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", media=Image.from_numpy(data=np.ones((5, 4, 3)))),
            ]
        )

        with TestDir() as test_dir:
            check_save_and_load(
                self, dataset, ImageDirExporter.convert, test_dir, importer="image_dir"
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
                DatasetItem(
                    id="q/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")
                ),
                DatasetItem(
                    id="a/b/c/2", media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp")
                ),
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
    def test_can_load_hash_key(self):
        hashkey_meta = {
            "hashkey": {
                "default/1": np.zeros((1, 64), dtype=np.uint8).tolist(),
                "default/2": np.zeros((1, 64), dtype=np.uint8).tolist(),
            }
        }
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, media=Image.from_numpy(data=np.ones((10, 6, 3)))),
                DatasetItem(id=2, media=Image.from_numpy(data=np.ones((5, 4, 3)))),
            ]
        )
        with TestDir() as test_dir:
            ImageDirExporter.convert(source_dataset, test_dir, save_media=True)

            meta_file = get_hashkey_file(test_dir)
            os.makedirs(os.path.join(test_dir, "hash_key_meta"))
            dump_json_file(meta_file, hashkey_meta, indent=True)

            imported_dataset = Dataset.import_from(test_dir, "image_dir")
            compare_hashkey_meta(self, hashkey_meta, imported_dataset)
