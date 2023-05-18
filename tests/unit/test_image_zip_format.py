import os
import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, save_image
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.image_zip import ImageZipExporter, ImageZipPath
from datumaro.util import dump_json_file
from datumaro.util.meta_file_util import get_hashkey_file

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets, compare_hashkey_meta


class ImageZipExporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_267)
    def _test_can_save_and_load(self, source_dataset, test_dir, **kwargs):
        archive_path = osp.join(test_dir, kwargs.get("name", ImageZipPath.DEFAULT_ARCHIVE_NAME))
        ImageZipExporter.convert(source_dataset, test_dir, **kwargs)
        parsed_dataset = Dataset.import_from(archive_path, "image_zip")

        compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((10, 6, 3)))),
                DatasetItem(id="2", media=Image.from_numpy(data=np.ones((5, 4, 3)))),
            ]
        )

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load_with_custom_archive_name(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="img_1", media=Image.from_numpy(data=np.ones((10, 10, 3)))),
            ]
        )

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir, name="my_archive.zip")

    @mark_requirement(Requirements.DATUM_267)
    def test_relative_paths(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((10, 10, 3)))),
                DatasetItem(id="a/2", media=Image.from_numpy(data=np.ones((4, 5, 3)))),
                DatasetItem(id="a/b/3", media=Image.from_numpy(data=np.ones((20, 10, 3)))),
            ]
        )

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load_custom_compresion_method(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((5, 5, 3)))),
                DatasetItem(id="2", media=Image.from_numpy(data=np.ones((4, 3, 3)))),
            ]
        )

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir, compression="ZIP_DEFLATED")

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load_with_arbitrary_extensions(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="subset/1", media=Image.from_numpy(data=np.ones((10, 10, 3)))),
                DatasetItem(id="2", media=Image.from_numpy(data=np.ones((4, 5, 3)))),
            ]
        )

        with TestDir() as test_dir:
            save_image(osp.join(test_dir, "2.jpg"), source_dataset.get("2").media.data)
            save_image(
                osp.join(test_dir, "subset", "1.png"),
                source_dataset.get("subset/1").media.data,
                create_dir=True,
            )

            self._test_can_save_and_load(source_dataset, test_dir)


DUMMY_DATASET_DIR = get_test_asset_path("image_zip_dataset")


class ImageZipImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_267)
    def test_can_import(self):
        source_dataset = Dataset.from_iterable(
            [DatasetItem(id="1", media=Image.from_numpy(data=np.ones((10, 10, 3))))]
        )

        zip_path = osp.join(DUMMY_DATASET_DIR, "1.zip")
        parsed_dataset = Dataset.import_from(zip_path, format="image_zip")
        compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_import_from_directory(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((10, 10, 3)))),
                DatasetItem(id="2", media=Image.from_numpy(data=np.ones((5, 10, 3)))),
            ]
        )

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, format="image_zip")
        compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_hash_key(self):
        hashkey_meta = {
            "hashkey": {
                "default/subset/1": np.zeros((1, 64), dtype=np.uint8).tolist(),
                "default/2": np.zeros((1, 64), dtype=np.uint8).tolist(),
            }
        }
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="subset/1", media=Image.from_numpy(data=np.ones((10, 10, 3)))),
                DatasetItem(id="2", media=Image.from_numpy(data=np.ones((4, 5, 3)))),
            ]
        )
        with TestDir() as test_dir:
            ImageZipExporter.convert(source_dataset, test_dir, save_media=True)

            meta_file = get_hashkey_file(test_dir)
            os.makedirs(osp.join(test_dir, "hash_key_meta"))
            dump_json_file(meta_file, hashkey_meta, indent=True)

            imported_dataset = Dataset.import_from(test_dir, "image_zip")
            compare_hashkey_meta(self, hashkey_meta, imported_dataset)
