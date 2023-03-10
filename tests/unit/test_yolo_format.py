import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    AnnotationImportError,
    DatasetExportError,
    DatasetImportError,
    InvalidAnnotationError,
    ItemImportError,
    UndeclaredLabelError,
)
from datumaro.components.media import Image
from datumaro.plugins.data_formats.yolo.base import YoloBase, YoloImporter
from datumaro.plugins.data_formats.yolo.exporter import YoloExporter
from datumaro.util.image import save_image

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets, compare_datasets_strict


class YoloExportertTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(0, 1, 2, 3, label=4),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                        Bbox(2, 1, 2, 3, label=4),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2),
                        Bbox(0, 2, 3, 2, label=5),
                        Bbox(0, 2, 4, 2, label=6),
                        Bbox(0, 7, 3, 2, label=7),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(path="1.jpg", size=(10, 15)),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloExporter.convert(source_dataset, test_dir)

            save_image(
                osp.join(test_dir, "obj_train_data", "1.jpg"), np.ones((10, 15, 3))
            )  # put the image for dataset
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_dataset_with_exact_image_info(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(path="1.jpg", size=(10, 15)),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloExporter.convert(source_dataset, test_dir)

            parsed_dataset = Dataset.import_from(test_dir, "yolo", image_info={"1": (10, 15)})

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(0, 1, 2, 3, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_relative_paths(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", media=Image(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", subset="train", media=Image(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", subset="train", media=Image(data=np.ones((5, 4, 3)))),
            ],
            categories=[],
        )

        for save_media in {True, False}:
            with self.subTest(save_media=save_media):
                with TestDir() as test_dir:
                    YoloExporter.convert(source_dataset, test_dir, save_media=save_media)
                    parsed_dataset = Dataset.import_from(test_dir, "yolo")

                    compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "q/1", subset="train", media=Image(path="q/1.JPEG", data=np.zeros((4, 3, 3)))
                ),
                DatasetItem(
                    "a/b/c/2",
                    subset="valid",
                    media=Image(path="a/b/c/2.bmp", data=np.zeros((3, 4, 3))),
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            YoloExporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))),
            ],
            categories=[],
        )

        with TestDir() as path:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                    DatasetItem(2, subset="train", media=Image(path="2.jpg", size=(3, 2))),
                    DatasetItem(3, subset="valid", media=Image(data=np.ones((2, 2, 3)))),
                ],
                categories=[],
            )
            dataset.export(path, "yolo", save_media=True)

            dataset.put(DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))))
            dataset.remove(3, "valid")
            dataset.save(save_media=True)

            self.assertEqual(
                {"1.txt", "2.txt", "1.jpg", "2.jpg"},
                set(os.listdir(osp.join(path, "obj_train_data"))),
            )
            self.assertEqual(set(), set(os.listdir(osp.join(path, "obj_valid_data"))))
            compare_datasets(self, expected, Dataset.import_from(path, "yolo"), require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(0, 1, 2, 3, label=4),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                        Bbox(2, 1, 2, 3, label=4),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2),
                        Bbox(0, 2, 3, 2, label=5),
                        Bbox(0, 2, 4, 2, label=6),
                        Bbox(0, 7, 3, 2, label=7),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloExporter.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_565)
    def test_can_save_and_load_with_custom_subset_name(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="anything",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2),
                        Bbox(0, 2, 3, 2, label=5),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_565)
    def test_cant_save_with_reserved_subset_name(self):
        for subset in ["backup", "classes"]:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id=3,
                        subset=subset,
                        media=Image(data=np.ones((8, 8, 3))),
                    ),
                ],
                categories=["a"],
            )

            with TestDir() as test_dir:
                with self.assertRaisesRegex(DatasetExportError, f"Can't export '{subset}' subset"):
                    YoloExporter.convert(dataset, test_dir)

    @mark_requirement(Requirements.DATUM_609)
    def test_can_save_and_load_without_path_prefix(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        with TestDir() as test_dir:
            YoloExporter.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            with open(osp.join(test_dir, "obj.data"), "r") as f:
                lines = f.readlines()
                self.assertIn("valid = valid.txt\n", lines)

            with open(osp.join(test_dir, "valid.txt"), "r") as f:
                lines = f.readlines()
                self.assertIn("obj_valid_data/3.jpg\n", lines)

            compare_datasets(self, source_dataset, parsed_dataset)


DUMMY_DATASET_DIR = get_test_asset_path("yolo_dataset", "strict")


class YoloImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([YoloImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "yolo")

        compare_datasets(self, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self):
        source = Dataset.import_from(DUMMY_DATASET_DIR, format="yolo")

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(self, source, parsed)


class YoloBaseTest(TestCase):
    def _prepare_dataset(self, path: str) -> Dataset:
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "a",
                    subset="train",
                    media=Image(np.ones((5, 10, 3))),
                    annotations=[Bbox(1, 1, 2, 4, label=0)],
                )
            ],
            categories=["test"],
        )
        dataset.export(path, "yolo", save_images=True)

        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_parse(self):
        with TestDir() as test_dir:
            expected = self._prepare_dataset(test_dir)

            actual = Dataset.import_from(test_dir, "yolo")
            compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_data_file(self):
        with TestDir() as test_dir:
            with self.assertRaisesRegex(DatasetImportError, "Can't read dataset descriptor file"):
                YoloBase(test_dir)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_ann_line_format(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            with open(osp.join(test_dir, "obj_train_data", "a.txt"), "w") as f:
                f.write("1 2 3\n")

            with self.assertRaises(AnnotationImportError) as capture:
                Dataset.import_from(test_dir, "yolo").init_cache()
            self.assertIsInstance(capture.exception.__cause__, InvalidAnnotationError)
            self.assertIn("Unexpected field count", str(capture.exception.__cause__))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_label(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            with open(osp.join(test_dir, "obj_train_data", "a.txt"), "w") as f:
                f.write("10 0.5 0.5 0.5 0.5\n")

            with self.assertRaises(AnnotationImportError) as capture:
                Dataset.import_from(test_dir, "yolo").init_cache()
            self.assertIsInstance(capture.exception.__cause__, UndeclaredLabelError)
            self.assertEqual(capture.exception.__cause__.id, "10")

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_field_type(self):
        for field, field_name in [
            (1, "bbox center x"),
            (2, "bbox center y"),
            (3, "bbox width"),
            (4, "bbox height"),
        ]:
            with self.subTest(field_name=field_name):
                with TestDir() as test_dir:
                    self._prepare_dataset(test_dir)
                    with open(osp.join(test_dir, "obj_train_data", "a.txt"), "w") as f:
                        values = [0, 0.5, 0.5, 0.5, 0.5]
                        values[field] = "a"
                        f.write(" ".join(str(v) for v in values))

                    with self.assertRaises(AnnotationImportError) as capture:
                        Dataset.import_from(test_dir, "yolo").init_cache()
                    self.assertIsInstance(capture.exception.__cause__, InvalidAnnotationError)
                    self.assertIn(field_name, str(capture.exception.__cause__))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_ann_file(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            os.remove(osp.join(test_dir, "obj_train_data", "a.txt"))

            with self.assertRaises(ItemImportError) as capture:
                Dataset.import_from(test_dir, "yolo").init_cache()
            self.assertIsInstance(capture.exception.__cause__, FileNotFoundError)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_image_info(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            os.remove(osp.join(test_dir, "obj_train_data", "a.jpg"))

            with self.assertRaises(ItemImportError) as capture:
                Dataset.import_from(test_dir, "yolo").init_cache()
            self.assertIsInstance(capture.exception.__cause__, DatasetImportError)
            self.assertIn("Can't find image info", str(capture.exception.__cause__))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_subset_info(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            os.remove(osp.join(test_dir, "train.txt"))

            with self.assertRaisesRegex(InvalidAnnotationError, "subset list file"):
                Dataset.import_from(test_dir, "yolo").init_cache()
