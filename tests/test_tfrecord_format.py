import os
import os.path as osp
from functools import partial
from unittest import TestCase, skipIf

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories, Mask
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import ByteImage, Image
from datumaro.util.image import encode_image
from datumaro.util.test_utils import TestDir, check_save_and_load, compare_datasets
from datumaro.util.tf_util import check_import

from .requirements import Requirements, mark_requirement

try:
    from datumaro.plugins.tf_detection_api_format.converter import TfDetectionApiConverter
    from datumaro.plugins.tf_detection_api_format.extractor import (
        TfDetectionApiExtractor,
        TfDetectionApiImporter,
    )

    import_failed = False
except ImportError:
    import_failed = True

    import importlib

    module_found = importlib.util.find_spec("tensorflow") is not None

    @skipIf(not module_found, "Tensorflow package is not found")
    class TfImportTest(TestCase):
        @mark_requirement(Requirements.DATUM_GENERAL_REQ)
        def test_raises_when_crashes_on_import(self):
            # Should fire if import can't be done for any reason except
            # module unavailability and import crash
            with self.assertRaisesRegex(ImportError, "Test process exit code"):
                check_import()


@skipIf(import_failed, "Failed to import tensorflow")
class TfrecordConverterTest(TestCase):
    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="tf_detection_api",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_bboxes(self):
        test_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((16, 16, 3))),
                    annotations=[
                        Bbox(0, 4, 4, 8, label=2),
                        Bbox(0, 4, 4, 4, label=3),
                        Bbox(2, 4, 4, 4),
                    ],
                    attributes={"source_id": ""},
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                test_dataset, partial(TfDetectionApiConverter.convert, save_media=True), test_dir
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_masks(self):
        test_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((4, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array(
                                [
                                    [1, 0, 0, 1],
                                    [0, 1, 1, 0],
                                    [0, 1, 1, 0],
                                    [1, 0, 0, 1],
                                ]
                            ),
                            label=1,
                        ),
                    ],
                    attributes={"source_id": ""},
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                test_dataset, partial(TfDetectionApiConverter.convert, save_masks=True), test_dir
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_no_subsets(self):
        test_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image(data=np.ones((16, 16, 3))),
                    annotations=[
                        Bbox(2, 1, 4, 4, label=2),
                        Bbox(4, 2, 8, 4, label=3),
                    ],
                    attributes={"source_id": ""},
                ),
                DatasetItem(
                    id=2,
                    media=Image(data=np.ones((8, 8, 3)) * 2),
                    annotations=[
                        Bbox(4, 4, 4, 4, label=3),
                    ],
                    attributes={"source_id": ""},
                ),
                DatasetItem(
                    id=3, media=Image(data=np.ones((8, 4, 3)) * 3), attributes={"source_id": ""}
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                test_dataset, partial(TfDetectionApiConverter.convert, save_media=True), test_dir
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        test_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    media=Image(data=np.ones((16, 16, 3))),
                    annotations=[
                        Bbox(2, 1, 4, 4, label=2),
                        Bbox(4, 2, 8, 4, label=3),
                    ],
                    attributes={"source_id": ""},
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                test_dataset, partial(TfDetectionApiConverter.convert, save_media=True), test_dir
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self):
        test_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1/q.e",
                    media=Image(path="1/q.e", size=(10, 15)),
                    attributes={"source_id": ""},
                )
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(test_dataset, TfDetectionApiConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_unknown_image_formats(self):
        test_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=ByteImage(data=encode_image(np.ones((5, 4, 3)), "png"), path="1/q.e"),
                    attributes={"source_id": ""},
                ),
                DatasetItem(
                    id=2,
                    media=ByteImage(data=encode_image(np.ones((6, 4, 3)), "png"), ext="qwe"),
                    attributes={"source_id": ""},
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                test_dataset,
                partial(TfDetectionApiConverter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "q/1",
                    subset="train",
                    media=Image(path="q/1.JPEG", data=np.zeros((4, 3, 3))),
                    attributes={"source_id": ""},
                ),
                DatasetItem(
                    "a/b/c/2",
                    subset="valid",
                    media=Image(path="a/b/c/2.bmp", data=np.zeros((3, 4, 3))),
                    attributes={"source_id": ""},
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                dataset,
                partial(TfDetectionApiConverter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            # generate initial dataset
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(1, subset="a", media=Image(data=np.ones((2, 3, 3)))),
                    DatasetItem(2, subset="b", media=Image(data=np.ones((2, 4, 3)))),
                    DatasetItem(3, subset="c", media=Image(data=np.ones((2, 5, 3)))),
                ]
            )
            dataset.export(path, "tf_detection_api", save_media=True)
            os.unlink(osp.join(path, "a.tfrecord"))
            os.unlink(osp.join(path, "b.tfrecord"))
            os.unlink(osp.join(path, "c.tfrecord"))

            dataset.put(DatasetItem(2, subset="a", media=Image(data=np.ones((3, 2, 3)))))
            dataset.remove(3, "c")
            dataset.save(save_media=True)

            self.assertTrue(osp.isfile(osp.join(path, "a.tfrecord")))
            self.assertFalse(osp.isfile(osp.join(path, "b.tfrecord")))
            self.assertTrue(osp.isfile(osp.join(path, "c.tfrecord")))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_labelmap_parsing(self):
        text = """
            {
                id: 4
                name: 'qw1'
            }
            {
                id: 5 name: 'qw2'
            }

            {
                name: 'qw3'
                id: 6
            }
            {name:'qw4' id:7}
        """
        expected = {
            "qw1": 4,
            "qw2": 5,
            "qw3": 6,
            "qw4": 7,
        }
        parsed = TfDetectionApiExtractor._parse_labelmap(text)

        self.assertEqual(expected, parsed)


DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "tf_detection_api_dataset")


@skipIf(import_failed, "Failed to import tensorflow")
class TfrecordImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertIn(TfDetectionApiImporter.NAME, detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((16, 16, 3))),
                    annotations=[
                        Bbox(0, 4, 4, 8, label=2),
                        Bbox(0, 4, 4, 4, label=3),
                        Bbox(2, 4, 4, 4),
                    ],
                    attributes={"source_id": "1"},
                ),
                DatasetItem(
                    id=2,
                    subset="val",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(1, 2, 4, 2, label=3),
                    ],
                    attributes={"source_id": "2"},
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image(data=np.ones((5, 4, 3)) * 3),
                    attributes={"source_id": "3"},
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(10)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "tf_detection_api")

        compare_datasets(self, target_dataset, dataset)
