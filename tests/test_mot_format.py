from functools import partial
import numpy as np
import os.path as osp

from unittest import TestCase
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (DatasetItem,
    AnnotationType, Bbox, LabelCategories
)
from datumaro.plugins.mot_format import MotSeqGtConverter, MotSeqImporter
from datumaro.util.test_utils import (TestDir, compare_datasets,
    test_save_and_load)


class MotConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None):
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='mot_seq',
            target_dataset=target_dataset, importer_args=importer_args)

    def test_can_save_bboxes(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                image=np.ones((16, 16, 3)),
                annotations=[
                    Bbox(0, 4, 4, 8, label=2, attributes={
                        'occluded': True,
                    }),
                    Bbox(0, 4, 4, 4, label=3, attributes={
                        'visibility': 0.4,
                    }),
                    Bbox(2, 4, 4, 4, attributes={
                        'ignored': True
                    }),
                ]
            ),

            DatasetItem(id=2, subset='val',
                image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(1, 2, 4, 2, label=3),
                ]
            ),

            DatasetItem(id=3, subset='test',
                image=np.ones((5, 4, 3)) * 3,
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=1,
                image=np.ones((16, 16, 3)),
                annotations=[
                    Bbox(0, 4, 4, 8, label=2, attributes={
                        'occluded': True,
                        'visibility': 0.0,
                        'ignored': False,
                    }),
                    Bbox(0, 4, 4, 4, label=3, attributes={
                        'occluded': False,
                        'visibility': 0.4,
                        'ignored': False,
                    }),
                    Bbox(2, 4, 4, 4, attributes={
                        'occluded': False,
                        'visibility': 1.0,
                        'ignored': True,
                    }),
                ]
            ),

            DatasetItem(id=2,
                image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(1, 2, 4, 2, label=3, attributes={
                        'occluded': False,
                        'visibility': 1.0,
                        'ignored': False,
                    }),
                ]
            ),

            DatasetItem(id=3,
                image=np.ones((5, 4, 3)) * 3,
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(MotSeqGtConverter.convert, save_images=True),
                test_dir, target_dataset=target_dataset)


DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'mot_dataset')

class MotImporterTest(TestCase):
    def test_can_detect(self):
        self.assertTrue(MotSeqImporter.detect(DUMMY_DATASET_DIR))

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1,
                image=np.ones((16, 16, 3)),
                annotations=[
                    Bbox(0, 4, 4, 8, label=2, attributes={
                        'occluded': False,
                        'visibility': 1.0,
                        'ignored': False,
                    }),
                ]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'mot_seq')

        compare_datasets(self, expected_dataset, dataset)