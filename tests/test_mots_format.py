from functools import partial
import numpy as np
import os.path as osp

from unittest import TestCase

from datumaro.components.extractor import DatasetItem, Mask
from datumaro.components.project import Dataset, Project
from datumaro.plugins.mots_format import MotsPngConverter, MotsImporter
from datumaro.util.test_utils import TestDir, compare_datasets

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'mots_dataset')


class MotsPngConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None):
        converter(source_dataset, test_dir)

        if importer_args is None:
            importer_args = {}
        parsed_dataset = MotsImporter()(test_dir, **importer_args).make_dataset()

        if target_dataset is None:
            target_dataset = source_dataset

        compare_datasets(self, expected=target_dataset, actual=parsed_dataset)

    def test_can_save_masks(self):
        source = Dataset.from_iterable([
            DatasetItem(id=1, subset='a', image=np.ones((5, 1)), annotations=[
                # overlapping masks, the first should be truncated
                # the first and third are different instances
                Mask(np.array([[0, 0, 0, 1, 0]]), label=3, z_order=3,
                    attributes={'track_id': 1}),
                Mask(np.array([[0, 1, 1, 1, 0]]), label=2, z_order=1,
                    attributes={'track_id': 2}),
                Mask(np.array([[1, 1, 0, 0, 0]]), label=3, z_order=2,
                    attributes={'track_id': 3}),
            ]),
            DatasetItem(id=2, subset='a', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[1, 0, 0, 0, 0]]), label=3,
                    attributes={'track_id': 2}),
            ]),
            DatasetItem(id=3, subset='b', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[0, 1, 0, 0, 0]]), label=0,
                    attributes={'track_id': 1}),
            ]),
        ], categories=['a', 'b', 'c', 'd'])

        target = Dataset.from_iterable([
            DatasetItem(id=1, subset='a', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[0, 0, 0, 1, 0]]), label=3,
                    attributes={'track_id': 1}),
                Mask(np.array([[0, 0, 1, 0, 0]]), label=2,
                    attributes={'track_id': 2}),
                Mask(np.array([[1, 1, 0, 0, 0]]), label=3,
                    attributes={'track_id': 3}),
            ]),
            DatasetItem(id=2, subset='a', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[1, 0, 0, 0, 0]]), label=3,
                    attributes={'track_id': 2}),
            ]),
            DatasetItem(id=3, subset='b', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[0, 1, 0, 0, 0]]), label=0,
                    attributes={'track_id': 1}),
            ]),
        ], categories=['a', 'b', 'c', 'd'])

        with TestDir() as test_dir:
            self._test_save_and_load(source,
                partial(MotsPngConverter.convert, save_images=True),
                test_dir, target_dataset=target)

class MotsImporterTest(TestCase):
    def test_can_detect(self):
        self.assertTrue(MotsImporter.detect(DUMMY_DATASET_DIR))

    def test_can_import(self):
        target = Dataset.from_iterable([
            DatasetItem(id=1, subset='a', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[0, 0, 0, 1, 0]]), label=3,
                    attributes={'track_id': 1}),
                Mask(np.array([[0, 0, 1, 0, 0]]), label=2,
                    attributes={'track_id': 2}),
                Mask(np.array([[1, 1, 0, 0, 0]]), label=3,
                    attributes={'track_id': 3}),
            ]),
            DatasetItem(id=2, subset='a', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[1, 0, 0, 0, 0]]), label=3,
                    attributes={'track_id': 2}),
            ]),
            DatasetItem(id=3, subset='b', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[0, 1, 0, 0, 0]]), label=0,
                    attributes={'track_id': 1}),
            ]),
        ], categories=['a', 'b', 'c', 'd'])

        parsed = Project.import_from(DUMMY_DATASET_DIR, 'mots').make_dataset()
        compare_datasets(self, expected=target, actual=parsed)