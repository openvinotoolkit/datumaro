from functools import partial
import numpy as np
import os.path as osp

from unittest import TestCase

from datumaro.components.extractor import DatasetItem, Mask
from datumaro.components.dataset import Dataset
from datumaro.plugins.mots_format import MotsPngConverter, MotsImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import (TestDir, compare_datasets,
    check_save_and_load)

import pytest
from tests.requirements import Requirements
from tests.requirements import DatumaroComponent

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'mots_dataset')


@pytest.mark.components(DatumaroComponent.Datumaro)
class MotsPngConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None, **kwargs):
        return check_save_and_load(self, source_dataset, converter, test_dir,
            importer='mots',
            target_dataset=target_dataset, importer_args=importer_args, **kwargs)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
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

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source = Dataset.from_iterable([
            DatasetItem(id='кириллица с пробелом', subset='a',
                image=np.ones((5, 1)), annotations=[
                    Mask(np.array([[1, 0, 0, 0, 0]]), label=0,
                        attributes={'track_id': 2}),
            ]),
        ], categories=['a'])

        with TestDir() as test_dir:
            self._test_save_and_load(source,
                partial(MotsPngConverter.convert, save_images=True),
                test_dir, require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable([
            DatasetItem('q/1', image=Image(
                path='q/1.JPEG', data=np.zeros((4, 3, 3))),
                annotations=[
                    Mask(np.array([[0, 1, 0, 0, 0]]), label=0,
                        attributes={'track_id': 1}),
                ]
            ),
            DatasetItem('a/b/c/2', image=Image(
                path='a/b/c/2.bmp', data=np.zeros((3, 4, 3))),
                annotations=[
                    Mask(np.array([[0, 1, 0, 0, 0]]), label=0,
                        attributes={'track_id': 1}),
                ]
            ),
        ], categories=['a'])

        with TestDir() as test_dir:
            self._test_save_and_load(expected,
                partial(MotsPngConverter.convert, save_images=True),
                test_dir, require_images=True)

@pytest.mark.components(DatumaroComponent.Datumaro)
class MotsImporterTest(TestCase):
    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_detect(self):
        self.assertTrue(MotsImporter.detect(DUMMY_DATASET_DIR))

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_import(self):
        target = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[0, 0, 0, 1, 0]]), label=3,
                    attributes={'track_id': 1}),
                Mask(np.array([[0, 0, 1, 0, 0]]), label=2,
                    attributes={'track_id': 2}),
                Mask(np.array([[1, 1, 0, 0, 0]]), label=3,
                    attributes={'track_id': 3}),
            ]),
            DatasetItem(id=2, subset='train', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[1, 0, 0, 0, 0]]), label=3,
                    attributes={'track_id': 2}),
            ]),
            DatasetItem(id=3, subset='val', image=np.ones((5, 1)), annotations=[
                Mask(np.array([[0, 1, 0, 0, 0]]), label=0,
                    attributes={'track_id': 1}),
            ]),
        ], categories=['a', 'b', 'c', 'd'])

        parsed = Dataset.import_from(DUMMY_DATASET_DIR, 'mots')
        compare_datasets(self, expected=target, actual=parsed)