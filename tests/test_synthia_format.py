
from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.util.test_utils import compare_datasets
import datumaro.plugins.synthia_format as Synthia

from .requirements import Requirements, mark_requirement

DUMMY_INST_SEGM_DATASET_DIR = osp.join(osp.dirname(__file__),
    'assets', 'synthia_dataset', 'synthia_dataset_1')

DUMMY_SEM_SEGM_DATASET_DIR = osp.join(osp.dirname(__file__),
    'assets', 'synthia_dataset', 'synthia_dataset_2')

DUMMY_DATASET_DIR_CUSTOM_LABELMAP = osp.join(osp.dirname(__file__),
    'assets', 'synthia_dataset', 'synthia_dataset_3')

class SynthiaImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_497)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_INST_SEGM_DATASET_DIR)
        self.assertEqual([Synthia.SynthiaImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_497)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='Stereo_Left/Omni_B/000000',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(np.array([[1, 1, 0, 0, 0]]), label=1),
                    Mask(np.array([[0, 0, 1, 1, 1]]), label=10),
                ],
            ),
            DatasetItem(id='Stereo_Left/Omni_B/000001',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(np.array([[1, 0, 0, 0, 0]]), label=8),
                    Mask(np.array([[0, 1, 1, 0, 0]]), label=11),
                    Mask(np.array([[0, 0, 0, 1, 1]]), label=3),
                ],
            ),
            DatasetItem(id='Stereo_Left/Omni_F/000000',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(np.array([[1, 1, 0, 0, 0]]), label=1),
                    Mask(np.array([[0, 0, 1, 1, 0]]), label=2),
                    Mask(np.array([[0, 0, 0, 0, 1]]), label=3),
                ],
            ),
            DatasetItem(id='Stereo_Left/Omni_F/000001',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(np.array([[1, 0, 0, 0, 0]]), label=1),
                    Mask(np.array([[0, 1, 0, 0, 0]]), label=2),
                    Mask(np.array([[0, 0, 1, 1, 0]]), label=15),
                    Mask(np.array([[0, 0, 0, 0, 1]]), label=3),
                ],
            )
        ], categories=Synthia.make_categories())

        dataset = Dataset.import_from(DUMMY_INST_SEGM_DATASET_DIR, 'synthia')

        compare_datasets(self, expected_dataset, dataset, require_images=True)


    @mark_requirement(Requirements.DATUM_497)
    def test_can_import_by_color_masks(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='Stereo_Left/Omni_F/000000',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(np.array([[1, 1, 0, 0, 0]]), label=1),
                    Mask(np.array([[0, 0, 1, 1, 0]]), label=2),
                    Mask(np.array([[0, 0, 0, 0, 1]]), label=3),
                ],
            ),
            DatasetItem(id='Stereo_Left/Omni_F/000001',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(np.array([[1, 0, 0, 0, 0]]), label=1),
                    Mask(np.array([[0, 1, 0, 0, 0]]), label=2),
                    Mask(np.array([[0, 0, 1, 1, 0]]), label=15),
                    Mask(np.array([[0, 0, 0, 0, 1]]), label=3),
                ],
            )
        ], categories=Synthia.make_categories())

        dataset = Dataset.import_from(DUMMY_SEM_SEGM_DATASET_DIR, 'synthia')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_497)
    def test_can_import_by_color_masks(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='Stereo_Left/Omni_F/000000',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(np.array([[1, 1, 1, 0, 0]]), label=1),
                    Mask(np.array([[0, 0, 0, 1, 1]]), label=4),
                ],
            ),
            DatasetItem(id='Stereo_Left/Omni_F/000001',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(np.array([[1, 1, 0, 0, 0]]), label=2),
                    Mask(np.array([[0, 0, 1, 1, 0]]), label=3),
                    Mask(np.array([[0, 0, 0, 0, 1]]), label=4),
                ],
            )
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                ['background', 'sky', 'building', 'person', 'road']),
            AnnotationType.mask: MaskCategories({0: (0, 0, 0), 1: (0, 0, 64),
                2: (0, 128, 128), 3: (128, 0, 64), 4: (0, 192, 128)})
        })

        dataset = Dataset.import_from(DUMMY_DATASET_DIR_CUSTOM_LABELMAP, 'synthia')

        compare_datasets(self, expected_dataset, dataset, require_images=True)
