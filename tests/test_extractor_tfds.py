from unittest.case import TestCase, skipIf

import numpy as np

from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.components.extractor_tfds import (
    TFDS_EXTRACTOR_AVAILABLE, make_tfds_extractor,
)
from datumaro.util.image import encode_image
from datumaro.util.test_utils import compare_datasets, mock_tfds_data

from tests.requirements import Requirements, mark_requirement

if TFDS_EXTRACTOR_AVAILABLE:
    import tensorflow_datasets as tfds

@skipIf(not TFDS_EXTRACTOR_AVAILABLE, reason="TFDS is not installed")
class TfdsExtractorTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_data_access(self):
        with mock_tfds_data():
            extractor = make_tfds_extractor('mnist')
            self.assertEqual(len(extractor), 1)

            train_subset = extractor.get_subset('train')
            compare_datasets(self, Dataset(extractor), Dataset(train_subset))

            self.assertRaises(KeyError, extractor.get_subset, 'test')

            subsets = extractor.subsets()
            self.assertEqual(len(subsets), 1)
            self.assertIn('train', subsets)
            compare_datasets(self,
                Dataset(extractor), Dataset(subsets['train']))

            self.assertIsNotNone(extractor.get('0'))
            self.assertIsNotNone(extractor.get('0', subset='train'))
            self.assertIsNone(extractor.get('x'))
            self.assertIsNone(extractor.get('0', subset='test'))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_extract_mnist(self):
        with mock_tfds_data():
            tfds_ds, tfds_info = tfds.load(
                'mnist', split='train', with_info=True)
            tfds_example = next(iter(tfds_ds))

            expected_dataset = Dataset.from_iterable([
                DatasetItem(id='0', subset='train',
                    image=tfds_example['image'].numpy().squeeze(axis=2),
                    annotations=[Label(tfds_example['label'].numpy())],
                ),
            ], categories=tfds_info.features['label'].names)

            extractor = make_tfds_extractor('mnist')
            actual_dataset = Dataset(extractor)

            compare_datasets(self, expected_dataset, actual_dataset,
                require_images=True)

    def _test_can_extract_cifar(self, name):
        with mock_tfds_data():
            tfds_ds, tfds_info = tfds.load(
                name, split='train', with_info=True)
            tfds_example = next(iter(tfds_ds))

            expected_dataset = Dataset.from_iterable([
                DatasetItem(
                    id=tfds_example['id'].numpy().decode('UTF-8'),
                    subset='train',
                    image=tfds_example['image'].numpy()[..., ::-1],
                    annotations=[Label(tfds_example['label'].numpy())],
                ),
            ], categories=tfds_info.features['label'].names)

            extractor = make_tfds_extractor(name)
            actual_dataset = Dataset(extractor)

            compare_datasets(self, expected_dataset, actual_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_extract_cifar10(self):
        self._test_can_extract_cifar('cifar10')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_extract_cifar100(self):
        self._test_can_extract_cifar('cifar100')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_extract_voc(self):
        # TFDS is unable to generate fake examples for object detection
        # datasets. See <https://github.com/tensorflow/datasets/issues/3633>.
        tfds_example = {
            'image/filename': 'test.png',
            'image': encode_image(np.ones((20, 10)), '.png'),
            'objects': {
                'bbox': [[0.1, 0.2, 0.3, 0.4]],
                'label': [5],
            }
        }

        with mock_tfds_data(example=tfds_example):
            tfds_info = tfds.builder('voc/2012').info

            expected_dataset = Dataset.from_iterable([
                DatasetItem(
                    id='test',
                    subset='train',
                    image=np.ones((20, 10)),
                    annotations=[
                        Bbox(2, 2, 2, 4, label=5),
                    ],
                ),
            ], categories=tfds_info.features['objects'].feature['label'].names)

            extractor = make_tfds_extractor('voc/2012')
            actual_dataset = Dataset(extractor)

            compare_datasets(self, expected_dataset, actual_dataset,
                require_images=True)
