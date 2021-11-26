from unittest import TestCase
import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check

import numpy as np

from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.cifar_format import CifarConverter, CifarImporter
from datumaro.util.meta_file_util import save_meta_file
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement


class CifarFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='image_2', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_3', subset='test',
                image=np.ones((32, 32, 3))
            ),
            DatasetItem(id='image_4', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            )
        ], categories=['label_0', 'label_1'])

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_without_saving_images(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train_1',
                annotations=[Label(0)]
            ),
            DatasetItem(id='b', subset='train_first',
                annotations=[Label(1)]
            ),
        ], categories=['x', 'y'])

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=False)
            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_different_image_size(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='image_1',
                image=np.ones((10, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_2',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            ),
        ], categories=['dog', 'cat'])

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id="кириллица с пробелом",
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
        ], categories=['label_0'])

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='q/1',
                image=Image(path='q/1.JPEG', data=np.zeros((32, 32, 3)))),
            DatasetItem(id='a/b/c/2',
                image=Image(path='a/b/c/2.bmp', data=np.zeros((32, 32, 3)))),
        ], categories=[])

        with TestDir() as test_dir:
            CifarConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_empty_image(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='a', annotations=[Label(0)]),
            DatasetItem(id='b')
        ], categories=['label_0'])

        with TestDir() as test_dir:
            CifarConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        expected = Dataset.from_iterable([
            DatasetItem(1, subset='a', image=np.ones((2, 1, 3)),
                annotations=[ Label(0) ]),
            DatasetItem(2, subset='a', image=np.ones((3, 2, 3)),
                annotations=[ Label(1) ]),

            DatasetItem(2, subset='b', image=np.ones((2, 2, 3)),
                annotations=[ Label(1) ]),
        ], categories=['a', 'b', 'c', 'd'])

        dataset = Dataset.from_iterable([
            DatasetItem(1, subset='a', image=np.ones((2, 1, 3)),
                annotations=[ Label(0) ]),

            DatasetItem(2, subset='b', image=np.ones((2, 2, 3)),
                annotations=[ Label(1) ]),

            DatasetItem(3, subset='c', image=np.ones((2, 3, 3)),
                annotations=[ Label(2) ]),
        ], categories=['a', 'b', 'c', 'd'])

        with TestDir() as path:
            dataset.export(path, 'cifar', save_images=True)

            dataset.put(DatasetItem(2, subset='a', image=np.ones((3, 2, 3)),
                annotations=[ Label(1) ]))
            dataset.remove(3, 'c')
            dataset.save(save_images=True)

            self.assertEqual({'a', 'b', 'batches.meta'},
                set(os.listdir(path)))
            compare_datasets(self, expected, Dataset.import_from(path, 'cifar'),
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_cifar100(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='image_2', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_3', subset='test',
                image=np.ones((32, 32, 3))
            ),
            DatasetItem(id='image_4', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            )
        ], categories=[
            ['class_0', 'superclass_0'],
            ['class_1', 'superclass_0']
        ])

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_cifar100_without_saving_images(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train_1',
                annotations=[Label(0)]
            ),
            DatasetItem(id='b', subset='train_1',
                annotations=[Label(1)]
            ),
        ], categories=[
            ['class_0', 'superclass_0'],
            ['class_1', 'superclass_0']
        ])

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=False)
            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_catch_pickle_exception(self):
        with TestDir() as test_dir:
            anno_file = osp.join(test_dir, 'test')
            with open(anno_file, 'wb') as file:
                pickle.dump(enumerate([1, 2, 3]), file)
            with self.assertRaisesRegex(pickle.UnpicklingError, "Global"):
                Dataset.import_from(test_dir, 'cifar')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_metafile(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='image_2', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_3', subset='test',
                image=np.ones((32, 32, 3))
            ),
            DatasetItem(id='image_4', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            )
        ], categories=['label_0', 'label_1'])

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=True,
                save_dataset_meta=True)
            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

DUMMY_10_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'cifar10_dataset')

DUMMY_100_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'cifar100_dataset')

class CifarImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_10(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='image_1', subset='data_batch_1',
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_2', subset='test_batch',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='image_3', subset='test_batch',
                image=np.ones((32, 32, 3)),
                annotations=[Label(3)]
            ),
            DatasetItem(id='image_4', subset='test_batch',
                image=np.ones((32, 32, 3)),
                annotations=[Label(2)]
            ),
            DatasetItem(id='image_5', subset='test_batch',
                image=np.array([[[1, 2, 3], [4, 5, 6]],
                                [[1, 2, 3], [4, 5, 6]]]),
                annotations=[Label(3)]
            )
        ], categories=['airplane', 'automobile', 'bird', 'cat'])

        dataset = Dataset.import_from(DUMMY_10_DATASET_DIR, 'cifar')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_10(self):
        detected_formats = Environment().detect_dataset(DUMMY_10_DATASET_DIR)
        self.assertIn(CifarImporter.NAME, detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_100(self):
        # Unless simple dataset merge can't overlap labels and add parent
        # information, the datasets must contain all the possible labels.
        # This should be normal on practice.
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='image_1', subset='train',
                image=np.ones((7, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_2', subset='train',
                image=np.ones((4, 5, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='image_3', subset='train',
                image=np.ones((4, 5, 3)),
                annotations=[Label(2)]
            ),

            DatasetItem(id='image_1', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_2', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='image_3', subset='test',
                image=np.array([[[1, 2, 3], [4, 5, 6]],
                                [[1, 2, 3], [4, 5, 6]]]),
                annotations=[Label(2)]
            )
        ], categories=[
            ['airplane', 'air_object'],
            ['automobile', 'ground_object'],
            ['bird', 'air_object'],
        ])

        dataset = Dataset.import_from(DUMMY_100_DATASET_DIR, 'cifar')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_100(self):
        detected_formats = Environment().detect_dataset(DUMMY_100_DATASET_DIR)
        self.assertIn(CifarImporter.NAME, detected_formats)
