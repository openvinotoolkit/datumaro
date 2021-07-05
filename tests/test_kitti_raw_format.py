from functools import partial
from unittest import TestCase
import os
import os.path as osp

from datumaro.components.project import Dataset
from datumaro.components.extractor import (DatasetItem,
    AnnotationType, Cuboid3d, LabelCategories)
from datumaro.plugins.kitti_raw_format.extractor import KittiRawImporter
from datumaro.plugins.kitti_raw_format.converter import KittiRawConverter
from datumaro.util.test_utils import (Dimensions, TestDir, compare_datasets_3d,
    test_save_and_load)

from tests.requirements import mark_requirement, Requirements

DUMMY_DATASET_DIR = osp.join(osp.dirname(
    __file__), 'assets', 'kitti_dataset', 'kitti_raw')


class KittiRawImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_image(self):
        self.assertTrue(KittiRawImporter.detect(DUMMY_DATASET_DIR))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_pcd(self):
        pcd1 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
            'velodyne_points', 'data', '0000000000.pcd'))
        pcd2 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
            'velodyne_points', 'data', '0000000001.pcd'))
        pcd3 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
            'velodyne_points', 'data', '0000000002.pcd'))

        image1 = osp.abspath(
            osp.join(DUMMY_DATASET_DIR, 'IMAGE_00', 'data', '0000000000.png'))
        image2 = osp.abspath(
            osp.join(DUMMY_DATASET_DIR, 'IMAGE_00', 'data', '0000000001.png'))
        image3 = osp.abspath(
            osp.join(DUMMY_DATASET_DIR, 'IMAGE_00', 'data', '0000000003.png'))

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[1, 2, 3], scale=[7.95, -3.62, -1.03],
                        label=1, attributes={'occluded': False, 'track_id': 1}),

                    Cuboid3d(position=[1, 1, 0], scale=[8.34, 23.01, -0.76],
                        label=0, attributes={'occluded': False, 'track_id': 2})
                ],
                pcd=pcd1, related_images=[image1],
                attributes={'frame': 0}),

            DatasetItem(id='0000000001',
                annotations=[
                    Cuboid3d(position=[0, 1, 0], scale=[8.34, 23.01, -0.76],
                        rotation=[1, 1, 3],
                        label=0, attributes={'occluded': True, 'track_id': 2})
                ],
                pcd=pcd2, related_images=[image2],
                attributes={'frame': 1}),

            DatasetItem(id='0000000002',
                annotations=[
                    Cuboid3d(position=[1, 2, 3], scale=[-9.41, 13.54, 0.24],
                        label=1, attributes={'occluded': False, 'track_id': 3})
                ],
                pcd=pcd3, related_images=[image3],
                attributes={'frame': 2})

        ], categories=['bus', 'car'])

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'kitti_raw')

        compare_datasets_3d(self, expected_dataset, parsed_dataset,
            require_pcd=True)


class KittiRawConverterTest(TestCase):
    pcd1 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'velodyne_points', 'data', '0000000000.pcd'))
    pcd2 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'velodyne_points', 'data', '0000000001.pcd'))
    pcd3 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'velodyne_points', 'data', '0000000002.pcd'))

    image1 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'IMAGE_00', 'data', '0000000000.png'))
    image2 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'IMAGE_00', 'data', '0000000001.png'))
    image3 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'IMAGE_00', 'data', '0000000003.png'))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None, **kwargs):
        kwargs.setdefault('dimension', Dimensions.dim_3d)
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='kitti_raw', target_dataset=target_dataset,
            importer_args=importer_args, **kwargs)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[13.54, -9.41, 0.24], label=0,
                        attributes={'occluded': False, 'track_id': 1}),

                    Cuboid3d(position=[3.4, -2.11, 4.4], label=1,
                        attributes={'occluded': True, 'track_id': 2})
                ],
                pcd=self.pcd1, related_images=[self.image1],
                attributes={'frame': 0}
            ),

            DatasetItem(id='0000000001',
                annotations=[
                    Cuboid3d(position=[1.4, 2.1, 1.4], label=0,
                        attributes={'track_id': 2}),

                    Cuboid3d(position=[11.4, -0.1, 4.2], scale=[2, 1, 2],
                        label=0, attributes={'track_id': 3})
                ],
            ),

            DatasetItem(id='0000000002',
                annotations=[
                    Cuboid3d(position=[0.4, -1, 2.24], label=0,
                        attributes={'track_id': 3}),
                ],
                pcd=self.pcd3,
                attributes={'frame': 2}
            ),
        ], categories=['cat', 'dog'])

        target_label_cat = LabelCategories(attributes={'occluded'})
        target_label_cat.add('cat', 'dog')

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[13.54, -9.41, 0.24], label=0,
                        attributes={'occluded': False, 'track_id': 1}),

                    Cuboid3d(position=[3.4, -2.11, 4.4], label=1,
                        attributes={'occluded': True, 'track_id': 2})
                ],
                pcd=self.pcd1, related_images=[self.image1],
                attributes={'frame': 0}
            ),

            DatasetItem(id='0000000001',
                annotations=[
                    Cuboid3d(position=[1.4, 2.1, 1.4], label=0,
                        attributes={'occluded': False, 'track_id': 2}),

                    Cuboid3d(position=[11.4, -0.1, 4.2], scale=[2, 1, 2],
                        label=0, attributes={'occluded': False, 'track_id': 3})
                ],
                attributes={'frame': 1}
            ),

            DatasetItem(id='0000000002',
                annotations=[
                    Cuboid3d(position=[0.4, -1, 2.24], label=0,
                        attributes={'occluded': False, 'track_id': 3}),
                ],
                pcd=self.pcd3,
                attributes={'frame': 2}
            ),
        ], categories={AnnotationType.label: target_label_cat})

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(KittiRawConverter.convert, save_images=True),
                test_dir, target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preserve_frame_ids(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='abc', attributes={'frame': 40})
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                KittiRawConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='abc', attributes={'frame': 20})
        ])

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000', attributes={'frame': 0})
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(KittiRawConverter.convert, reindex=True),
                test_dir, target_dataset=expected_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_without_related_images(self):
        src_label_cat = LabelCategories(attributes={'occluded'})
        src_label_cat.add('car')
        src_label_cat.items[0].attributes.update(['a1', 'a2', 'empty'])

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[13.54, -9.41, 0.24],
                        attributes={'occluded': 0}, label=0)
                ],
                pcd=self.pcd1, related_images=[],
                attributes={'frame': 0}
                        ),
        ], categories={AnnotationType.label: src_label_cat})

        target_label_cat = LabelCategories(attributes={'occluded'})
        target_label_cat.add('ca')
        target_label_cat.items[0].attributes.update(['a1', 'a2', 'empty'])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[13.54, -9.41, 0.24],
                        attributes={'occluded': 0}, label=0)
                ],
                pcd=self.pcd1, related_images=[],
                attributes={'frame': 0}
                        ),
        ], categories={AnnotationType.label: target_label_cat})

        with TestDir() as test_dir:

            self._test_save_and_load(source_dataset,
                                     partial(KittiRawConverter.convert,
                                             save_images=True), test_dir,
                                     target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            # generate initial dataset

            dataset = Dataset.from_iterable([
                DatasetItem('0000000000.pcd', subset='tracklets'),
                DatasetItem('0000000001.pcd', subset='tracklets'),
                DatasetItem('0000000002.pcd',

                            pcd=self.pcd3,
                            related_images=[self.image2],
                            )
            ])

            dataset.export(path, 'velodyne_points', save_images=True)

            os.unlink(osp.join(path, 'tracklets.xml'))
            self.assertTrue(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00', 'data', '0000000002.png'))))
            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00', 'data', '0000000001.png'))))

            dataset.put(DatasetItem(2,

                                    pcd=self.pcd2,
                                    related_images=[self.image2],
                                    ))

            dataset.remove('0000000002.pcd', 'tracklets')
            related_image_path = {'related_paths': [
                'IMAGE_00', 'data'], 'image_names': ['0000000002.png']}
            dataset.save(save_images=True, **related_image_path)

            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'velodyne_points', 'data', '0000000000.pcd'))))
            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'velodyne_points', 'data', '0000000002.pcd'))))
            self.assertTrue(osp.isfile(osp.abspath(
                osp.join(path, 'velodyne_points', 'data', '0000000001.pcd'))))

            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00', 'data', '0000000000.png'))))
            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00', 'data', '0000000002.png'))))
            self.assertTrue(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00', 'data', '0000000001.png'))))
