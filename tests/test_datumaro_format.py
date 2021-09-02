from functools import partial
from unittest import TestCase
import os
import os.path as osp

import numpy as np

from datumaro.components.annotation import (
    AnnotationType, Bbox, Caption, Cuboid3d, Label, LabelCategories, Mask,
    MaskCategories, Points, PointsCategories, Polygon, PolyLine,
)
from datumaro.components.extractor import DatasetItem
from datumaro.components.project import Dataset
from datumaro.plugins.datumaro_format.converter import DatumaroConverter
from datumaro.plugins.datumaro_format.extractor import DatumaroImporter
from datumaro.util.image import Image
from datumaro.util.mask_tools import generate_colormap
from datumaro.util.test_utils import (
    Dimensions, TestDir, compare_datasets_strict, test_save_and_load,
)

from .requirements import Requirements, mark_requirement


class DatumaroConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None,
            compare=compare_datasets_strict, **kwargs):
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='datumaro',
            target_dataset=target_dataset, importer_args=importer_args,
            compare=compare, **kwargs)

    @property
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset(self):
        label_categories = LabelCategories(attributes={'a', 'b', 'score'})
        for i in range(5):
            label_categories.add('cat' + str(i), attributes={'x', 'y'})

        mask_categories = MaskCategories(
            generate_colormap(len(label_categories.items)))

        points_categories = PointsCategories()
        for index, _ in enumerate(label_categories.items):
            points_categories.add(index, ['cat1', 'cat2'], joints=[[0, 1]])

        return Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Caption('hello', id=1),
                    Caption('world', id=2, group=5),
                    Label(2, id=3, attributes={
                        'x': 1,
                        'y': '2',
                    }),
                    Bbox(1, 2, 3, 4, label=4, id=4, z_order=1, attributes={
                        'score': 1.0,
                    }),
                    Bbox(5, 6, 7, 8, id=5, group=5, attributes={
                        'a': 1.5,
                        'b': 'text',
                    }),
                    Points([1, 2, 2, 0, 1, 1], label=0, id=5, z_order=4,
                        attributes={ 'x': 1, 'y': '2', }),
                    Mask(label=3, id=5, z_order=2, image=np.ones((2, 3)),
                        attributes={ 'x': 1, 'y': '2', }),
                ]),
            DatasetItem(id=21, subset='train',
                annotations=[
                    Caption('test'),
                    Label(2),
                    Bbox(1, 2, 3, 4, label=5, id=42, group=42)
                ]),

            DatasetItem(id=2, subset='val',
                annotations=[
                    PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11, z_order=1),
                    Polygon([1, 2, 3, 4, 5, 6, 7, 8], id=12, z_order=4),
                ]),

            DatasetItem(id=1, subset='test',
                annotations=[
                    Cuboid3d([1.0, 2.0, 3.0], [2.0, 2.0, 4.0], [1.0, 3.0, 4.0],
                        id=6, label=0, attributes={'occluded': True}, group=6
                    )
                ]),

            DatasetItem(id=42, subset='test',
                attributes={'a1': 5, 'a2': '42'}),

            DatasetItem(id=42),
            DatasetItem(id=43, image=Image(path='1/b/c.qq', size=(2, 4))),
        ], categories={
            AnnotationType.label: label_categories,
            AnnotationType.mask: mask_categories,
            AnnotationType.points: points_categories,
        })

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        with TestDir() as test_dir:
            self._test_save_and_load(self.test_dataset,
                partial(DatumaroConverter.convert, save_images=True), test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        with TestDir() as test_dir:
            DatumaroConverter.convert(self.test_dataset, save_dir=test_dir)

            self.assertTrue(DatumaroImporter.detect(test_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_relative_paths(self):
        test_dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((4, 2, 3))),
            DatasetItem(id='subdir1/1', image=np.ones((2, 6, 3))),
            DatasetItem(id='subdir2/1', image=np.ones((5, 4, 3))),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(test_dataset,
                partial(DatumaroConverter.convert, save_images=True), test_dir)


    @mark_requirement(Requirements.DATUM_231)
    def test_can_save_dataset_with_cjk_categories(self):
        expected = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Bbox(0, 1, 2, 2,
                        label=0, group=1, id=1,
                        attributes={ 'is_crowd': False }),
                ], attributes={'id': 1}),
            DatasetItem(id=2, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Bbox(1, 0, 2, 2, label=1, group=2, id=2,
                        attributes={ 'is_crowd': False }),
                ], attributes={'id': 2}),

            DatasetItem(id=3, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Bbox(0, 1, 2, 2, label=2, group=3, id=3,
                        attributes={ 'is_crowd': False }),
                ], attributes={'id': 3}),
            ],
            categories=[
                "고양이", "ネコ", "猫"
            ]
        )

        with TestDir() as test_dir:
            self._test_save_and_load(expected,
                partial(DatumaroConverter.convert, save_images=True), test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        test_dataset = Dataset.from_iterable([
            DatasetItem(id='кириллица с пробелом', image=np.ones((4, 2, 3))),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(test_dataset,
                partial(DatumaroConverter.convert, save_images=True),
                test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable([
            DatasetItem(id='q/1', image=Image(path='q/1.JPEG',
                data=np.zeros((4, 3, 3))), attributes={'frame': 1}),
            DatasetItem(id='a/b/c/2', image=Image(path='a/b/c/2.bmp',
                data=np.zeros((3, 4, 3))), attributes={'frame': 2}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected,
                partial(DatumaroConverter.convert, save_images=True),
                test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_direct_changes(self):
        expected = Dataset.from_iterable([
            DatasetItem(1, subset='a'),
            DatasetItem(2, subset='a', image=np.ones((3, 2, 3))),

            DatasetItem(2, subset='b'),
        ])

        with TestDir() as path:
            # generate initial dataset
            dataset = Dataset.from_iterable([
                # modified subset
                DatasetItem(1, subset='a'),

                # unmodified subset
                DatasetItem(2, subset='b'),

                # removed subset
                DatasetItem(3, subset='c', image=np.ones((2, 2, 3))),
            ])
            dataset.save(path, save_images=True)

            dataset.put(DatasetItem(2, subset='a', image=np.ones((3, 2, 3))))
            dataset.remove(3, 'c')
            dataset.save(save_images=True)

            self.assertEqual({'a.json', 'b.json'},
                set(os.listdir(osp.join(path, 'annotations'))))
            self.assertEqual({'2.jpg'},
                set(os.listdir(osp.join(path, 'images', 'a'))))
            compare_datasets_strict(self, expected, Dataset.load(path))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_transforms(self):
        with TestDir() as path:
            expected = Dataset.from_iterable([
                DatasetItem(2, subset='test'),
                DatasetItem(3, subset='train', image=np.ones((2, 2, 3))),
                DatasetItem(4, subset='train', image=np.ones((2, 3, 3))),
                DatasetItem(5, subset='test',
                    point_cloud=osp.join(path, 'point_clouds', 'test', '5.pcd'),
                    related_images=[
                        Image(data=np.ones((3, 4, 3)),
                            path=osp.join(path, 'test', '5', 'image_0.jpg')),
                        osp.join(path, 'test', '5', 'a', '5.png'),
                    ]
                ),
            ])
            dataset = Dataset.from_iterable([
                DatasetItem(1, subset='a'),
                DatasetItem(2, subset='b'),
                DatasetItem(3, subset='c', image=np.ones((2, 2, 3))),
                DatasetItem(4, subset='d', image=np.ones((2, 3, 3))),
                DatasetItem(5, subset='e', point_cloud='5.pcd',
                    related_images=[
                        np.ones((3, 4, 3)),
                        'a/5.png',
                    ]
                ),
            ])

            dataset.save(path, save_images=True)

            dataset.filter('/item[id >= 2]')
            dataset.transform('random_split', (('train', 0.5), ('test', 0.5)),
                seed=42)
            dataset.save(save_images=True)

            self.assertEqual(
                {'images', 'annotations', 'point_clouds', 'related_images'},
                set(os.listdir(path)))
            self.assertEqual({'train.json', 'test.json'},
                set(os.listdir(osp.join(path, 'annotations'))))
            self.assertEqual({'3.jpg', '4.jpg'},
                set(os.listdir(osp.join(path, 'images', 'train'))))
            self.assertEqual({'train', 'c', 'd'},
                set(os.listdir(osp.join(path, 'images'))))
            self.assertEqual(set(),
                set(os.listdir(osp.join(path, 'images', 'c'))))
            self.assertEqual(set(),
                set(os.listdir(osp.join(path, 'images', 'd'))))
            self.assertEqual({'image_0.jpg'},
                set(os.listdir(osp.join(path, 'related_images', 'test', '5'))))
            compare_datasets_strict(self, expected, Dataset.load(path))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_pointcloud(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='test', point_cloud='1.pcd',
                related_images= [
                    Image(data=np.ones((5, 5, 3)), path='1/a.jpg'),
                    Image(data=np.ones((5, 4, 3)), path='1/b.jpg'),
                    Image(size=(5, 3), path='1/c.jpg'),
                    '1/d.jpg',
                ],
                annotations=[
                    Cuboid3d([2, 2, 2], [1, 1, 1], [3, 3, 1],
                        id=1, group=1, label=0, attributes={'x': True}
                    )
                ]),
        ], categories=['label'])

        with TestDir() as test_dir:
            target_dataset = Dataset.from_iterable([
                DatasetItem(id=1, subset='test',
                    point_cloud=osp.join(test_dir, 'point_clouds',
                        'test', '1.pcd'),
                    related_images= [
                        Image(data=np.ones((5, 5, 3)), path=osp.join(
                            test_dir, 'related_images', 'test',
                            '1', 'image_0.jpg')),
                        Image(data=np.ones((5, 4, 3)), path=osp.join(
                            test_dir, 'related_images', 'test',
                            '1', 'image_1.jpg')),
                        Image(size=(5, 3), path=osp.join(
                            test_dir, 'related_images', 'test',
                            '1', 'image_2.jpg')),
                        osp.join(test_dir, 'related_images', 'test',
                            '1', 'image_3.jpg'),
                    ],
                    annotations=[
                        Cuboid3d([2, 2, 2], [1, 1, 1], [3, 3, 1],
                            id=1, group=1, label=0, attributes={'x': True}
                        )
                    ]),
            ], categories=['label'])
            self._test_save_and_load(source_dataset,
                partial(DatumaroConverter.convert, save_images=True), test_dir,
                target_dataset, compare=None, dimension=Dimensions.dim_3d)
