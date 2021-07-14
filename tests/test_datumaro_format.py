from functools import partial
from unittest import TestCase
import os
import os.path as osp

import numpy as np

from datumaro.components.extractor import (
    AnnotationType, Bbox, Caption, Cuboid3d, DatasetItem, Label,
    LabelCategories, Mask, MaskCategories, Points, PointsCategories, Polygon,
    PolyLine,
)
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
    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            # generate initial dataset
            dataset = Dataset.from_iterable([
                DatasetItem(1, subset='a'),
                DatasetItem(2, subset='b'),
                DatasetItem(3, subset='c', image=np.ones((2, 2, 3))),
            ])
            dataset.export(path, 'datumaro', save_images=True)
            os.unlink(osp.join(path, 'annotations', 'a.json'))
            os.unlink(osp.join(path, 'annotations', 'b.json'))
            os.unlink(osp.join(path, 'annotations', 'c.json'))
            self.assertFalse(osp.isfile(osp.join(path, 'images', '2.jpg')))
            self.assertTrue(osp.isfile(osp.join(path, 'images', '3.jpg')))

            dataset.put(DatasetItem(2, subset='a', image=np.ones((3, 2, 3))))
            dataset.remove(3, 'c')
            dataset.save(save_images=True)

            self.assertTrue(osp.isfile(osp.join(path, 'annotations', 'a.json')))
            self.assertFalse(osp.isfile(osp.join(path, 'annotations', 'b.json')))
            self.assertTrue(osp.isfile(osp.join(path, 'annotations', 'c.json')))
            self.assertTrue(osp.isfile(osp.join(path, 'images', '2.jpg')))
            self.assertFalse(osp.isfile(osp.join(path, 'images', '3.jpg')))


    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_pointcloud(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='test', point_cloud='1.pcd',
                related_images= [Image(data=np.ones((5, 5, 3)), path='1/a.jpg')],
                annotations=[
                    Cuboid3d([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [3.0, 3.0, 1.0],
                        id=1, label=0, attributes={'occluded': True}, group=1
                    )
                ]),
        ], categories=['label'])

        with TestDir() as test_dir:
            target_dataset = Dataset.from_iterable([
                DatasetItem(id=1, subset='test',
                    point_cloud=osp.join(test_dir, 'point_clouds', '1.pcd'),
                    related_images= [Image(data=np.ones((5, 5, 3)),
                        path=osp.join(test_dir, 'related_images', '1/a.jpg'))],
                    annotations=[
                        Cuboid3d([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [3.0, 3.0, 1.0],
                            id=1, label=0, attributes={'occluded': True}, group=1
                        )
                    ]),
            ], categories=['label'])
            self._test_save_and_load(source_dataset,
                partial(DatumaroConverter.convert, save_images=True), test_dir,
                target_dataset, compare=None, dimension=Dimensions.dim_3d)
