import os.path as osp
from functools import partial
from unittest import TestCase

from datumaro.components.extractor import (DatasetItem,
                                           AnnotationType, Cuboid3D,
                                           LabelCategories,
                                           )
from datumaro.components.project import Dataset
from datumaro.plugins.pointcloud_format.converter import PointCloudConverter
from datumaro.plugins.pointcloud_format.extractor import PointCloudImporter
from datumaro.util.test_utils import (TestDir, compare_datasets_3d,
                                      test_save_and_load)
from .requirements import Requirements, mark_requirement

DUMMY_PCD_DATASET_DIR = osp.join(osp.dirname(
    __file__), 'assets', 'pointcloud_dataset')


class PointCloudImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_image(self):
        self.assertTrue(PointCloudImporter.detect(DUMMY_PCD_DATASET_DIR))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_pcd(self):
        pcd1 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                           r"ds0/pointcloud/frame.pcd"))
        pcd2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                           'assets', r"ds0/pointcloud/kitti_0000000001.pcd"))

        image1 = osp.abspath(osp.join(
            DUMMY_PCD_DATASET_DIR, r"ds0/related_images/kitti_0000000001_pcd/0000000000.png"))
        image2 = osp.abspath(osp.join(
            DUMMY_PCD_DATASET_DIR, r"ds0/related_images/frame_pcd/0000000002.png"))

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        subset="key_id_map",
                        annotations=[Cuboid3D(id=215,
                                              attributes={"label_id": 0},
                                              group=0,
                                              points=[320.59, 979.48, 1.03, 0.0,
                                                      0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label=0,
                                              z_order=0)],
                        path=[],
                        pcd=pcd1,
                        related_images=[{"name": "0000000002.png", "save_path": None,
                                         "path": image2}],
                        attributes={'frame': 0}),
            DatasetItem(id='frame_000001',
                        subset="key_id_map",
                        annotations=[Cuboid3D(id=216,
                                              attributes={"label_id": 1},
                                              group=0,
                                              points=[0.59, 14.41, -0.61, 0.0,
                                                      0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=1, z_order=0)],
                        path=[],
                        image=None,
                        pcd=pcd2,
                        related_images=[{"name": "0000000000.png", "save_path": None,
                                         "path": image1}],
                        attributes={'frame': 1})
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable([
                ['car'],
                ['bus'],
            ])
        })

        parsed_dataset = Dataset.import_from(
            DUMMY_PCD_DATASET_DIR, 'point_cloud')

        compare_datasets_3d(self, expected_dataset,
                            parsed_dataset, ignored_attrs=["label_id"])


class PointCloudConverterTest(TestCase):
    pcd1 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                       r"ds0/pointcloud/frame.pcd"))
    pcd2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                       r"ds0/pointcloud/kitti_0000000001.pcd"))

    image1 = osp.abspath(osp.join(
        DUMMY_PCD_DATASET_DIR, r"ds0/related_images/kitti_0000000001_pcd/0000000000.png"))
    image2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                         r"ds0/related_images/frame_pcd/0000000002.png"))

    dimension = {"dimension": "3d"}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def _test_save_and_load(self, source_dataset, converter, test_dir,
                            target_dataset=None, importer_args=None, **kwargs):
        return test_save_and_load(self, source_dataset, converter, test_dir,
                                  importer='point_cloud',
                                  target_dataset=target_dataset, importer_args=importer_args, **kwargs)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        src_label_cat = LabelCategories(attributes={'occluded'})
        src_label_cat.add('car')
        src_label_cat.add('bus')

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[
                            Cuboid3D(id=206,
                                     attributes={"occluded": 0, "label_id": 0},
                                     group=0,
                                     points=[320.86, 979.18, 1.04, 0.0,
                                             0.0,
                                             0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     label=0,
                                     z_order=0),
                            Cuboid3D(id=207,
                                     attributes={"occluded": 0, "label_id": 1},
                                     group=0,
                                     points=[318.19, 974.65, 1.29, 0.0, 0.0,
                                             0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     label=1,
                                     z_order=0)],
                        subset='key_id_map', path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[], attributes={'frame': 0, "name": "Anil", "createdAt": "", "updatedAt": "",
                                                       "labels": [{"label_id": 0, "name": "car", "color": "#fa3253"},
                                                                  {"label_id": 1, "name": "bus",
                                                                   "color": "#83e070"}]}),
            DatasetItem(id='frame_000001',
                        annotations=[Cuboid3D(id=208,
                                              attributes={
                                                  "occluded": 0, "label_id": 1},
                                              group=0,
                                              points=[23.04, 8.75, -0.78, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=1,
                                              z_order=0)],
                        subset='key_id_map', path=[],
                        image=None,
                        pcd=self.pcd2,
                        related_images=[{"name": "000000000.png", "save_path": None,
                                         "path": self.image1}],
                        attributes={'frame': 1})
        ], categories={AnnotationType.label: src_label_cat})

        target_label_cat = LabelCategories(attributes={'occluded'})
        target_label_cat.add("car")
        target_label_cat.add("bus")

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid3D(id=206,
                                              attributes={
                                                  "occluded": 0, "label_id": 0},
                                              group=0,
                                              points=[320.86, 979.18, 1.04, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0,
                                              z_order=0),
                                     Cuboid3D(id=207,
                                              attributes={
                                                  "occluded": 0, "label_id": 1},
                                              group=0,
                                              points=[318.19, 974.65, 1.29, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=1,
                                              z_order=0)],
                        subset='key_id_map', path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[], attributes={'frame': 0}),
            DatasetItem(id='frame_000001',
                        annotations=[Cuboid3D(id=208,
                                              attributes={
                                                  "occluded": 0, "label_id": 1},
                                              group=0,
                                              points=[23.04, 8.75, -0.78, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=1,
                                              z_order=0)],
                        subset='key_id_map', path=[],
                        image=None,
                        pcd=self.pcd2,
                        related_images=[{"name": "000000000.png", "save_path": None,
                                         "path": self.image1}],
                        attributes={'frame': 1})
        ], categories={AnnotationType.label: target_label_cat})

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                                     partial(PointCloudConverter.convert,
                                             save_images=True), test_dir,
                                     target_dataset=target_dataset, ignored_attrs=["label_id", "occluded"], **self.dimension)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preserve_frame_ids(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000020',
                        annotations=[Cuboid3D(id=206,
                                              attributes={
                                                  "occluded": 0, "label_id": 0},
                                              group=0,
                                              points=[320.86, 979.18, 1.04, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0,
                                              z_order=0),
                                     Cuboid3D(id=207,
                                              attributes={
                                                  "occluded": 0, "label_id": 1},
                                              group=0,
                                              points=[318.19, 974.65, 1.29, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=1,
                                              z_order=0)],
                        subset='key_id_map', path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[], attributes={'frame': 20, "name": "Anil", "createdAt": "", "updatedAt": "",
                                                       "labels": [{"label_id": 0, "name": "car", "color": "#fa3253"},
                                                                  {"label_id": 1, "name": "bus",
                                                                   "color": "#83e070"}]}),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(['car', 'bus'])
        })

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                                     PointCloudConverter.convert, test_dir,
                                     ignored_attrs=["label_id", "occluded", "name", "createdAt", "updatedAt",
                                                    "labels"],
                                     **self.dimension)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id="frame.pcd",

                        annotations=[Cuboid3D(id=206,
                                              attributes={
                                                  "occluded": 0, "label_id": 0},
                                              group=0,
                                              points=[320.86, 979.18, 1.04, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0,
                                              z_order=0),
                                     ],

                        path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[],
                        attributes={'frame': 20, "name": "user1", "createdAt": "2021-05-22 19:36:22.199768",
                                    "updatedAt": "2021-05-23 19:36:22.199768",
                                    "labels": [{"label_id": 0, "name": "car", "color": "#fa3253"},
                                               {"label_id": 1, "name": "bus",
                                                "color": "#83e070"}]}),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(['car', 'bus'])
        })

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id="frame_000000",
                        subset="key_id_map",
                        annotations=[Cuboid3D(id=206,
                                              attributes={
                                                  "occluded": 0, "label_id": 0},
                                              group=0,
                                              points=[320.86, 979.18, 1.04, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0,
                                              z_order=0),
                                     ],
                        path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[], attributes={'frame': 0}),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(['car', 'bus'])
        })

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                                     partial(PointCloudConverter.convert,
                                             reindex=True), test_dir,
                                     target_dataset=expected_dataset,
                                     ignored_attrs=["label_id", "occluded", "name", "createdAt", "updatedAt",
                                                    "labels"],
                                     **self.dimension)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            # generate initial dataset

            dataset = Dataset.from_iterable([
                DatasetItem(DatasetItem(id='frame.pcd',
                                        annotations=[Cuboid3D(id=215,
                                                              attributes={
                                                                  "label_id": 0},
                                                              group=0,
                                                              points=[320.59, 979.48,
                                                                      1.03, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label=0,
                                                              z_order=0)],
                                        subset='key_id_map', path=[],
                                        pcd=self.pcd1,
                                        related_images=[{"name": "0000000002.png", "save_path": None,
                                                         "path": self.image2}],
                                        attributes={'frame': 0, "name": "user1",
                                                    "createdAt": "2021-05-22 19:36:22.199768",
                                                    "updatedAt": "2021-05-23 19:36:22.199768",
                                                    "labels": [{"label_id": 0, "name": "car", "color": "#fa3253"},
                                                               {"label_id": 1, "name": "bus",
                                                                "color": "#83e070"}]})
                            )],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(
                        ['car', 'bus'])
            }
            )

            dataset.export(path, 'point_cloud', save_images=True)

            dataset.put(DatasetItem(id='kitti_0000000001.pcd',
                                    annotations=[Cuboid3D(id=216,
                                                          attributes={
                                                              "label_id": 1},
                                                          group=0, points=[0.59, 14.41,
                                                                           -0.61, 0.0, 0.0, 0.0, 1.0, 1.0,
                                                                           1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                          label=1, z_order=0)],
                                    subset='key_id_map', path=[],
                                    image=None,
                                    pcd=self.pcd2,
                                    related_images=[{"name": "0000000000.png", "save_path": None,
                                                     "path": self.image1}],
                                    attributes={'frame': 1, "name": "user1", "createdAt": "2021-05-22 19:36:22.199768",
                                                "updatedAt": "2021-05-23 19:36:22.199768",
                                                "labels": [{"label_id": 0, "name": "car", "color": "#fa3253"},
                                                           {"label_id": 1, "name": "bus",
                                                            "color": "#83e070"}],
                                                })
                        )

            dataset.remove("frame.pcd", "key_id_map")
            related_image_path = {'related_paths': [
                r"ds0/related_images/frame_pcd"], "image_names": ["0000000002.png"]}
            dataset.save(save_images=True, **related_image_path)

            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'ds0/pointcloud', 'frame.pcd'))))
            self.assertTrue(osp.isfile(osp.abspath(
                osp.join(path, 'ds0/pointcloud', 'kitti_0000000001.pcd'))))

            self.assertTrue(
                osp.isfile(osp.abspath(osp.join(path, r'ds0/related_images/kitti_0000000001_pcd', '0000000000.png'))))
            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, r'ds0/related_images/frame_pcd', '0000000002.png'))))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_without_related_images(self):
        src_label_cat = LabelCategories(attributes={'occluded'})
        src_label_cat.add('car')
        src_label_cat.add('bus')
        src_label_cat.items[0].attributes.update(['a1'])
        src_label_cat.items[1].attributes.update(['a4', 'a3', 'a5'])

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid3D(id=206,
                                              attributes={'a1': 'hello', 'a1__values': 'type\nhello\nmello',
                                                          "occluded": 0, "label_id": 0},
                                              group=0,
                                              points=[320.86, 979.18, 1.04, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0,
                                              z_order=0),
                                     Cuboid3D(id=207,
                                              attributes={"a4": "rare", "a3": False, 'a5':5.6,
                                                  "occluded": 0, "label_id": 1},
                                              group=0,
                                              points=[318.19, 974.65, 1.29, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=1,
                                              z_order=0)],
                        subset='key_id_map', path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[], attributes={'frame': 0, "name": "Anil", "createdAt": "", "updatedAt": "",
                                                       "labels": [{"label_id": 0, "name": "car", "color": "#fa3253"},
                                                                  {"label_id": 1, "name": "bus",
                                                                   "color": "#83e070"}]}),
            DatasetItem(id='frame_000001',
                        annotations=[Cuboid3D(id=208,
                                              attributes={'a1': 'hello', 'a1__values': 'type\nhello\nmello',
                                                  "occluded": 0, "label_id": 1},
                                              group=0,
                                              points=[23.04, 8.75, -0.78, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0,
                                              z_order=0)],
                        subset='key_id_map', path=[],
                        image=None,
                        pcd=self.pcd2,
                        related_images=[],
                        attributes={'frame': 1})
        ], categories={AnnotationType.label: src_label_cat})

        target_label_cat = LabelCategories(attributes={'occluded'})
        target_label_cat.add("car")
        target_label_cat.add("bus")
        target_label_cat.items[0].attributes.update(['a1'])
        target_label_cat.items[1].attributes.update(['a4', 'a3', 'a5'])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid3D(id=206,
                                              attributes={'a1': 'hello', 'a1__values': 'type\nhello\nmello',
                                                  "occluded": 0, "label_id": 0},
                                              group=0,
                                              points=[320.86, 979.18, 1.04, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0,
                                              z_order=0),
                                     Cuboid3D(id=207,
                                              attributes={"a4": "rare", "a3": False, "a5": 5.6,
                                                  "occluded": 0, "label_id": 1},
                                              group=0,
                                              points=[318.19, 974.65, 1.29, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=1,
                                              z_order=0)],
                        subset='key_id_map', path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[], attributes={'frame': 0}),
            DatasetItem(id='frame_000001',
                        annotations=[Cuboid3D(id=208,
                                              attributes={'a1': 'hello', 'a1__values': 'type\nhello\nmello',
                                                  "occluded": 0, "label_id": 1},
                                              group=0,
                                              points=[23.04, 8.75, -0.78, 0.0, 0.0,
                                                      0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0,
                                              z_order=0)],
                        subset='key_id_map', path=[],
                        image=None,
                        pcd=self.pcd2,
                        related_images=[],
                        attributes={'frame': 1})
        ], categories={AnnotationType.label: target_label_cat})

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                                     partial(PointCloudConverter.convert,
                                             save_images=True), test_dir,
                                     target_dataset=target_dataset, ignored_attrs=["a1__values", "label_id", "occluded"], **self.dimension)
