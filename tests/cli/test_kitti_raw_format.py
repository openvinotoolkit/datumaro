from unittest import TestCase
import os.path as osp

from datumaro.cli.__main__ import main
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (
    AnnotationType, Cuboid3d, DatasetItem, LabelCategories,
)
from datumaro.util.test_utils import TestDir, compare_datasets_3d

from ..requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
    'tests', 'assets', 'kitti_dataset', 'kitti_raw')

def run(test, *args, expected_code=0):
    test.assertEqual(expected_code, main(args), str(args))

class KittiRawIntegrationScenarios(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_to_kitti_raw(self):
        with TestDir() as test_dir:
            export_dir = osp.join(test_dir, 'export_dir')
            expected_label_cat = LabelCategories(attributes={'occluded'})
            expected_label_cat.add('bus')
            expected_label_cat.add('car')
            expected_dataset = Dataset.from_iterable([
                DatasetItem(id='0000000000',
                    annotations=[
                        Cuboid3d(position=[1, 2, 3],
                            scale=[7.95, -3.62, -1.03],
                            label=1, attributes={'occluded': False,
                                'track_id': 1}),

                        Cuboid3d(position=[1, 1, 0],
                            scale=[8.34, 23.01, -0.76],
                            label=0, attributes={'occluded': False,
                                'track_id': 2})
                    ],
                    point_cloud=osp.join(export_dir, 'ds0', 'pointcloud',
                        '0000000000.pcd'),
                    related_images=[osp.join(export_dir, 'ds0',
                        'related_images', '0000000000_pcd', '0000000000.png')
                    ],
                    attributes={'frame': 0, 'description': ''}
                ),

                DatasetItem(id='0000000001',
                    annotations=[
                        Cuboid3d(position=[0, 1, 0],
                            scale=[8.34, 23.01, -0.76],
                            rotation=[1, 1, 3],
                            label=0, attributes={'occluded': True,
                                'track_id': 2})
                    ],
                    point_cloud=osp.join(export_dir, 'ds0', 'pointcloud',
                        '0000000001.pcd'),
                    related_images=[osp.join(export_dir, 'ds0',
                        'related_images', '0000000001_pcd', '0000000001.png')
                    ],
                    attributes={'frame': 1, 'description': ''}
                ),

                DatasetItem(id='0000000002',
                    annotations=[
                        Cuboid3d(position=[1, 2, 3],
                            scale=[-9.41, 13.54, 0.24],
                            label=1, attributes={'occluded': False,
                                'track_id': 3})
                    ],
                    point_cloud=osp.join(export_dir, 'ds0', 'pointcloud',
                        '0000000002.pcd'),
                    related_images=[osp.join(export_dir, 'ds0',
                        'related_images', '0000000002_pcd', '0000000002.png')
                    ],
                    attributes={'frame': 2, 'description': ''}
                ),
            ], categories={AnnotationType.label: expected_label_cat})

            run(self, 'convert',
                '-if', 'kitti_raw', '-i', DUMMY_DATASET_DIR,
                '-f', 'sly_pointcloud', '-o', export_dir,
                '--', '--save-images')

            parsed_dataset = Dataset.import_from(export_dir,
                format='sly_pointcloud')
            compare_datasets_3d(self, expected_dataset, parsed_dataset,
                require_point_cloud=True)
