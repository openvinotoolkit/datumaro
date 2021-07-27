from unittest import TestCase
import os.path as osp

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (
    AnnotationType, Cuboid3d, DatasetItem, LabelCategories,
)
from datumaro.util.test_utils import TestDir, compare_datasets_3d
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
    'tests', 'assets', 'sly_pointcloud_dataset')

class SlyPointCloudIntegrationScenarios(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_to_kitti_raw(self):
        with TestDir() as test_dir:
            export_dir = osp.join(test_dir, 'export_dir')
            expected_label_cat = LabelCategories(attributes={'occluded'})
            expected_label_cat.add('bus', attributes={'tag1', 'tag3'})
            expected_label_cat.add('car', attributes={'tag1', 'tag3'})
            expected_dataset = Dataset.from_iterable([
                DatasetItem(id='frame1',
                    annotations=[
                        Cuboid3d(label=1,
                            position=[0.47, 0.23, 0.79],
                            scale=[0.01, 0.01, 0.01],
                            attributes={'track_id': 2,
                                'tag1': 'fd', 'tag3': '4s', 'occluded': False}),

                        Cuboid3d(label=1,
                            position=[0.36, 0.64, 0.93],
                            scale=[0.01, 0.01, 0.01],
                            attributes={'track_id': 3,
                                'tag1': 'v12', 'tag3': '', 'occluded': False}),
                    ],
                    point_cloud=osp.join(export_dir, 'velodyne_points', 'data',
                        'frame1.pcd'),
                    related_images=[osp.join(export_dir, 'image_00', 'data',
                        'frame1.png')
                    ],
                    attributes={'frame': 0}
                ),

                DatasetItem(id='frame2',
                    annotations=[
                        Cuboid3d(label=0,
                            position=[0.59, 14.41, -0.61],
                            attributes={'track_id': 1,
                                'tag1': '', 'tag3': '', 'occluded': False})
                    ],
                    point_cloud=osp.join(export_dir, 'velodyne_points', 'data',
                        'frame2.pcd'),
                    related_images=[osp.join(export_dir, 'image_00', 'data',
                        'frame2.png')
                    ],
                    attributes={'frame': 1}
                ),
            ], categories={AnnotationType.label: expected_label_cat})

            run(self, 'convert',
                '-if', 'sly_pointcloud', '-i', DUMMY_DATASET_DIR,
                '-f', 'kitti_raw', '-o', export_dir,
                '--', '--save-images', '--allow-attrs')

            parsed_dataset = Dataset.import_from(export_dir, format='kitti_raw')
            compare_datasets_3d(self, expected_dataset, parsed_dataset,
                require_point_cloud=True)
