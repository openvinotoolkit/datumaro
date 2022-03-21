import os
import os.path as osp
from unittest import TestCase

import datumaro.util.image as image_module
from datumaro.plugins.synthetic_data import ImageGenerator
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement


class ImageGeneratorTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_image_can_create_dir(self):
        with TestDir() as test_dir:
            dataset_size = 2
            ImageGenerator(test_dir, dataset_size, shape=[224, 256]).generate_dataset()
            image_files = os.listdir(test_dir)
            self.assertEqual(len(image_files), dataset_size)

            for filename in image_files:
                image = image_module.load_image(osp.join(test_dir, filename))
                H, W, C = image.shape

                self.assertEqual(H, 224)
                self.assertEqual(W, 256)
                self.assertEqual(C, 3)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_generate_image(self):
        ref_dir = osp.join(osp.dirname(__file__), "assets", "synthetic_dataset", "images")
        with TestDir() as test_dir:
            dataset_size = 1
            ImageGenerator(test_dir, dataset_size, shape=[24, 36]).generate_dataset()
            image_files = os.listdir(test_dir)
            self.assertEqual(len(image_files), dataset_size)

            for filename in image_files:
                image = image_module.load_image(osp.join(test_dir, filename))
                ref_image = image_module.load_image(osp.join(ref_dir, filename))
                self.assertListEqual(image.tolist(), ref_image.tolist())
