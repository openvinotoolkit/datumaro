from unittest import TestCase
import os
import os.path as osp

from datumaro.plugins.synthetic_data import ImageGenerator
from datumaro.util.test_utils import TestDir
import datumaro.util.image as image_module

from .requirements import Requirements, mark_requirement


class ImageGeneratorTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_image_can_create_dir(self):
        with TestDir() as test_dir:
            dataset_size = 2
            ImageGenerator(test_dir, dataset_size, shape=[224,256]).generate_dataset()
            image_files = os.listdir(test_dir)
            self.assertEqual(len(image_files), dataset_size)

            for filename in image_files:
                image = image_module.load_image(osp.join(test_dir, filename))
                H, W, C = image.shape

                self.assertEqual(H, 224)
                self.assertEqual(W, 256)
                self.assertEqual(C, 3)
