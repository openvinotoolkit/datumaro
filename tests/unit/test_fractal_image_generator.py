import os
import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.plugins.synthetic_data import FractalImageGenerator
from datumaro.util.image import load_image
from datumaro.util.test_utils import TestDir

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path


class FractalImageGeneratorTest(TestCase):
    @mark_requirement(Requirements.DATUM_677)
    def test_save_image_can_create_dir(self):
        with TestDir() as test_dir:
            dataset_size = 2
            FractalImageGenerator(test_dir, dataset_size, shape=[22, 25]).generate_dataset()
            image_files = os.listdir(test_dir)
            self.assertEqual(len(image_files), dataset_size)

            for filename in image_files:
                image = load_image(osp.join(test_dir, filename))
                H, W, C = image.shape

                self.assertEqual(H, 22)
                self.assertEqual(W, 25)
                self.assertEqual(C, 3)

    @mark_requirement(Requirements.DATUM_677)
    def test_can_generate_image(self):
        ref_dir = get_test_asset_path("synthetic_dataset", "images")
        with TestDir() as test_dir:
            dataset_size = 3
            FractalImageGenerator(test_dir, dataset_size, shape=[24, 36]).generate_dataset()
            image_files = os.listdir(test_dir)
            self.assertEqual(len(image_files), dataset_size)

            for filename in image_files:
                actual = load_image(osp.join(test_dir, filename))
                expected = load_image(osp.join(ref_dir, filename))
                np.testing.assert_array_equal(actual, expected)
