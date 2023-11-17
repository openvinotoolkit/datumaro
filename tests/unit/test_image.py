# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from itertools import product
from unittest import TestCase

import numpy as np
import pytest

import datumaro.util.image as image_module

from ..requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir


class ImageOperationsTest(TestCase):
    def setUp(self):
        self.default_backend = image_module._IMAGE_BACKEND.get()

    def tearDown(self):
        image_module._IMAGE_BACKEND.set(self.default_backend)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_and_load_backends(self):
        backends = image_module._IMAGE_BACKENDS
        for save_backend, load_backend, c in product(backends, backends, [1, 3]):
            with TestDir() as test_dir:
                if c == 1:
                    src_image = np.random.randint(0, 255 + 1, (2, 4))
                else:
                    src_image = np.random.randint(0, 255 + 1, (2, 4, c))
                path = osp.join(test_dir, "img.png")  # lossless

                image_module._IMAGE_BACKEND.set(save_backend)
                image_module.save_image(path, src_image, jpeg_quality=100)

                image_module._IMAGE_BACKEND.set(load_backend)
                dst_image = image_module.load_image(path)

                self.assertTrue(
                    np.array_equal(src_image, dst_image),
                    "save: %s, load: %s" % (save_backend, load_backend),
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_encode_and_decode_backends(self):
        backends = image_module._IMAGE_BACKENDS
        for save_backend, load_backend, c in product(backends, backends, [1, 3]):
            if c == 1:
                src_image = np.random.randint(0, 255 + 1, (2, 4))
            else:
                src_image = np.random.randint(0, 255 + 1, (2, 4, c))

            image_module._IMAGE_BACKEND.set(save_backend)
            buffer = image_module.encode_image(src_image, ".png", jpeg_quality=100)  # lossless

            image_module._IMAGE_BACKEND.set(load_backend)
            dst_image = image_module.decode_image(buffer)

            self.assertTrue(
                np.array_equal(src_image, dst_image),
                "save: %s, load: %s" % (save_backend, load_backend),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_image_to_inexistent_dir_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            image_module.save_image("some/path.jpg", np.ones((5, 4, 3)), create_dir=False)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_image_can_create_dir(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, "some", "path.jpg")
            image_module.save_image(path, np.ones((5, 4, 3)), create_dir=True)
            self.assertTrue(osp.isfile(path))


class ImageDecodeTest:
    @pytest.fixture
    def fxt_img_four_channels(self) -> np.ndarray:
        return np.random.randint(low=0, high=256, size=(5, 4, 4), dtype=np.uint8)

    def test_decode_image_context(self, fxt_img_four_channels: np.ndarray):
        img_bytes = image_module.encode_image(fxt_img_four_channels, ".png")

        # 3 channels from ImageColorScale.COLOR
        with image_module.decode_image_context(image_module.ImageColorScale.COLOR):
            img_decoded = image_module.decode_image(img_bytes)
            assert img_decoded.shape[-1] == 3
            assert np.allclose(fxt_img_four_channels[:, :, :3], img_decoded)

        # 4 channels from ImageColorScale.UNCHANGED
        with image_module.decode_image_context(image_module.ImageColorScale.UNCHANGED):
            img_decoded = image_module.decode_image(img_bytes)
            assert img_decoded.shape[-1] == 4
            assert np.allclose(fxt_img_four_channels, img_decoded)
