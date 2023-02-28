# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import numpy as np
import pytest

from datumaro.components.crypter import NULL_CRYPTER, Crypter
from datumaro.components.media import Image


@pytest.fixture(scope="class")
def fxt_crypter():
    return Crypter(Crypter.gen_key())


@pytest.fixture()
def fxt_image_file(test_dir):
    img = Image(data=np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8))
    path = osp.join(test_dir, "test_crypter", "test.png")
    img.save(path)

    return path


@pytest.fixture()
def fxt_encrypted_image_file(test_dir, fxt_image_file, fxt_crypter):
    img = Image(path=fxt_image_file)
    path = osp.join(test_dir, "test_crypter", "test_encrypted.png")
    img.save(path, crypter=fxt_crypter)

    return path


class CrypterTest:
    def test_load_encrypted_image(self, fxt_image_file, fxt_encrypted_image_file, fxt_crypter):
        img = Image(path=fxt_image_file)
        encrypted_img = Image(path=fxt_encrypted_image_file, crypter=fxt_crypter)

        assert img == encrypted_img

    def _test_save_and_load(
        self, fxt_encrypted_image_file, fxt_crypter, test_dir, fname, new_crypter
    ):
        src_img = Image(path=fxt_encrypted_image_file, crypter=fxt_crypter)
        src_img_data = src_img.data  # Get data first until it is changed

        new_path = osp.join(test_dir, "test_crypter", fname)

        src_img.save(new_path, crypter=new_crypter)
        dst_img = Image(path=new_path, crypter=new_crypter)

        assert np.array_equal(src_img_data, dst_img.data)

    @pytest.mark.parametrize(
        "fname", ["new_encrypted.png", "test_encrypted.png"], ids=["new-path", "overwrite"]
    )
    def test_save_and_load_image_with_new_crypter(
        self, fxt_encrypted_image_file, fxt_crypter, test_dir, fname
    ):
        new_crypter = fxt_crypter
        while new_crypter == fxt_crypter:
            new_crypter = Crypter(Crypter.gen_key())

        self._test_save_and_load(
            fxt_encrypted_image_file, fxt_crypter, test_dir, fname, new_crypter
        )

    @pytest.mark.parametrize(
        "fname", ["new_encrypted.png", "test_encrypted.png"], ids=["new-path", "overwrite"]
    )
    def test_save_and_load_image_with_null_crypter(
        self, fxt_encrypted_image_file, fxt_crypter, test_dir, fname
    ):
        new_crypter = NULL_CRYPTER

        self._test_save_and_load(
            fxt_encrypted_image_file, fxt_crypter, test_dir, fname, new_crypter
        )
