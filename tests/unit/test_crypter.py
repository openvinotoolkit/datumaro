# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest
from datumaro.components.media import Image
from datumaro.components.crypter import Crypter
import numpy as np
import os.path as osp


@pytest.fixture(scope="module")
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


def test_load_and_save(fxt_image_file, fxt_encrypted_image_file, fxt_crypter):
    img = Image(path=fxt_image_file)
    encrypted_img = Image(path=fxt_encrypted_image_file, crypter=fxt_crypter)

    assert img == encrypted_img
