# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import pytest

from datumaro.components.crypter import Crypter
from datumaro.components.dataset import Dataset
from datumaro.components.media import Image
from datumaro.errors import DatasetImportError
from datumaro.plugins.data_formats.datumaro_binary.format import DatumaroBinaryPath
from datumaro.util.test_utils import TestCaseHelper, TestDir, compare_datasets
from datumaro.util.test_utils import run_datum as run

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path

yolo_dir = get_test_asset_path("yolo_dataset")


@pytest.fixture
def export_dir():
    with TestDir() as export_dir:
        yield export_dir


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_yolo_to_dm_binary_encryption(test_dir: str, export_dir: str, helper_tc: TestCaseHelper):
    """
    1. Create project
    2. Import yolo format dataset as "src_yolo" name
    3. Export it to DatumaroBinary format with encryption
    4. Check the encryption
    5. Succeed to import the encrypted dataset with the true key
    6. Re-export it to the yolo format
    7. Test whether it is the same as the "src_yolo"
    """
    yolo_dir = get_test_asset_path("yolo_dataset")

    # 1. Create project
    run(helper_tc, "create", "-o", test_dir)

    # 2. Import yolo format dataset as "src_yolo" name
    run(helper_tc, "import", "-n", "src_yolo", "-p", test_dir, "-f", "yolo", yolo_dir)

    # 3. Export it to DatumaroBinary format with encryption
    run(
        helper_tc,
        "export",
        "-p",
        test_dir,
        "-o",
        osp.join(export_dir, "dm_binary"),
        "-f",
        "datumaro_binary",
        "--",
        "--save-media",
        "--encrypt",
    )

    # Remove src_yolo dataset from the project
    run(helper_tc, "remove", "-p", test_dir, "src_yolo")

    # Check whether the key exists
    key_path = osp.join(export_dir, "dm_binary", DatumaroBinaryPath.SECRET_KEY_FILE)
    assert osp.exists(key_path)

    # 4-0. Get secret key
    with open(key_path, "r") as fp:
        true_key = fp.read().encode()
        wrong_key = Crypter.gen_key(true_key)

    # 4-1. Wrong key cannot import the encrypted dataset.
    with pytest.raises(DatasetImportError):
        Dataset.import_from(
            osp.join(export_dir, "dm_binary"), format="datumaro_binary", encryption_key=wrong_key
        )

    # 4-2. You cannot open the encrypted image.
    with pytest.raises(Exception):
        for root, _, files in os.walk(export_dir):
            for file in files:
                fpath = osp.join(root, file)
                _, ext = osp.splitext(fpath)
                if ext == ".jpg":
                    assert Image(path=fpath).data is None

    # 5. Succeed to import the encrypted dataset with the true key
    run(
        helper_tc,
        "import",
        "-p",
        test_dir,
        "-f",
        "datumaro_binary",
        osp.join(export_dir, "dm_binary"),
        "--",
        "--encryption-key",
        true_key.decode(),
    )

    # 6. Re-export it to the yolo format
    run(
        helper_tc,
        "export",
        "-p",
        test_dir,
        "-o",
        osp.join(export_dir, "yolo"),
        "-f",
        "yolo",
        "--",
        "--save-media",
    )

    # 7. Test whether it is the same as the "src_yolo"
    expect = Dataset.import_from(yolo_dir, format="yolo")
    actual = Dataset.import_from(osp.join(export_dir, "yolo"), format="yolo")

    compare_datasets(helper_tc, expect, actual)
