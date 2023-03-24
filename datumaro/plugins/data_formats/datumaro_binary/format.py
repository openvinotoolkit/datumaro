# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.errors import DatasetImportError

_SIGNATURE = "signature:datumaro_binary"


class DatumaroBinaryPath:
    IMAGES_DIR = "images"
    ANNOTATIONS_DIR = "annotations"
    PCD_DIR = "point_clouds"
    RELATED_IMAGES_DIR = "related_images"
    MASKS_DIR = "masks"

    ANNOTATION_EXT = ".datum"
    IMAGE_EXT = ".jpg"
    MASK_EXT = ".png"
    SIGNATURE = _SIGNATURE
    SIGNATURE_LEN = len(_SIGNATURE)

    SECRET_KEY_FILE = "secret_key.txt"

    MAX_BLOB_SIZE = 2**20  # 1 Mega bytes

    @classmethod
    def check_signature(cls, signature: str):
        if signature != cls.SIGNATURE:
            raise DatasetImportError(
                f"Input signature={signature} is not aligned with the ground truth signature={cls.SIGNATURE}"
            )
