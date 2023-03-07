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

    ANNOTATION_EXT = ".datumaro"
    IMAGE_EXT = ".jpg"
    MASK_EXT = ".png"
    SIGNATURE = _SIGNATURE
    SIGNATURE_LEN = len(_SIGNATURE)

    @classmethod
    def check_signature(cls, signature: str):
        if signature != cls.SIGNATURE:
            raise DatasetImportError(
                f"Input signature={signature} is not aligned with the ground truth signature={cls.SIGNATURE}"
            )
