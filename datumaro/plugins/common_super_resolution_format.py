# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.annotation import SuperResolutionAnnotation
from datumaro.components.extractor import DatasetItem, SubsetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.util.image import find_images


class CommonSuperResolutionPath:
    HR_IMAGES_DIR = "HR"
    LR_IMAGES_DIR = "LR"
    UPSAMPLED_IMAGES_DIR = "upsampled"


class CommonSuperResolutionExtractor(SubsetBase):
    def __init__(self, path, subset=None):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__(subset=subset)

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        upsampled_image_dir = osp.join(path, CommonSuperResolutionPath.UPSAMPLED_IMAGES_DIR)
        if osp.isdir(upsampled_image_dir):
            upsampled_images = {
                osp.splitext(osp.relpath(p, upsampled_image_dir))[0].replace("\\", "/"): p
                for p in find_images(upsampled_image_dir, recursive=True)
            }
        else:
            upsampled_images = {}

        lr_image_dir = osp.join(path, CommonSuperResolutionPath.LR_IMAGES_DIR)
        for lr_image in find_images(lr_image_dir, recursive=True):
            item_id = osp.splitext(osp.relpath(lr_image, lr_image_dir))[0].replace("\\", "/")

            attributes = {}
            upsampled_image = upsampled_images.get(item_id)
            if upsampled_image:
                attributes["upsampled"] = Image(path=upsampled_image)

            items[item_id] = DatasetItem(
                id=item_id, subset=self._subset, media=Image(path=lr_image), attributes=attributes
            )

        hr_image_dir = osp.join(path, CommonSuperResolutionPath.HR_IMAGES_DIR)
        for hr_image in find_images(hr_image_dir, recursive=True):
            item_id = osp.splitext(osp.relpath(hr_image, hr_image_dir))[0].replace("\\", "/")
            if item_id not in items:
                attributes = {}
                upsampled_image = upsampled_images.get(item_id)
                if upsampled_image:
                    attributes["upsampled"] = Image(path=upsampled_image)

                items[item_id] = DatasetItem(id=item_id, subset=self._subset, attributes=attributes)

            items[item_id].annotations = [SuperResolutionAnnotation(Image(path=hr_image))]

        return items


class CommonSuperResolutionImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(osp.join(CommonSuperResolutionPath.HR_IMAGES_DIR, "**", "*"))
        context.require_file(osp.join(CommonSuperResolutionPath.LR_IMAGES_DIR, "**", "*"))

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": "common_super_resolution"}]
