# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.media import Image
from datumaro.util.image import find_images


class SuperResolutionPath:
    HR_IMAGES_DIR = "HR"
    LR_IMAGES_DIR = "LR"
    UPSAMPLED_IMAGES_DIR = "upsampled"


class SuperResolutionExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__(subset=subset)

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        lr_image_dir = osp.join(path, SuperResolutionPath.LR_IMAGES_DIR)
        if osp.isdir(lr_image_dir):
            lr_images = {
                osp.splitext(osp.relpath(p, lr_image_dir))[0].replace("\\", "/"): p
                for p in find_images(lr_image_dir, recursive=True)
            }
        else:
            lr_images = {}

        hr_image_dir = osp.join(path, SuperResolutionPath.HR_IMAGES_DIR)
        if osp.isdir(hr_image_dir):
            hr_images = {
                osp.splitext(osp.relpath(p, hr_image_dir))[0].replace("\\", "/"): p
                for p in find_images(hr_image_dir, recursive=True)
            }
        else:
            hr_images = {}

        upsampled_image_dir = osp.join(path, SuperResolutionPath.UPSAMPLED_IMAGES_DIR)
        if osp.isdir(upsampled_image_dir):
            upsampled_images = {
                osp.splitext(osp.relpath(p, upsampled_image_dir))[0].replace("\\", "/"): p
                for p in find_images(upsampled_image_dir, recursive=True)
            }
        else:
            upsampled_images = {}

        for item_id, image in lr_images.items():
            attributes = {}
            hr_image = hr_images.get(item_id)
            if hr_image:
                attributes["hr"] = Image(path=hr_image)

            upsampled_image = upsampled_images.get(item_id)
            if upsampled_image:
                attributes["upsampled"] = Image(path=upsampled_image)

            items[item_id] = DatasetItem(
                id=item_id, subset=self._subset, media=Image(path=image), attributes=attributes
            )

        return items


class SuperResolutionImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(osp.join(SuperResolutionPath.HR_IMAGES_DIR, "**", "*"))
        context.require_file(osp.join(SuperResolutionPath.LR_IMAGES_DIR, "**", "*"))

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": "super_resolution"}]
