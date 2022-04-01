# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np

from datumaro.components.annotation import AnnotationType, Mask
from datumaro.components.dataset import Dataset
from datumaro.components.media import Image
from datumaro.plugins.sampler.random_sampler import RandomSampler


def dataset_mangling(dataset, count=-1, image_size=(3, 1, 3)):
    if count > 0:
        dataset = RandomSampler(dataset, count)
        dataset = Dataset.from_extractors(dataset)

    for subset in dataset.subsets().values():
        for item in subset:
            item_id = ""
            for i in range(len(item.id)):
                num = 97 + np.random.randint(0, 25)
                item_id += chr(num)
            item.id = item_id

            item.media = Image(data=np.ones(image_size))

            annotations = []

            labels = [anno for anno in item.annotations if anno.type == AnnotationType.label]
            for label in labels:
                label.label = (label.label + np.random.randint(0, 10)) % len(
                    dataset.categories()[AnnotationType.label]
                )

            annotations += labels

            bboxes = [anno for anno in item.annotations if anno.type == AnnotationType.bbox]
            for bbox in bboxes:
                x0 = bbox.points[0]
                bbox.points[0] = np.random.uniform(max(x0 - x0 / 2, 0), x0 + x0 / 2)
                y0 = bbox.points[1]
                bbox.points[1] = np.random.uniform(max(y0 - y0 / 2, 0), y0 + y0 / 2)
                x1 = bbox.points[2]
                bbox.points[2] = np.random.uniform(max(x1 - x1 / 2, 0), x1 + x1 / 2)
                y1 = bbox.points[3]
                bbox.points[3] = np.random.uniform(max(y1 - y1 / 2, 0), y1 + y1 / 2)

            annotations += bboxes

            masks = [anno for anno in item.annotations if anno.type == AnnotationType.mask]
            if masks:
                mask_size = image_size[:2]
                mask = np.ones(mask_size[0] * mask_size[1])

                mask_cat = dataset.categories()[AnnotationType.mask]

                for i in mask:
                    i = np.random.randint(0, 100) % len(mask_cat)
                mask = mask.reshape(mask_size)

                segm_ids = np.unique(mask)
                for segm_id in segm_ids:
                    annotations.append(Mask(image=lazy_extract_mask(mask, segm_id), label=segm_id))

            item.annotations = annotations

    return dataset


def lazy_extract_mask(mask, c):
    return lambda: mask == c
