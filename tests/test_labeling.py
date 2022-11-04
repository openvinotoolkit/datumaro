# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT
import tempfile
from unittest.case import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.components.project import Dataset

from .requirements import Requirements, mark_requirement


class LabelingTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_label_group(self):
        label_categories = LabelCategories()
        label_categories.add("car", parent="")
        label_categories.add("bicycle", parent="")

        label_categories.add_label_group("manmade", ["car", "bicycle"], group_type="exclusive")

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Label(
                            0,
                            id=0,
                        ),
                        Label(
                            1,
                            id=1,
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: label_categories,
            },
        )

        with tempfile.TemporaryDirectory() as temp_home:
            dataset.export(temp_home, format="datumaro")
            dataset_imported = Dataset.import_from(temp_home, format="datumaro")

        self.assertEqual(len(dataset_imported.categories()[AnnotationType.label].label_groups), 1)
