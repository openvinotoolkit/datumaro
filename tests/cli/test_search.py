from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.util.scope import scope_add, scoped
from datumaro.util.test_utils import TestDir
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement


class SearchTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_search_dataset(self):
        test_dir = scope_add(TestDir())

        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        train_img[2, :] = 0
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)
        test_img[2, :] = 255
        train_Image = Image(data=train_img)

        train_image_path = train_Image.path

        Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=train_Image,
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=train_Image,
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ]
        ).export(test_dir, "datumaro")

        run(
            self,
            "search",
            "-q",
            train_image_path,
            "-topk",
            "2",
        )
