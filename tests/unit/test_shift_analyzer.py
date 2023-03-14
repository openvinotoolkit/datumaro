from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.shift_analyzer import ShiftAnalyzer

from ..requirements import Requirements, mark_requirement


class SearcherTest(TestCase):
    @property
    def test_dataset(self):
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        train_img[2, :] = 0
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)
        test_img[2, :] = 255

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ]
        )
        return dataset

    @property
    def test_dataset_black_white(self):
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ]
        )
        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_covariate_shift_analyze_emd(self):
        """
        <b>Description:</b>
        Check that search topk data for image query.

        <b>Input data:</b>
        Image query and dataset to retrieval, in this case use white and black image.

        <b>Expected results:</b>
        Datasetitem with same hash_key as query.

        <b>Steps</b>
        1. Import dataset and set one of datasetitems as query.
        2. Set Searcher and try search_topk to find similar media of image query.
        3. Check whether each result have same subset as query.
        """

        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(id=0, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]),
                DatasetItem(id=1, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]),
                DatasetItem(id=2, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(id=0, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]),
                DatasetItem(id=1, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]),
                DatasetItem(id=2, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]),
                DatasetItem(id=3, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]),
                DatasetItem(id=4, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]),
                DatasetItem(id=5, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(3)]),
            ],
            categories=["a", "b", "c", "d"],
        )

        shift_analyzer = ShiftAnalyzer()
        result = shift_analyzer.compute_covariate_shift([dataset1, dataset2], method="emd")
        print(result)
        # self.assertEqual(query.subset, result[0].subset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_covariate_shift_analyze_fid(self):
        """
        <b>Description:</b>
        Check that search topk data for image query.

        <b>Input data:</b>
        Image query and dataset to retrieval, in this case use white and black image.

        <b>Expected results:</b>
        Datasetitem with same hash_key as query.

        <b>Steps</b>
        1. Import dataset and set one of datasetitems as query.
        2. Set Searcher and try search_topk to find similar media of image query.
        3. Check whether each result have same subset as query.
        """

        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(id=0, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]),
                DatasetItem(id=1, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]),
                DatasetItem(id=2, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(id=0, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]),
                DatasetItem(id=1, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]),
                DatasetItem(id=2, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]),
                DatasetItem(id=3, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]),
                DatasetItem(id=4, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]),
                DatasetItem(id=5, media=Image(data=np.ones((8, 8, 3))), annotations=[Label(3)]),
            ],
            categories=["a", "b", "c", "d"],
        )

        shift_analyzer = ShiftAnalyzer()
        result = shift_analyzer.compute_covariate_shift([dataset1, dataset2], method="fid")
        print(result)
        # self.assertEqual(query.subset, result[0].subset)
