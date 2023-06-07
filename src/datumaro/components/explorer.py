# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Optional, Sequence, Union

import numpy as np

from datumaro.components.annotation import HashKey
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image, MediaElement
from datumaro.plugins.explorer import ExplorerLauncher


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]  # max inner product value
    distH = 0.5 * (q - B1 @ B2.transpose())
    return distH


def select_uninferenced_dataset(dataset):
    uninferenced_dataset = Dataset(media_type=MediaElement)
    for item in dataset:
        if not any(isinstance(annotation, HashKey) for annotation in item.annotations):
            uninferenced_dataset.put(item)
    return uninferenced_dataset


class Explorer:
    def __init__(
        self,
        *datasets: Sequence[Dataset],
        topk: int = 10,
    ) -> None:
        """
        Explorer for Datumaro dataitems

        Parameters
        ----------
        dataset:
            Datumaro dataset to explore similar dataitem.
        topk:
            Number of images.
        """
        self._model = None
        self._text_model = None
        self._topk = topk
        database_keys = []
        item_list = []

        datasets_to_infer = [select_uninferenced_dataset(dataset) for dataset in datasets]
        datasets = self.compute_hash_key(datasets, datasets_to_infer)

        for dataset in datasets:
            for item in dataset:
                for annotation in item.annotations:
                    if isinstance(annotation, HashKey):
                        try:
                            hash_key = annotation.hash_key[0]
                            hash_key = np.unpackbits(hash_key, axis=-1)
                            database_keys.append(hash_key)
                            item_list.append(item)
                        except Exception:
                            continue

        self._database_keys = database_keys
        self._item_list = item_list

    @property
    def model(self):
        if self._model is None:
            self._model = ExplorerLauncher(model_name="clip_visual_ViT-B_32")
        return self._model

    @property
    def text_model(self):
        if self._text_model is None:
            self._text_model = ExplorerLauncher(model_name="clip_text_ViT-B_32")
        return self._text_model

    def compute_hash_key(self, datasets, datasets_to_infer):
        for dataset in datasets_to_infer:
            if len(dataset) > 0:
                dataset.run_model(self.model, append_annotation=True)
        for dataset, dataset_to_infer in zip(datasets, datasets_to_infer):
            dataset.update(dataset_to_infer)
        return datasets

    def explore_topk(
        self,
        query: Union[DatasetItem, str, List[DatasetItem], List[str]],
        topk: Optional[int] = None,
    ):
        """
        Explore topk similar results based on hamming distance for query DatasetItem
        """
        if not topk:
            topk = self._topk

        if all(i is None for i in self._database_keys):
            # media.data is None case
            raise ValueError("Database should have hash_key")
        database_keys = np.stack(self._database_keys, axis=0)
        db_len = database_keys.shape[0]

        if isinstance(query, List):
            query_hash_key_list = []
            for q in query:
                if isinstance(q, DatasetItem):
                    q_hash_key = np.zeros((1, 64))
                    for annotation in q.annotations:
                        if isinstance(annotation, HashKey):
                            q_hash_key = annotation.hash_key
                            break
                    query_hash_key_list.append(q_hash_key)
                elif isinstance(q, str):
                    q_hash_key = self.text_model.launch(q)[0][0].hash_key
                    query_hash_key_list.append(q_hash_key)

            sims = np.zeros(shape=database_keys.shape[0] * len(query_hash_key_list))
            for i, query_hash_key in enumerate(query_hash_key_list):
                query_hash_key = np.unpackbits(query_hash_key[0], axis=-1)
                sims[i * db_len : (i + 1) * db_len] = calculate_hamming(
                    query_hash_key, database_keys
                )

            def cal_ind(x):
                x = x % db_len
                return x

            ind = np.argsort(sims).tolist()
            ind = list(map(cal_ind, ind))

            item_list = np.array(self._item_list)[ind]
            result = item_list[:topk].tolist()
            return result

        if isinstance(query, DatasetItem):
            query_key = np.zeros((1, 64))
            for annotation in query.annotations:
                if isinstance(annotation, HashKey):
                    query_key = annotation.hash_key
                    break

            if not query_key.any():
                try:
                    if not isinstance(query.media, Image):
                        raise MediaTypeError(
                            f"Media type should be Image, Current type={type(query.media)}"
                        )
                    query_key = self._model.launch(query.media.data)[0][0].hash_key
                except Exception:
                    # media.data is None case
                    pass

        elif isinstance(query, str):
            query_key = self.text_model.launch(query)[0][0].hash_key
        else:
            raise MediaTypeError(
                "Unexpected media type of query '%s'. "
                "Expected 'DatasetItem' or 'string', actual'%s'" % (query, type(query))
            )

        if not query_key.any():
            # media.data is None case
            raise ValueError("Query should have hash_key")

        query_key = np.unpackbits(query_key[0], axis=-1)
        logits = calculate_hamming(query_key, database_keys)
        ind = np.argsort(logits)

        item_list = np.array(self._item_list)[ind]
        result = item_list[:topk].tolist()

        return result
