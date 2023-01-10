# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Optional, Union

import numpy as np

from datumaro.components.annotation import HashKey
from datumaro.components.dataset import IDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.plugins.searcher import SearcherLauncher


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]  # max inner product value
    distH = 0.5 * (q - B1 @ B2.transpose())
    return distH


class Searcher:
    def __init__(
        self,
        dataset: IDataset,
        topk: int = 10,
    ) -> None:
        """
        Searcher for Datumaro dataitems

        Parameters
        ----------
        dataset:
            Datumaro dataset to search similar dataitem.
        topk:
            Number of images.
        """
        self._model = SearcherLauncher()
        self._text_model = SearcherLauncher(
            description="clip_text_ViT-B_32.xml", weights="clip_text_ViT-B_32.bin"
        )
        inference = dataset.run_model(self._model)
        self._topk = topk

        database_keys = []
        item_list = []

        for item in inference:
            for annotation in item.annotations:
                if isinstance(annotation, HashKey):
                    try:
                        hash_key = annotation.hash_key[0]
                        hash_key = self.unpack_hash_key(hash_key)
                        database_keys.append(hash_key)
                        item_list.append(item)
                    except Exception:
                        hash_key = None

        self._database_keys = database_keys
        self._item_list = item_list

    def unpack_hash_key(self, hash_key: List):
        hash_key_list = [hash_key[i : i + 2] for i in range(0, len(hash_key), 2)]
        hash_key = np.array([int(s, 16) for s in hash_key_list], dtype="uint8")
        hash_key = np.unpackbits(hash_key, axis=-1)
        return hash_key

    def search_topk(
        self,
        query: Union[DatasetItem, str, List[DatasetItem], List[str]],
        topk: Optional[int] = None,
    ):
        """
        Search topk similar results based on hamming distance for query DatasetItem
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
                    q_hash_key = None
                    for annotation in q.annotations:
                        if isinstance(annotation, HashKey):
                            q_hash_key = annotation.hash_key
                            break
                    query_hash_key_list.append(q_hash_key)
                elif isinstance(q, str):
                    q_hash_key = self._text_model.launch(q)[0][0].hash_key
                    query_hash_key_list.append(q_hash_key)

            sims = np.zeros(shape=database_keys.shape[0] * len(query_hash_key_list))
            for i, query_hash_key in enumerate(query_hash_key_list):
                query_hash_key = self.unpack_hash_key(query_hash_key[0])
                sims[i * db_len : (i + 1) * db_len] = calculate_hamming(
                    query_hash_key, database_keys
                )

            def cal_ind(x):
                if x >= db_len:
                    x = x % db_len
                return x

            ind = np.argsort(sims).tolist()
            ind = list(map(cal_ind, ind))

            item_list = np.array(self._item_list)[ind]
            result = item_list[:topk].tolist()
            return result

        if isinstance(query, DatasetItem):
            query_key = None
            for annotation in query.annotations:
                if isinstance(annotation, HashKey):
                    query_key = annotation.hash_key
                    break

            if not query_key:
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
            query_key = self._text_model.launch(query)[0][0].hash_key
        else:
            raise MediaTypeError(
                "Unexpected media type of query '%s'. "
                "Expected 'DatasetItem' or 'string', actual'%s'" % (query, type(query))
            )

        if not query_key:
            # media.data is None case
            raise ValueError("Query should have hash_key")

        query_key = self.unpack_hash_key(query_key[0])

        logits = calculate_hamming(query_key, database_keys)
        ind = np.argsort(logits)

        item_list = np.array(self._item_list)[ind]
        result = item_list[:topk].tolist()

        return result
