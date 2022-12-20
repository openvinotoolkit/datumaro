# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Optional, Union

import numpy as np

from datumaro.components.annotation import HashKey
from datumaro.components.dataset import IDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.media import MultiframeImage, PointCloud, Video
from datumaro.components.model_inference import hash_inference


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
            Number of images
        """
        self._dataset = dataset
        self._topk = topk

        database_keys = []
        item_list = []
        for datasetitem in self._dataset:
            if type(datasetitem.media) in [Video, PointCloud, MultiframeImage]:
                raise MediaTypeError(
                    f"Media type should be Image, Current type={type(datasetitem.media)}"
                )
            try:
                hash_key = datasetitem.set_hash_key[0]
                hash_key = self.unpack_hash_key(hash_key)
                database_keys.append(hash_key)
                item_list.append(datasetitem)
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

        if not self._database_keys:
            # media.data is None case
            raise ValueError("Database should have hash_key")
        database_keys = np.stack(self._database_keys, axis=0)

        if isinstance(query, List):
            topk_for_query = int(topk // len(query)) * 2
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
                    q_hash_key = hash_inference(q)
                    query_hash_key_list.append(q_hash_key)

            for query_hash_key in query_hash_key_list:
                result_list = []
                query_hash_key = self.unpack_hash_key(query_hash_key[0])
                logits = calculate_hamming(query_hash_key, database_keys)
                ind = np.argsort(logits)

                item_list = np.array(self._item_list)[ind]
                result_list.extend(item_list[:topk_for_query].tolist())
            return np.random.choice(result_list, topk)

        if isinstance(query, DatasetItem):
            query_key = None
            for annotation in query.annotations:
                if isinstance(annotation, HashKey):
                    query_key = annotation.hash_key
                    break

            if not query_key:
                query_key = query.set_hash_key
        elif isinstance(query, str):
            query_key = hash_inference(query)
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
