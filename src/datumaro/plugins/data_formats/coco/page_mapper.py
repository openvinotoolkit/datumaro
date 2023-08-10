# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datumaro.rust_api import CocoPageMapper as CocoPageMapperImpl

__all__ = ["COCOPageMapper"]


class COCOPageMapper:
    """Construct page maps for items and annotations from the JSON file,
    which are used for the stream importer.

    It also provides __iter__() to produce item and annotation dictionaries
    in stream manner after constructing the page map.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._impl = CocoPageMapperImpl(path)

    def __iter__(self) -> Iterator[Tuple[Dict, List[Dict]]]:
        for item_key in self.iter_item_ids():
            yield self._impl.get_item_dict(item_key), self._impl.get_anns_dict(item_key)

    def get_item_dict(self, item_key: int) -> Optional[Dict]:
        try:
            return self._impl.get_item_dict(item_key)
        except Exception as e:
            log.error(e)
            return None

    def get_anns_dict(self, item_key: int) -> Optional[List[Dict]]:
        try:
            return self._impl.get_anns_dict(item_key)
        except Exception as e:
            log.error(e)
            return None

    def __len__(self) -> int:
        return len(self._impl)

    def iter_item_ids(self) -> Iterator[int]:
        for item_id in self._impl.get_img_ids():
            yield item_id

    def __del__(self):
        pass

    def stream_parse_categories_data(self) -> Dict[str, Any]:
        """Parse "categories" section from the given JSON file using the stream json parser"""
        return self._impl.categories()

    def __reduce__(self):
        return (self.__class__, (self._path,))
