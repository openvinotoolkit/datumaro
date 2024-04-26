# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from typing import Any, Dict, Iterator, Optional, Set

from datumaro.components.annotation import AnnotationType
from datumaro.components.media import MediaType
from datumaro.rust_api import DatumPageMapper as DatumPageMapperImpl

__all__ = ["DatumPageMapper"]


class DatumPageMapper:
    """Construct page maps for items and annotations from the JSON file,
    which are used for the stream importer.

    It also provides __iter__() to produce item and annotation dictionaries
    in stream manner after constructing the page map.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._impl = DatumPageMapperImpl(path)

    def __iter__(self) -> Iterator[Dict]:
        for item_key in self.iter_item_ids():
            yield self._impl.get_item_dict(item_key)

    def get_item_dict(self, item_key: str) -> Optional[Dict]:
        try:
            return self._impl.get_item_dict(item_key)
        except Exception as e:
            log.error(e)
            return None

    def __len__(self) -> int:
        return len(self._impl)

    def iter_item_ids(self) -> Iterator[str]:
        for item_id in self._impl.get_img_ids():
            yield item_id

    def __del__(self):
        pass

    @property
    def dm_format_version(self) -> Optional[str]:
        """Parse "dm_format_version" section from the given JSON file using the stream json parser"""
        return self._impl.dm_format_version()

    @property
    def media_type(self) -> Optional[MediaType]:
        """Parse "media_type" section from the given JSON file using the stream json parser"""
        media_type = self._impl.media_type()
        if media_type is not None:
            return MediaType(media_type)
        return None

    @property
    def ann_types(self) -> Optional[Set[AnnotationType]]:
        """Parse "media_type" section from the given JSON file using the stream json parser"""
        ann_types = self._impl.ann_types()
        if ann_types is not None:
            return ann_types
        return None

    @property
    def infos(self) -> Dict[str, Any]:
        """Parse "infos" section from the given JSON file using the stream json parser"""
        return self._impl.infos()

    @property
    def categories(self) -> Dict[str, Any]:
        """Parse "categories" section from the given JSON file using the stream json parser"""
        return self._impl.categories()

    def __reduce__(self):
        return (self.__class__, (self._path,))
