# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from datumaro.util import dump_json, parse_json


class Mapper(ABC):
    @staticmethod
    @abstractmethod
    def forward(obj: Any) -> bytes:
        """Convert an object to bytes."""

    @staticmethod
    @abstractmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[Any, int]:
        """Build an object from bytes."""


class StringMapper(Mapper):
    @staticmethod
    def forward(obj: str) -> bytes:
        obj = obj.encode()
        length = len(obj)
        return struct.pack(f"<I{length}s", length, obj)

    @staticmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[str, int]:
        length = struct.unpack_from("<I", _bytes, offset)[0]
        offset += 4
        string = struct.unpack_from(f"<{length}s", _bytes, offset)[0].decode()
        return string, offset + length


class ListMapper(Mapper):
    _format = ""

    @classmethod
    def forward(cls, obj: List[Any]) -> bytes:
        length = len(obj)
        return struct.pack(f"I{length}{cls._format}", length, *obj)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[List[Any], int]:
        (length,) = struct.unpack_from("<I", _bytes, offset)
        offset += 4
        obj = struct.unpack_from(f"<{length}{cls._format}", _bytes, offset)
        offset += 4 * length
        return obj, offset


class IntListMapper(ListMapper):
    _format = "i"


class FloatListMapper(ListMapper):
    _format = "f"


class DictMapper(Mapper):
    @staticmethod
    def forward(obj: Dict[str, Any]) -> bytes:
        if len(obj) == 0:
            msg = b""
        else:
            msg = dump_json(obj)
        length = len(msg)
        return struct.pack(f"<I{length}s", length, msg)

    @staticmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[Dict[str, Any], int]:
        length = struct.unpack_from("<I", _bytes, offset)[0]
        offset += 4
        if length == 0:
            parsed_dict = {}
        else:
            parsed_dict = parse_json(_bytes[offset : offset + length])
        return parsed_dict, offset + length
