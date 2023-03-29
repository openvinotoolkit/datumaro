# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base64
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


def _b64encode(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _b64encode(v)
    elif isinstance(obj, (list, tuple)):
        _obj = []
        for v in obj:
            _obj.append(_b64encode(v))
        if isinstance(obj, list):
            _obj = list(_obj)
        obj = _obj
    elif isinstance(obj, bytes):
        obj = base64.b64encode(obj).decode()
    return obj


def _b64decode(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _b64decode(v)
    elif isinstance(obj, (list, tuple)):
        _obj = []
        for v in obj:
            _obj.append(_b64decode(v))
        if isinstance(obj, list):
            _obj = list(_obj)
        obj = _obj
    elif isinstance(obj, str):
        try:
            _obj = base64.b64decode(obj)
            if base64.b64encode(_obj).decode() == obj:
                obj = _obj
        except Exception:
            pass
    return obj


class DictMapper(Mapper):
    @staticmethod
    def forward(obj: Dict[str, Any]) -> bytes:
        if len(obj) == 0:
            msg = b""
        else:
            msg = dump_json(_b64encode(obj))
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

        return _b64decode(parsed_dict), offset + length
