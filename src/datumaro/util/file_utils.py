# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import re
from typing import Union


def to_bytes(size: Union[str, int]) -> int:
    if isinstance(size, int):
        return size
    assert isinstance(size, str)

    match = re.match(r"^([\d\.]+)\s*([a-zA-Z]{0,3})$", size.strip())

    if match is None:
        raise ValueError(f"Cannot parse {size} string.")

    units = {
        "": 1,
        "B": 1,
        "KIB": 2**10,
        "MIB": 2**20,
        "GIB": 2**30,
        "TIB": 2**40,
        "KB": 10**3,
        "MB": 10**6,
        "GB": 10**9,
        "TB": 10**12,
        "K": 2**10,
        "M": 2**20,
        "G": 2**30,
        "T": 2**40,
    }

    number, unit = int(match.group(1)), match.group(2).upper()

    if unit not in units:
        raise ValueError(f"{size} has disallowed unit ({unit}).")

    return number * units[unit]
