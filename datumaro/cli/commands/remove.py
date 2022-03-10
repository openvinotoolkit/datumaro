# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from ..contexts.source import build_remove_parser as build_parser
from ..contexts.source import get_remove_sensitive_args as get_sensitive_args

__all__ = [
    "build_parser",
    "get_sensitive_args",
]
