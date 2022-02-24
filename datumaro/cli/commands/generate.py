# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from ..contexts.project import build_generate_parser as build_parser
from ..contexts.project import get_generate_sensitive_args as get_sensitive_args

__all__ = [
    'build_parser',
    'get_sensitive_args',
]
