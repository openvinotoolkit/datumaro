# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from ..contexts.source import build_add_parser as build_parser
from ..contexts.source import get_add_params_with_paths as get_sensitive_args

__all__ = [
    'build_parser',
    'get_sensitive_args',
]
