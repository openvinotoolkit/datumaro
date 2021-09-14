# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from ..contexts.project import build_stats_parser as build_parser
from ..contexts.project import \
    get_stats_params_with_paths as get_sensitive_args

__all__ = [
    'build_parser',
    'get_sensitive_args',
]
