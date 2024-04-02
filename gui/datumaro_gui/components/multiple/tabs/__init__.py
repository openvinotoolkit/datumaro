# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .compare import main as call_compare
from .export import main as call_export
from .general import main as call_general
from .merge import main as call_merge
from .transform import main as call_transform

__all__ = [
    "call_compare",
    "call_export",
    "call_general",
    "call_merge",
    "call_transform",
]
