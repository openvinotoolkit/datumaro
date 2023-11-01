# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .compare import main as call_compare
from .explore import main as call_explore
from .export import main as call_export
from .general import main as call_general
from .merge import main as call_merge
from .transform import main as call_transform
from .validate import main as call_validate
from .visualize import main as call_visualize

__all__ = [
    "call_compare",
    "call_explore",
    "call_export",
    "call_general",
    "call_merge",
    "call_transform",
    "call_validate",
    "call_visualize",
]
