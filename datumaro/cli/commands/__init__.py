# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=redefined-builtin

from . import convert, detect_format, download, explain, filter, generate, merge, patch, search
from .require_project import get_project_commands

__all__ = [
    "get_non_project_commands",
    "get_project_commands",
]


def get_non_project_commands():
    return [
        ("convert", convert, "Convert dataset between formats"),
        ("detect-format", detect_format, "Detect the format of a dataset"),
        ("download", download, "Download a publicly available dataset"),
        ("explain", explain, "Run Explainable AI algorithm for model"),
        ("filter", filter, "Filter dataset items"),
        ("generate", generate, "Generate synthetic dataset"),
        ("merge", merge, "Merge datasets"),
        ("patch", patch, "Update dataset from another one"),
        ("search", search, "Search similar datasetitems of query"),
    ]
