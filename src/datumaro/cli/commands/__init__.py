# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=redefined-builtin

from . import (
    convert,
    detect_format,
    diff,
    download,
    explain,
    explore,
    filter,
    generate,
    info,
    merge,
    patch,
    stats,
    transform,
    validate,
)
from .require_project import get_project_commands

__all__ = [
    "get_non_project_commands",
    "get_project_commands",
]


def get_non_project_commands():
    return [
        ("convert", convert, "Convert dataset between formats"),
        ("detect", detect_format, "Detect the format of a dataset"),
        ("diff", diff, "Compare datasets"),
        ("dinfo", info, "Print dataset info"),
        ("download", download, "Download a publicly available dataset"),
        ("explain", explain, "Run Explainable AI algorithm for model"),
        ("filter", filter, "Filter dataset items"),
        ("generate", generate, "Generate synthetic dataset"),
        ("merge", merge, "Merge datasets"),
        ("patch", patch, "Update dataset from another one"),
        ("explore", explore, "Explore similar datasetitems of query"),
        ("stats", stats, "Compute dataset statistics"),
        ("transform", transform, "Modify dataset items"),
        ("validate", validate, "Validate dataset"),
    ]
