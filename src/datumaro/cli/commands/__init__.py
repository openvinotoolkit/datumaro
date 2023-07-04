# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=redefined-builtin

from . import (
    compare,
    convert,
    detect_format,
    download,
    explain,
    explore,
    filter,
    generate,
    info,
    merge,
    patch,
    prune,
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
        ("compare", compare, "Compare datasets"),
        ("dinfo", info, "Print dataset info"),
        ("download", download, "Download a publicly available dataset"),
        ("explain", explain, "Run Explainable AI algorithm for model"),
        ("explore", explore, "Explore similar datasetitems of query"),
        ("filter", filter, "Filter dataset items"),
        ("generate", generate, "Generate synthetic dataset"),
        ("merge", merge, "Merge datasets"),
        ("patch", patch, "Update dataset from another one"),
        ("prune", prune, "Prune dataset"),
        ("stats", stats, "Compute dataset statistics"),
        ("transform", transform, "Modify dataset items"),
        ("validate", validate, "Validate dataset"),
    ]
