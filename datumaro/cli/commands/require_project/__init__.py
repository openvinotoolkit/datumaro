# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from . import dataset_operations, modification, versioning


def get_project_commands():
    return [
        ("Project modification:", None, ""),
        ("add", modification.add, "Add dataset"),
        ("create", modification.create, "Create empty project"),
        ("import", modification.import_, "Import dataset"),
        ("remove", modification.remove, "Remove dataset"),
        ("", None, ""),
        ("Project versioning:", None, ""),
        ("checkout", versioning.checkout, "Switch to another branch or revision"),
        ("commit", versioning.commit, "Commit changes in tracked files"),
        ("log", versioning.log, "List history"),
        ("status", versioning.status, "Display current status"),
        ("pinfo", versioning.info, "Print project info"),
        ("", None, ""),
        ("Dataset operations:", None, ""),
        ("diff", dataset_operations.diff, "Compare datasets"),
        ("export", dataset_operations.export, "Export dataset in some format"),
        ("stats", dataset_operations.stats, "Compute dataset statistics"),
        ("transform", dataset_operations.transform, "Modify dataset items"),
        ("validate", dataset_operations.validate, "Validate dataset"),
        ("dinfo", dataset_operations.info, "Print dataset info"),
    ]
