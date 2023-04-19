# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from . import modification, versioning


def get_project_commands():
    return [
        ("Project modification:", None, ""),
        ("add", modification.add, "Add dataset"),  # TODO: We will deprecated it with import soon.
        ("create", modification.create, "Create empty project"),
        ("export", modification.export, "Export dataset in some format"),
        ("import", modification.import_, "Import dataset"),
        ("remove", modification.remove, "Remove dataset"),
        ("", None, ""),
        ("Project versioning:", None, ""),
        ("checkout", versioning.checkout, "Switch to another branch or revision"),
        ("commit", versioning.commit, "Commit changes in tracked files"),
        ("log", versioning.log, "List history"),
        ("info", versioning.info, "Print project information"),
        ("status", versioning.status, "Display current branch and revision status"),
    ]
