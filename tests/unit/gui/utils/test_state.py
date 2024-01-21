# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os

from tests.requirements import Requirements, mark_requirement

cwd = os.getcwd()

import sys

sys.path.append(os.path.join(cwd, "gui"))
from gui.datumaro_gui.utils.dataset.data_loader import DatasetHelper
from gui.datumaro_gui.utils.dataset.state import get_download_folder_path, reset_state, reset_subset


class StateTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_download_folder_path(self):
        result = get_download_folder_path()
        assert isinstance(result, str)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reset_subset(self):
        state = {"subset": None, "subset_1": None, "subset_2": None}
        reset_subset(state)
        assert state["subset"] == []
        assert state["subset_1"] == []
        assert state["subset_2"] == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reset_state(self):
        keys = ["key1", "key2"]
        state = {"key1": None, "key2": None}
        reset_state(keys, state)
        assert state["key1"] is None
        assert state["key2"] is None
