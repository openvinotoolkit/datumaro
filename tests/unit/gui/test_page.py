# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from unittest import TestCase

from streamlit.testing.v1 import AppTest

from tests.requirements import Requirements, mark_requirement

cwd = os.getcwd()
app_path = os.path.join(cwd, "gui", "streamlit_app.py")

import sys

sys.path.append(os.path.join(cwd, "gui"))


class PageTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_smoke(self):
        """Test if the app runs without throwing an exception."""
        at = AppTest.from_file(app_path, default_timeout=10).run()
        assert not at.exception

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_sidebar(self):
        at = AppTest.from_file(app_path, default_timeout=10).run()
        assert len(at.sidebar.checkbox) == 4
