# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT
from typing import Any, List

import pytest


class TestCaseHelper:
    """This class will exist until we complete the migration from unittest to pytest.
    It is designed to mimic unittest.TestCase behaviors to minimize the migration work labor cost.
    """

    def assertTrue(self, boolean: bool, err_msg: str = ""):
        assert boolean, err_msg

    def assertFalse(self, boolean: bool, err_msg: str = ""):
        assert not boolean, err_msg

    def assertEqual(self, item1: Any, item2: Any, err_msg: str = ""):
        assert item1 == item2, err_msg

    def assertListEqual(self, list1: List[Any], list2: List[Any], err_msg: str = ""):
        assert isinstance(list1, list) and isinstance(list2, list), err_msg
        assert len(list1) == len(list2), err_msg
        for item1, item2 in zip(list1, list2):
            self.assertEqual(item1, item2, err_msg)

    def fail(self, msg):
        pytest.fail(reason=msg)
