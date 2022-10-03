# Copyright (C) 2022 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import io
from contextlib import redirect_stdout

import pytest


@pytest.fixture
def fxt_stdout():
    stream = io.StringIO()
    with redirect_stdout(stream):
        yield stream
