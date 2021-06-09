# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest


def mark_requirement(requirement):
    def wrapper(test_func):
        @pytest.mark.components(DatumaroComponent.Datumaro)
        @pytest.mark.component
        @pytest.mark.priority_medium
        @pytest.mark.reqids(requirement)
        def test_wrapper(*args, **kwargs):
            return test_func(*args, **kwargs)
        return test_wrapper
    return wrapper

def mark_bug(bugs):
    def wrapper(test_func):
        @pytest.mark.components(DatumaroComponent.Datumaro)
        @pytest.mark.component
        @pytest.mark.priority_medium
        @pytest.mark.bugs(bugs)
        def test_wrapper(*args, **kwargs):
            return test_func(*args, **kwargs)
        return test_wrapper
    return wrapper


class DatumaroComponent:
    Datumaro = "datumaro"


class Requirements:
    # Exact requirements
    DATUM_GENERAL_REQ = "Datumaro general requirement"

    # GitHub issues (not bugs)
    # https://github.com/openvinotoolkit/datumaro/issues
    DATUM_244 = "Add Snyk integration"
    DATUM_283 = "Create cli tests for testing convert command for VOC format"

    # GitHub issues (bugs)
    # https://github.com/openvinotoolkit/datumaro/issues
    DATUM_BUG_219 = "Return format is not uniform"


class SkipMessages:
    NOT_IMPLEMENTED = "NOT IMPLEMENTED"
