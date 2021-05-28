# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT
import pytest


def mark_requirement(requirement):
    def wrapper(test_func):
        @pytest.mark.components(DatumaroComponent.Datumaro)
        @pytest.mark.component
        @pytest.mark.priority_high
        @pytest.mark.reqids(requirement)
        def test_wrapper(*args, **kwargs):
            return test_func(*args, **kwargs)
        return test_wrapper
    return wrapper


class DatumaroComponent:
    Datumaro = "datumaro"


class Requirements:
    DATUM_GENERAL_REQ = "Datumaro general requirement"
    DATUM_244 = "DATUM-244 Add Snyk integration"


class SkipMessages:
    NOT_IMPLEMENTED = "NOT IMPLEMENTED"
