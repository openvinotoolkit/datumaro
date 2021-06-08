# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import typing

from attr import attrs

import pytest

class DatumaroComponent:
    Datumaro = "datumaro"


class Requirements:
    # Exact requirements
    DATUM_GENERAL_REQ = "Datumaro general requirement"

    # GitHub issues (not bugs)
    # https://github.com/openvinotoolkit/datumaro/issues
    DATUM_244 = "Add Snyk integration"

    # GitHub issues (bugs)
    # https://github.com/openvinotoolkit/datumaro/issues
    DATUM_BUG_219 = "Return format is not uniform"


class SkipMessages:
    NOT_IMPLEMENTED = "NOT IMPLEMENTED"


@attrs(auto_attribs=True)
class _CombinedDecorator:
    decorators: typing.List[typing.Callable]

    def __call__(self, function):
        for d in reversed(self.decorators):
            function = d(function)

        return function


_SHARED_DECORATORS = [
    pytest.mark.components(DatumaroComponent.Datumaro),
    pytest.mark.component,
    pytest.mark.priority_medium,
]

def mark_requirement(requirement):
    return _CombinedDecorator([
        *_SHARED_DECORATORS,
        pytest.mark.reqids(requirement),
    ])

def mark_bug(bugs):
    return _CombinedDecorator([
        *_SHARED_DECORATORS,
        pytest.mark.bugs(bugs),
    ])
