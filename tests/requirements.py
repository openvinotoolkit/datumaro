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
    DATUM_TELEMETRY = "Datumaro telemetry requirement"

    # GitHub issues (not bugs)
    # https://github.com/openvinotoolkit/datumaro/issues
    DATUM_231 = "Readable formats for CJK"
    DATUM_244 = "Add Snyk integration"
    DATUM_267 = "Add Image zip format"
    DATUM_274 = "Support the Open Images dataset"
    DATUM_280 = "Support KITTI dataset formats"
    DATUM_283 = "Create cli tests for testing convert command for VOC format"
    DATUM_399 = "Implement import for ADE20K dataset"

    # GitHub issues (bugs)
    # https://github.com/openvinotoolkit/datumaro/issues
    DATUM_BUG_219 = "Return format is not uniform"
    DATUM_BUG_257 = "Dataset.filter doesn't count removed items"
    DATUM_BUG_259 = "Dataset.filter fails on merged datasets"
    DATUM_BUG_314 = "Unsuccessful remap_labels"
    DATUM_BUG_402 = "Troubles running 'remap_labels' on ProjectDataset"
    DATUM_BUG_404 = "custom importer/extractor not loading"
    DATUM_BUG_425 = "Bug: concatenation for the different types in COCO format"
    DATUM_BUG_466 = "Can't correct import Open Images dataset without images"
    DATUM_BUG_470 = "Cannot to import Cityscapes dataset without images"


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
