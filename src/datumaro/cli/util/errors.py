# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from attr import attrib, attrs

from datumaro.components.errors import DatumaroError


class CliException(DatumaroError):
    pass


@attrs
class WrongRevpathError(CliException):
    problems = attrib()

    def __str__(self):
        return "Failed to parse revspec:\n  " + "\n  ".join(str(p) for p in self.problems)
