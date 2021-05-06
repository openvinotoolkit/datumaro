# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import attr
from inspect import isclass

_NOTSET = object()

def not_empty(inst, attribute, x):
    assert len(x) != 0, x

def default_if_none(conv):
    def validator(inst, attribute, value):
        default = attribute.default
        atype = attribute.type or (default.factory
            if isinstance(default, attr.Factory) and isclass(default.factory)
            else None)
        if value is None:
            if callable(default):
                value = default()
            elif isinstance(default, attr.Factory):
                value = default.factory()
            else:
                value = default
        elif not isinstance(value, atype or conv):
            value = conv(value)
        setattr(inst, attribute.name, value)
    return validator

def ensure_cls(c):
    def converter(arg):
        if isinstance(arg, c):
            return arg
        else:
            return c(**arg)
    return converter