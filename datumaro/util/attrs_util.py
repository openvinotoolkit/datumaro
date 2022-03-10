# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import inspect

import attrs


def not_empty(inst, attribute, x):
    assert len(x) != 0, x


def default_if_none(conv):
    def validator(inst, attribute, value):
        default = attribute.default
        if value is None:
            if callable(default):
                value = default()
            elif isinstance(default, attrs.Factory):
                value = default.factory()
            else:
                value = default
        else:
            dst_type = None
            if attribute.type and inspect.isclass(attribute.type):
                dst_type = attribute.type
            elif conv and inspect.isclass(conv):
                dst_type = conv

            if not dst_type or not isinstance(value, dst_type):
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
