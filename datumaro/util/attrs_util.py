# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import inspect

import attr


def not_empty(inst, attribute, x):
    assert len(x) != 0, x

def default_if_none(conv):
    def validator(inst, attribute, value):
        default = attribute.default
        if value is None:
            if callable(default):
                value = default()
            elif isinstance(default, attr.Factory):
                value = default.factory()
            else:
                value = default
        else:
            dst_type = None
            if attribute.type and inspect.isclass(attribute.type) and \
                    not hasattr(attribute.type, '__origin__'):
                #       ^^^^^^^
                # Disallow Generics in python 3.6
                # Can be dropped with 3.6 support. Generics canot be used
                # in isinstance() checks.

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
