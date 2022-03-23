# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT


def ensure_type(cls, conv=None):
    """
    Checks value type, and if it's wrong, tries to build a class
    instance from the value, treating it as the c-tor (or converter) arg.
    """

    conv = conv or cls

    def _converter(v):
        if not isinstance(v, cls):
            v = conv(v)
        return v

    return _converter


def has_length(n):
    def _validator(inst, attribute, x):
        assert len(x) != 0, x

    return _validator


def ensure_type_kw(cls):
    """
    Checks value type, and if it's wrong, tries to build a class
    instance from the value, treating it as the c-tor kwargs.
    """

    def _converter(arg):
        if isinstance(arg, cls):
            return arg
        else:
            return cls(**arg)

    return _converter


def optional_cast(cls, *, conv=None):
    """
    Equivalent to:
    attrs.converters.pipe(
        attrs.converters.optional(ensure_type(t, c)),
    )

    But provides better performance (mostly, due to less function calls). It
    may sound insignificant, but in datasets, there are millions of calls,
    and function invocations start to hit.
    """

    conv = conv or cls

    def _conv(v):
        if v is not None and not isinstance(v, cls):
            v = conv(v)
        return v

    return _conv


def cast_with_default(cls, *, conv=None, factory=None):
    """
    Equivalent to:
    attrs.converters.pipe(
        attrs.converters.optional(ensure_type(t, c)),
        attrs.converters.default_if_none(factory=f),
    )

    But provides better performance (mostly, due to less function calls). It
    may sound insignificant, but in datasets, there are millions of calls,
    and function invocations start to hit.
    """

    factory = factory or cls
    conv = conv or cls

    def _conv(v):
        if v is None:
            v = factory()
        elif not isinstance(v, cls):
            v = conv(v)
        return v

    return _conv


def not_empty(inst, attribute, x):
    assert len(x) != 0, x
