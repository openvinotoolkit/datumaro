
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import attr
from contextlib import ExitStack
from distutils.util import strtobool as str_to_bool # pylint: disable=unused-import
from functools import partial, wraps
from itertools import islice
from typing import Iterable, Tuple


NOTSET = object()

def find(iterable, pred=lambda x: True, default=None):
    return next((x for x in iterable if pred(x)), default)

def cast(value, type_conv, default=None):
    if value is None:
        return default
    try:
        return type_conv(value)
    except Exception:
        return default

def to_snake_case(s):
    if not s:
        return ''

    name = [s[0].lower()]
    for idx, char in enumerate(s[1:]):
        idx = idx + 1
        if char.isalpha() and char.isupper():
            prev_char = s[idx - 1]
            if not (prev_char.isalpha() and prev_char.isupper()):
                # avoid "HTML" -> "h_t_m_l"
                name.append('_')
            name.append(char.lower())
        else:
            name.append(char)
    return ''.join(name)

def pairs(iterable):
    a = iter(iterable)
    return zip(a, a)

def take_by(iterable, count):
    """
    Returns elements from the input iterable by batches of N items.
    ('abcdefg', 3) -> ['a', 'b', 'c'], ['d', 'e', 'f'], ['g']
    """

    it = iter(iterable)
    while True:
        batch = list(islice(it, count))
        if len(batch) == 0:
            break

        yield batch

def filter_dict(d, exclude_keys):
    return { k: v for k, v in d.items() if k not in exclude_keys }

def parse_str_enum_value(value, enum_class, default=NOTSET,
        unknown_member_error=None):
    if value is None and default is not NOTSET:
        value = default
    elif isinstance(value, str):
        try:
            value = enum_class[value]
        except KeyError:
            raise ValueError((unknown_member_error or
                    "Unknown element of {cls} '{value}'. "
                    "The only known are: {available}") \
                .format(
                    cls=enum_class.__name__,
                    value=value,
                    available=', '.join(e.name for e in enum_class)
                )
            )
    elif isinstance(value, enum_class):
        pass
    else:
        raise TypeError("Expected value type string or %s, but got %s" % \
            (enum_class.__name__, type(value).__name__))
    return value

def escape(s: str, escapes: Iterable[Tuple[str, str]]) -> str:
    """
    'escapes' is an iterable of (pattern, substitute) pairs
    """

    for pattern, sub in escapes:
        s = s.replace(pattern, sub)
    return s

def unescape(s: str, escapes: Iterable[Tuple[str, str]]) -> str:
    """
    'escapes' is an iterable of (pattern, substitute) pairs
    """

    for pattern, sub in escapes:
        s = s.replace(sub, pattern)
    return s

def optional_arg_decorator(fn):
    @wraps(fn)
    def wrapped_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return fn(args[0], **kwargs)

        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args, **kwargs)

            return real_decorator

    return wrapped_decorator

class Rollback:
    @attr.attrs
    class Handler:
        callback = attr.attrib()
        enabled = attr.attrib(default=True)
        ignore_errors = attr.attrib(default=False)

        def __call__(self):
            if self.enabled:
                try:
                    self.callback()
                except: # pylint: disable=bare-except
                    if not self.ignore_errors:
                        raise

    def __init__(self):
        self._handlers = {}
        self._stack = ExitStack()
        self.enabled = True

    def add(self, callback, *args,
            name=None, enabled=True, ignore_errors=False,
            fwd_kwargs=None, **kwargs):
        if args or kwargs or fwd_kwargs:
            if fwd_kwargs:
                kwargs.update(fwd_kwargs)
            callback = partial(callback, *args, **kwargs)
        name = name or hash(callback)
        assert name not in self._handlers
        handler = self.Handler(callback,
            enabled=enabled, ignore_errors=ignore_errors)
        self._handlers[name] = handler
        self._stack.callback(handler)
        return name

    do = add # readability alias

    def enable(self, name=None):
        if name:
            self._handlers[name].enabled = True
        else:
            self.enabled = True

    def disable(self, name=None):
        if name:
            self._handlers[name].enabled = False
        else:
            self.enabled = False

    def clean(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        return self

    # pylint: disable=redefined-builtin
    def __exit__(self, type=None, value=None, traceback=None):
        if type is None:
            return
        if not self.enabled:
            return
        self._stack.__exit__(type, value, traceback)
    # pylint: enable=redefined-builtin

@optional_arg_decorator
def error_rollback(func, arg_name='on_error', implicit=False):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        with Rollback() as manager:
            if implicit:
                fglobals = func.__globals__

                has_arg = arg_name in fglobals
                old_val = fglobals.get(arg_name)
                fglobals[arg_name] = manager
                try:
                    func(*args, **kwargs)
                finally:
                    if has_arg:
                        func.__globals__[arg_name] = old_val
                    else:
                        func.__globals__.pop(arg_name)
            else:
                kwargs[arg_name] = manager
                func(*args, **kwargs)
    return wrapped_func
