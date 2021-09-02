# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from contextlib import ExitStack, contextmanager
from functools import partial, wraps
from typing import Any, ContextManager, Dict, Optional
import threading

from attr import attrs

from datumaro.util import optional_arg_decorator


class Scope:
    """
    A context manager that allows to register error and exit callbacks.
    """

    _thread_locals = threading.local()

    @attrs(auto_attribs=True)
    class Handler:
        callback: Any
        enabled: bool = True
        ignore_errors: bool = False

        def __call__(self):
            if self.enabled:
                try:
                    self.callback()
                except: # pylint: disable=bare-except
                    if not self.ignore_errors:
                        raise

    def __init__(self):
        self._handlers = {}
        self._error_stack = ExitStack()
        self._exit_stack = ExitStack()
        self.enabled = True

    def on_error_do(self, callback, *args, name: Optional[str] = None,
            enabled: bool = True, ignore_errors: bool = False,
            fwd_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Registers a function to be called on scope exit because of an error.
        Equivalent to the "except" block of "try-except".
        """

        if args or kwargs or fwd_kwargs:
            if fwd_kwargs:
                kwargs.update(fwd_kwargs)
            callback = partial(callback, *args, **kwargs)

        name = name or hash(callback)
        assert name not in self._handlers, "Callback is already registered"

        handler = self.Handler(callback,
            enabled=enabled, ignore_errors=ignore_errors)
        self._handlers[name] = handler
        self._error_stack.callback(handler)
        return name

    def on_exit_do(self, callback, *args, name: Optional[str] = None,
            enabled: bool = True, ignore_errors: bool = False,
            fwd_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Registers a function to be called on scope exit unconditionally.
        Equivalent to the "finally" block of "try-except".
        """

        if args or kwargs or fwd_kwargs:
            if fwd_kwargs:
                kwargs.update(fwd_kwargs)
            callback = partial(callback, *args, **kwargs)

        name = name or hash(callback)
        assert name not in self._handlers, "Callback is already registered"

        handler = self.Handler(callback,
            enabled=enabled, ignore_errors=ignore_errors)
        self._handlers[name] = handler
        self._exit_stack.callback(handler)
        return name

    def add(self, cm: ContextManager) -> Any:
        """
        Enters a context manager and adds it to the exit stack.
        """

        return self._exit_stack.enter_context(cm)

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
        self.__exit__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, exc_traceback=None):
        if not self.enabled:
            return

        try:
            if exc_type:
                self._error_stack.__exit__(exc_type, exc_value, exc_traceback)
        finally:
            self._exit_stack.__exit__(exc_type, exc_value, exc_traceback)

    @classmethod
    def current(cls) -> 'Scope':
        return cls._thread_locals.current

    @contextmanager
    def as_current(self):
        previous = getattr(self._thread_locals, 'current', None)
        self._thread_locals.current = self
        try:
            yield
        finally:
            self._thread_locals.current = previous

@optional_arg_decorator
def scoped(func, arg_name=None):
    """
    A function decorator, which allows to do actions with the current scope,
    such as registering error and exit callbacks and context managers.
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        with Scope() as scope:
            if arg_name is None:
                with scope.as_current():
                    ret_val = func(*args, **kwargs)
            else:
                kwargs[arg_name] = scope
                ret_val = func(*args, **kwargs)
            return ret_val

    return wrapped_func

# Shorthands for common cases
def on_error_do(callback, *args, ignore_errors=False):
    return Scope.current().on_error_do(callback, *args,
        ignore_errors=ignore_errors)
on_error_do.__doc__ = Scope.on_error_do.__doc__

def on_exit_do(callback, *args, ignore_errors=False):
    return Scope.current().on_exit_do(callback, *args,
        ignore_errors=ignore_errors)
on_exit_do.__doc__ = Scope.on_exit_do.__doc__

def add(cm: ContextManager):
    return Scope.current().add(cm)
add.__doc__ = Scope.add.__doc__

def current():
    return Scope.current()
current.__doc__ = Scope.current.__doc__
