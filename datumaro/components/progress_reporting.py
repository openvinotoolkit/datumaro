# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import ContextManager, Iterable, Optional, Tuple, TypeVar
import math

T = TypeVar('T')

class ProgressReporter(ContextManager['ProgressReporter']):
    """
    Only one set of methods must be called:
    - init - report_status - close
    - iter - close
    - split

    Must be close()-d after use, can be used as a context manager.
    Must not be reused after closing.
    """

    def close(self):
        """Closes the progress bar"""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.close()

    @property
    def frequency(self) -> float:
        """
        Returns reporting frequency.

        For example, 0.1 would mean every 10%.
        """
        raise NotImplementedError

    def init(self, total: int, *, desc: Optional[str] = None):
        """Initializes the progress bar"""
        raise NotImplementedError

    def report_status(self, progress: int):
        """Updates the progress bar"""
        raise NotImplementedError

    def iter(self, iterable: Iterable[T], *,
            total: Optional[int] = None,
            desc: Optional[str] = None
    ) -> Iterable[T]:
        """
        Traverses the iterable and reports progress simultaneously.

        Starts the progress bar automatically.

        Args:
            iterable: An iterable to be traversed
            total: The expected number of iterations. If not provided, will
              try to use iterable.__len__.
            desc: The status message

        Returns:
            An iterable over elements of the input sequence
        """

        if total is None and hasattr(iterable, '__len__'):
            total = len(iterable)

        self.init(total, desc=desc)

        if total:
            display_step = math.ceil(total * self.frequency)

        for i, elem in enumerate(iterable):
            if not total or i % display_step == 0:
                self.report_status(i)

            yield elem

    def split(self, count: int) -> Tuple[ProgressReporter, ...]:
        """
        Splits the progress bar into few independent parts
        In case of 0 must return an empty tuple.
        """
        raise NotImplementedError

class NullProgressReporter(ProgressReporter):
    @property
    def frequency(self) -> float:
        return 0

    def init(self, total: int, *, desc: Optional[str] = None):
        pass

    def report_status(self, progress: int):
        pass

    def close(self):
        pass

    def iter(self, iterable: Iterable[T], *,
            total: Optional[int] = None,
            desc: Optional[str] = None
    ) -> Iterable[T]:
        yield from iterable

    def split(self, count: int) -> Tuple[ProgressReporter]:
        return (self, ) * count
