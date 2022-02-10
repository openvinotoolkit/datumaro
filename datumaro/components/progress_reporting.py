# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Iterable, Optional, TypeVar
import math

from datumaro.util.scope import on_exit_do, scoped

T = TypeVar('T')

class ProgressReporter:
    def get_frequency(self) -> float:
        raise NotImplementedError

    def start(self, total: int, *, desc: Optional[str] = None):
        raise NotImplementedError

    def report_status(self, progress: int):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError

    @scoped
    def iter(self, iterable: Iterable[T], *,
            total: Optional[int] = None,
            desc: Optional[str] = None
    ) -> Iterable[T]:
        """
        Traverses the iterable and reports progress simultaneously.

        Starts and finishes the progress bar automatically.

        Args:
            iterable - An iterable to be traversed
            total - The expected number of iterations. If not provided, will
              try to use iterable.__len__.
            desc - The status message

        Returns:
            An iterable over elements of the input sequence
        """

        if total is None and hasattr(iterable, '__len__'):
            total = len(iterable)

        self.start(total, desc=desc)
        on_exit_do(self.finish)

        if total:
            display_step = math.ceil(total * self.get_frequency())

        i = 0
        for i, elem in enumerate(iterable):
            if not total or i % display_step == 0:
                self.report_status(i)

            yield elem

        if i:
            self.report_status(i)
