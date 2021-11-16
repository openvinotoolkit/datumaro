# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import IntEnum
from typing import Callable, Iterator, List, Optional, Sequence, TextIO
import contextlib
import glob
import os.path as osp

from typing_extensions import NoReturn


class FormatDetectionConfidence(IntEnum):
    """
    Represents the level of confidence that a detector has in a dataset
    belonging to the detector's format.
    """

    LOW = 10
    """
    The dataset seems to belong to the format, but the format is too loosely
    defined to be able to distinguish it from other formats.
    """
    MEDIUM = 20
    """
    The dataset seems to belong to the format, and is likely not to belong
    to any other format.
    """
    # There's no HIGH confidence yet, because none of the detectors
    # deserve it. It's reserved for when the detector is sure that
    # the dataset belongs to the format; for example, because the format
    # has explicit identification via magic numbers/files.

# All confidence levels should be positive for a couple of reasons:
# * It makes it possible to use 0 or a negative number as a special
#   value that is guaranteed to be less than any real value.
# * It makes sure that every confidence level is a true value.
assert all(level > 0 for level in FormatDetectionConfidence)

class FormatRequirementsUnmet(Exception):
    """
    Represents a situation where a dataset does not meet the requirements
    of a given dataset format.
    More specifically, if this exception is raised, then it is necessary
    (but may not be sufficient) for the dataset to meet at least
    one of these requirements to be detected as being in that format.

    Each element of `failed_alternatives` must be a human-readable
    statement describing a requirement that was not met.

    Must not be constructed or raised directly; use `FormatDetectionContext`
    methods.
    """

    def __init__(self, failed_alternatives: Sequence[str]) -> None:
        assert failed_alternatives
        self.failed_alternatives = tuple(failed_alternatives)

class FormatDetectionContext:
    """
    An instance of this class is given to a dataset format detector.
    See the `FormatDetector` documentation. The class should not
    be instantiated directly.

    A context encapsulates information about the dataset whose format
    is being detected. It also offers methods that place requirements
    on that dataset. Each such method raises a `FormatRequirementsUnmet`
    exception if the requirement is not met. If the requirement _is_
    met, the return value depends on the method.
    """

    class _AlternativesContext:
        failed_alternatives: List[str]
        had_successful_alternatives: bool

        def __init__(self) -> None:
            self.failed_alternatives = []
            self.had_successful_alternatives = False

    # This points to an _AlternativesContext when and only when the detector
    # is directly within an `alternatives` block.
    _alternatives_context: Optional[_AlternativesContext]

    def __init__(self, root_path: str) -> None:
        self._root_path = root_path
        self._alternatives_context = None

    @property
    def root_path(self) -> str:
        """
        Returns the path to the root directory of the dataset.
        Detectors should avoid using this property in favor of specific
        requirement methods.
        """
        return self._root_path

    def _is_path_within_root(self, path: str) -> bool:
        """
        Checks that `path` is a relative path and does not attempt to leave
        the dataset root by using `..` segments.

        Requirement-placing methods that use this to verify their arguments
        should raise a FormatRequirementsUnmet rather than a "hard" error like
        AssertionError if False is returned. The reason is that the path passed
        by the detector might not have been hardcoded, and instead might have
        been acquired from another file in the dataset. In that case, an invalid
        pattern signifies a problem with the dataset, not with the detector.
        """
        if osp.isabs(path) or osp.splitdrive(path)[0]:
            return False

        path = osp.normpath(path)
        if path.startswith('..' + osp.sep):
            return False

        return True

    def _start_requirement(self) -> None:
        assert not self._alternatives_context, \
            "checks can't be done directly within 'alternatives' blocks"

    def fail(self, requirement: str) -> NoReturn:
        """
        Places a requirement that is never met. `requirement` must contain
        a human-readable description of the requirement.
        """
        self._start_requirement()

        raise FormatRequirementsUnmet((requirement,))

    def require_file(self, pattern: str) -> str:
        """
        Places the requirement that the dataset contains at least one file whose
        relative path matches the given pattern. The pattern must be a glob-like
        pattern; `**` can be used to indicate a sequence of zero or more
        subdirectories.
        If the pattern does not describe a relative path, or refers to files
        outside the dataset root, the requirement is considered unmet.
        If the requirement is met, the relative path to one of the files that
        match the pattern is returned. If there are multiple such files, it's
        unspecified which one of them is returned.
        """

        self._start_requirement()

        requirement_desc = \
            f"dataset must contain a file matching pattern \"{pattern}\""

        if not self._is_path_within_root(pattern):
            self.fail(requirement_desc)

        pattern_abs = osp.join(glob.escape(self._root_path), pattern)
        for path in glob.iglob(pattern_abs, recursive=True):
            if osp.isfile(path):
                return osp.relpath(path, self._root_path)

        self.fail(requirement_desc)

    @contextlib.contextmanager
    def probe_text_file(
        self, path: str, requirement_desc: str,
    ) -> Iterator[TextIO]:
        """
        Returns a context manager that can be used to place a requirement on
        the contents of the file referred to by `path`. To do so, you must
        enter and exit this context manager (typically, by using the `with`
        statement). On entering, the file is opened for reading in text mode and
        the resulting file object is returned. On exiting, the file object is
        closed.

        The requirement that is placed by doing this is considered met if all
        of the following are true:

        * `path` is a relative path that refers to a file within the dataset
          root.
        * The file is opened successfully.
        * The context is exited without an exception.

        If the context is exited with an exception that was produced by another
        requirement being unmet, that exception is reraised and the new
        requirement is abandoned.

        `requirement_desc` must be a human-readable statement describing the
        requirement.
        """

        self._start_requirement()

        requirement_desc_full = f"{path}: {requirement_desc}"

        if not self._is_path_within_root(path):
            self.fail(requirement_desc_full)

        try:
            with open(osp.join(self._root_path, path), encoding='utf-8') as f:
                yield f
        except FormatRequirementsUnmet:
            raise
        except Exception:
            self.fail(requirement_desc_full)

    @contextlib.contextmanager
    def alternatives(self) -> Iterator[None]:
        """
        Returns a context manager that can be used to place a requirement that
        is considered met if at least one of several alternative sets of
        requirements is met.
        To do so, use a `with` statement, with the alternative sets of
        requirements represented as nested `with` statements using the context
        manager returned by `alternative`:

            with context.alternatives():
                with context.alternative():
                    # place requirements from alternative set 1 here
                with context.alternative():
                    # place requirements from alternative set 2 here
                ...

        The contents of all `with context.alternative()` blocks will be
        executed, even if an alternative that is met is found early.

        Requirements must not be placed directly within a
        `with context.alternatives()` block.
        """

        self._start_requirement()

        self._alternatives_context = self._AlternativesContext()

        try:
            yield

            # If at least one `alternative` block succeeded,
            # then the `alternatives` block succeeds.
            if self._alternatives_context.had_successful_alternatives:
                return

            # If no alternatives succeeded, and none failed, then there were
            # no alternatives at all.
            assert self._alternatives_context.failed_alternatives, \
                "an 'alternatives' block must contain " \
                "at least one 'alternative' block"

            raise FormatRequirementsUnmet(
                self._alternatives_context.failed_alternatives)
        finally:
            self._alternatives_context = None

    @contextlib.contextmanager
    def alternative(self) -> Iterator[None]:
        """
        Returns a context manager that can be used in combination with
        `alternatives` to define alternative requirements. See the documentation
        for `alternatives` for more details.

        Must only be used directly within a `with context.requirements()` block.
        """

        assert self._alternatives_context, \
            "An 'alternative' block must be directly within " \
            "an 'alternatives' block"

        saved_alternatives_context = self._alternatives_context
        self._alternatives_context = None

        try:
            yield
        except FormatRequirementsUnmet as e:
            saved_alternatives_context.failed_alternatives.extend(
                e.failed_alternatives)
        else:
            saved_alternatives_context.had_successful_alternatives = True
        finally:
            self._alternatives_context = saved_alternatives_context


FormatDetector = Callable[
    [FormatDetectionContext],
    Optional[FormatDetectionConfidence],
]
"""
Denotes a callback that implements detection for a specific dataset format.
The callback receives an instance of `FormatDetectionContext` and must call
methods on that instance to place requirements that the dataset must meet
in order for it to be considered as belonging to the format.

Must return the level of confidence in the dataset belonging to the format
(or `None`, which is equivalent to the `MEDIUM` level)
or terminate via a `FormatRequirementsUnmet` exception raised by one of
the `FormatDetectionContext` methods.
"""

def apply_format_detector(
    dataset_root_path: str, detector: FormatDetector,
) -> FormatDetectionConfidence:
    """
    Checks whether the dataset located at `dataset_root_path` belongs to the
    format detected by `detector`. If it does, returns the confidence level
    of the detection. Otherwise, raises a `FormatRequirementsUnmet` exception.
    """
    context = FormatDetectionContext(dataset_root_path)

    if not osp.isdir(dataset_root_path):
        context.fail(f"root path {dataset_root_path} must refer to a directory")

    return detector(context) or FormatDetectionConfidence.MEDIUM
