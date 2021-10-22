# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import IntEnum
from typing import Callable, Optional, Sequence
import glob
import os.path as osp


class FormatDetectionConfidence(IntEnum):
    """
    Represents the level of confidence that a detector has in a dataset
    belonging to the detector's format.
    """

    # Note to developers: more confidence levels could be added in the future,
    # but they should all have positive values. This is to ensure that 0
    # can be used as a sentinel value for comparing confidence levels.
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

    # Note: it's currently impossible for an exception with more than one
    # alternative to be raised; but that will change once support for
    # alternative requirements in FormatDetectionContext is implemented.
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

    def __init__(self, root_path: str) -> None:
        self._root_path = root_path

    @property
    def root_path(self) -> str:
        """
        Returns the path to the root directory of the dataset.
        Detectors should avoid using this property in favor of specific
        requirement methods.
        """
        return self._root_path

    def fail(self, requirement: str) -> str:
        """
        Places a requirement that is never met. `requirement` must contain
        a human-readable description of the requirement.
        """
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

        requirement_str = \
            f"dataset must contain a file matching pattern \"{pattern}\""

        # These pattern checks raise a FormatRequirementsUnmet rather than an
        # AssertionError, because the detector might have gotten the pattern
        # from another file in the dataset. In that case, an invalid pattern
        # signifies a problem with the dataset, rather than with the detector.
        if osp.isabs(pattern) or osp.splitdrive(pattern)[0]:
            self.fail(requirement_str)

        pattern = osp.normpath(pattern)
        if pattern.startswith('..' + osp.sep):
            self.fail(requirement_str)

        pattern_abs = osp.join(glob.escape(self._root_path), pattern)
        for path in glob.iglob(pattern_abs, recursive=True):
            if osp.isfile(path):
                return osp.relpath(path, self._root_path)

        self.fail(requirement_str)

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

    if not osp.exists(dataset_root_path):
        context.fail(f"root path {dataset_root_path} must exist")

    return detector(context) or FormatDetectionConfidence.MEDIUM
