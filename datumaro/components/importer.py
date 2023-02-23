# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import os.path as osp
from glob import iglob
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, TypeVar

import attr
from attr import define, field

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.errors import (
    AnnotationImportError,
    DatasetNotFoundError,
    DatumaroError,
    ItemImportError,
)
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.progress_reporting import NullProgressReporter, ProgressReporter

T = TypeVar("T")


class _ImportFail(DatumaroError):
    pass


class ImportErrorPolicy:
    def report_item_error(self, error: Exception, *, item_id: Tuple[str, str]) -> None:
        """
        Allows to report a problem with a dataset item.
        If this function returns, the extractor must skip the item.
        """

        if not isinstance(error, _ImportFail):
            ie = ItemImportError(item_id)
            ie.__cause__ = error
            return self._handle_item_error(ie)
        else:
            raise error

    def report_annotation_error(self, error: Exception, *, item_id: Tuple[str, str]) -> None:
        """
        Allows to report a problem with a dataset item annotation.
        If this function returns, the extractor must skip the annotation.
        """

        if not isinstance(error, _ImportFail):
            ie = AnnotationImportError(item_id)
            ie.__cause__ = error
            return self._handle_annotation_error(ie)
        else:
            raise error

    def _handle_item_error(self, error: ItemImportError) -> None:
        """This function must either call fail() or return."""
        self.fail(error)

    def _handle_annotation_error(self, error: AnnotationImportError) -> None:
        """This function must either call fail() or return."""
        self.fail(error)

    def fail(self, error: Exception) -> NoReturn:
        raise _ImportFail from error


class FailingImportErrorPolicy(ImportErrorPolicy):
    pass


@define(eq=False)
class ImportContext:
    progress_reporter: ProgressReporter = field(
        default=None, converter=attr.converters.default_if_none(factory=NullProgressReporter)
    )
    error_policy: ImportErrorPolicy = field(
        default=None, converter=attr.converters.default_if_none(factory=FailingImportErrorPolicy)
    )


class NullImportContext(ImportContext):
    pass


class Importer(CliPlugin):
    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        if not cls.find_sources_with_params(context.root_path):
            context.fail("specific requirement information unavailable")

        return FormatDetectionConfidence.LOW

    @classmethod
    def find_sources(cls, path: str) -> List[Dict]:
        raise NotImplementedError()

    @classmethod
    def find_sources_with_params(cls, path, **extra_params) -> List[Dict]:
        return cls.find_sources(path)

    def __call__(self, path, **extra_params):
        if not path or not osp.exists(path):
            raise DatasetNotFoundError(path)

        found_sources = self.find_sources_with_params(osp.normpath(path), **extra_params)
        if not found_sources:
            raise DatasetNotFoundError(path)

        sources = []
        for desc in found_sources:
            params = dict(extra_params)
            params.update(desc.get("options", {}))
            desc["options"] = params
            sources.append(desc)

        return sources

    @classmethod
    def _find_sources_recursive(
        cls,
        path: str,
        ext: Optional[str],
        extractor_name: str,
        filename: str = "*",
        dirname: str = "",
        file_filter: Optional[Callable[[str], bool]] = None,
        max_depth: int = 3,
    ):
        """
        Finds sources in the specified location, using the matching pattern
        to filter file names and directories.
        Supposed to be used, and to be the only call in subclasses.

        Parameters:
            path: a directory or file path, where sources need to be found.
            ext: file extension to match. To match directories,
                set this parameter to None or ''. Comparison is case-independent,
                a starting dot is not required.
            extractor_name: the name of the associated Extractor type
            filename: a glob pattern for file names
            dirname: a glob pattern for filename prefixes
            file_filter: a callable (abspath: str) -> bool, to filter paths found
            max_depth: the maximum depth for recursive search.

        Returns: a list of source configurations
            (i.e. Extractor type names and c-tor parameters)
        """

        if ext:
            if not ext.startswith("."):
                ext = "." + ext
            ext = ext.lower()

        if (path.lower().endswith(ext) and osp.isfile(path)) or (
            not ext
            and dirname
            and osp.isdir(path)
            and os.sep + osp.normpath(dirname.lower()) + os.sep
            in osp.abspath(path.lower()) + os.sep
        ):
            sources = [{"url": path, "format": extractor_name}]
        else:
            sources = []
            for d in range(max_depth + 1):
                sources.extend(
                    {"url": p, "format": extractor_name}
                    for p in iglob(osp.join(path, *("*" * d), dirname, filename + ext))
                    if (callable(file_filter) and file_filter(p)) or (not callable(file_filter))
                )
                if sources:
                    break
        return sources

    @property
    def extractor_name(cls) -> str:
        return cls.NAME.replace("_importer", "")
