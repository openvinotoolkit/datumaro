# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os.path as osp
from typing import Dict, List, Optional, Set, Tuple, Type, Union

from datumaro.components.annotation import AnnotationType, Categories, Tabular, TabularCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Table, TableDtype, TableRow
from datumaro.util.os_util import find_files

# Only supports '.csv' extention.
TABULAR_EXTENSIONS = [
    "csv",
]


class TabularDataBase(SubsetBase):
    NAME = "tabular"
    """
    Compose a tabular dataset from a file.
    """

    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
        dtype: Optional[Dict[str, Type[TableDtype]]] = None,
        ctx: Optional[ImportContext] = None,
    ) -> None:
        """
        Read a tabular data file and compose dataset.

        Args:
            path (str) : Path to a tabular data file.
            subset (optional, str) : Subset name for the dataset.
            target (optional, str or list) : Target column or list of target columns.
                If this is not specified (None), the last column is regarded as a target column.
                In case of a dataset with no targets, give an empty list as a parameter.
            dtype (optional, dict (str: str)) : Dictionay of column name -> type (str, int, or float).
                This can be used when automatic type inferencing is failed.
        """
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find CSV file", path)

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]

        super().__init__(subset=subset, media_type=TableRow, ctx=ctx)

        self._infos = {"path": path}
        self._items, self._categories = self._parse(path, target, dtype)
        self._length = len(self._items)

    def _parse(
        self,
        path: str,
        target: Optional[Union[str, List[str]]] = None,
        dtype: Optional[Dict[str, Type[TableDtype]]] = None,
    ) -> Tuple[List[DatasetItem], Dict[AnnotationType, Categories]]:
        """
        parse a csv file.

        Args:
            path (str) : Path to a tabular data file.
            target (optional, str or list) : Target column or list of target columns.
                If this is not specified (None), the last column is regarded as a target column.
                In case of a dataset with no targets, give an empty list as a parameter.
            dtype (optional, dict (str: type)) : Dictionay of column name -> type (str, int, or float).
                This can be used when automatic type inferencing is failed.

        Returns:
            list (DatasetItem): dataset items
            dict (AnnotationType: Categories): categories info
        """
        # assert paths
        items: List[DatasetItem] = list()
        categories: TabularCategories = TabularCategories()

        # for path in paths:
        table = Table.from_csv(path, dtype=dtype)

        targets: List[str] = list()
        if target is None:
            targets.append(table.columns[-1])  # last column
        elif isinstance(target, str):
            if target in table.columns:  # add valid column name only
                targets.append(target)
        elif isinstance(target, list):  # add valid column names only
            for t in target:
                if t in table.columns:
                    targets.append(t)

        # set categories
        for target in targets:
            _, category = categories.find(target)
            dtype = table.dtype(target)
            if dtype == str:
                labels = Set(table.features(target, unique=True))
                if category is None:
                    categories.add(target, dtype, labels)
                else:  # update labels if they are different.
                    category.labels.union(labels)
            elif dtype in [int, float]:
                if category is None:
                    categories.add(target, dtype)
            else:
                raise TypeError(f"Unsupported type '{dtype}' for target column '{target}'.")

        # load annotations
        row: TableRow
        for row in table:  # type: TableRow
            values: Dict[str, Union[str, int, float]] = row.data(targets)
            id = f"{row.index}@{self._subset}"
            ann = Tabular(values=values)
            item = DatasetItem(id=id, subset=self._subset, media=row, annotations=[ann])
            items.append(item)

        return items, {AnnotationType.tabular: categories}

    def __iter__(self):
        yield from self._items


class TabularDataImporter(Importer):
    NAME = "tabular"

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--subset",
            help="The name of the subset for the produced dataset items " "(default: None)",
        )
        parser.add_argument(
            "--target",
            type=lambda x: x.split(","),
            help="Target column or list of target columns. (ex. 'class', 'class,breed') (default:None) "
            "If this is not specified (None), the last column is regarded as a target column."
            "In case of a dataset with no targets, give an empty list as a parameter.",
        )
        parser.add_argument(
            "--dtype",
            type=lambda x: {k: v for k, v in (map.split(":") for map in x.split(","))},
            help="Type information for a column. (ex. 'date:str,x:int') (default:None) "
            "This can be used when automatic type inferencing is failed",
        )
        return parser

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> Optional[FormatDetectionConfidence]:
        try:
            next(find_files(context.root_path, TABULAR_EXTENSIONS, recursive=True))
            return FormatDetectionConfidence.LOW
        except StopIteration:
            context.fail(
                "No tabular files found in '%s'. "
                "Checked extensions: %s" % (context.root_path, ", ".join(TABULAR_EXTENSIONS))
            )

    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            ext = osp.splitext(path)[1][1:]  # exclude "."
            if ext in TABULAR_EXTENSIONS:
                return [{"url": path, "format": TabularDataBase.NAME}]
            else:
                return []
        else:
            sources = []
            for fname in find_files(path, TABULAR_EXTENSIONS, recursive=True):
                sources.append({"url": fname, "format": TabularDataBase.NAME})
            return sources
