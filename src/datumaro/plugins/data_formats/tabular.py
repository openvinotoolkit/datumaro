# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Type, Union

from datumaro.components.annotation import AnnotationType, Categories, Tabular, TabularCategories
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Table, TableDtype, TableRow
from datumaro.util.os_util import find_files

# Only supports '.csv' extention.
TABULAR_EXTENSIONS = [
    "csv",
]


class TabularDataBase(DatasetBase):
    NAME = "tabular"

    def __init__(
        self,
        path: str,
        *,
        target: Optional[Union[str, List[str]]] = None,
        dtype: Optional[Dict[str, Type[TableDtype]]] = None,
        ctx: Optional[ImportContext] = None,
    ) -> None:
        """
        Read and compose a tabular dataset.
        The file name of each '.csv' file is regarded as subset.

        Args:
            path (str) : Path to a tabular dataset. (csv file or folder contains csv files).
            target (optional, str or list(str)) : Target column or list of target columns.
                If this is not specified (None), the last column is regarded as a target column.
                In case of a dataset with no targets, give an empty list as a parameter.
            dtype (optional, dict(str,str)) : Dictionay of column name -> type str ('str', 'int', or 'float').
                This can be used when automatic type inferencing is failed.
        """
        paths: List[str] = []
        if osp.isfile(path):
            paths.append(path)
        else:
            for path in find_files(path, TABULAR_EXTENSIONS):
                paths.append(path)

        if not paths:
            raise FileNotFoundError(errno.ENOENT, "Can't find tabular files", path)

        super().__init__(media_type=TableRow, ctx=ctx)

        self._infos = {"path": path}
        self._items, self._categories = self._parse(paths, target, dtype)

    def _parse(
        self,
        paths: List[str],
        target: Optional[Union[str, List[str]]] = None,
        dtype: Optional[Dict[str, Type[TableDtype]]] = None,
    ) -> Tuple[List[DatasetItem], Dict[AnnotationType, Categories]]:
        """
        parse tabular files. Each file is regarded as a subset.

        Args:
            paths (list(str)) : A list of paths to tabular data files(csv files).
            target (optional, str or list) : Target column or list of target columns.
                If this is not specified (None), the last column is regarded as a target column.
                In case of a dataset with no targets, give an empty list as a parameter.
            dtype (optional, dict(str,str)) : Dictionay of column name -> type str ('str', 'int', or 'float').
                This can be used when automatic type inferencing is failed.

        Returns:
            list (DatasetItem): dataset items
            dict (AnnotationType, Categories): categories info
        """
        # assert paths
        items: List[DatasetItem] = []
        categories: TabularCategories = TabularCategories()

        for path in paths:
            table = Table.from_csv(path, dtype=dtype)

            targets: List[str] = []
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
                target_dtype = table.dtype(target)
                if target_dtype == str:
                    labels = set(table.features(target, unique=True))
                    if category is None:
                        categories.add(target, target_dtype, labels)
                    else:  # update labels if they are different.
                        category.labels.union(labels)
                elif target_dtype in [int, float]:
                    # 'int' can be categorical, but we don't know this unless user gives information.
                    if category is None:
                        categories.add(target, target_dtype)
                else:
                    raise TypeError(
                        f"Unsupported type '{target_dtype}' for target column '{target}'."
                    )

            # load annotations
            subset = osp.splitext(osp.basename(path))[0]
            row: TableRow
            for row in table:  # type: TableRow
                id = f"{row.index}@{subset}"
                ann = [Tabular(values=row.data(targets))] if targets else None
                item = DatasetItem(
                    id=id,
                    subset=subset,
                    media=row,
                    annotations=ann,
                )
                items.append(item)

        return items, {AnnotationType.tabular: categories}

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items


class TabularDataImporter(Importer):
    """
    Import a tabular dataset.
    Each '.csv' file is regarded as a subset.
    """

    NAME = "tabular"

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
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
    def find_sources(cls, path):
        if not osp.isdir(path):
            ext = osp.splitext(path)[1][1:]  # exclude "."
            if ext in TABULAR_EXTENSIONS:
                return [{"url": path, "format": TabularDataBase.NAME}]
        else:
            for _ in find_files(path, TABULAR_EXTENSIONS):  # find 1 depth only.
                return [{"url": path, "format": TabularDataBase.NAME}]
        return []


class TabularDataExporter(Exporter):
    """
    Export a tabular dataset.
    This will save each subset into a '.csv' file regardless of 'save_media' value
    """

    NAME = "tabular"
    EXPORT_EXT = ".csv"
    DEFAULT_IMAGE_EXT = ".jpg"  # just to avoid assert error.

    def _apply_impl(self):
        extractor = self._extractor

        if extractor.media_type() and not issubclass(extractor.media_type(), TableRow):
            raise MediaTypeError("Media type is not a table.")

        # we don't check self._save_media.
        # regardless of the value, we always save media(csv) file.

        os.makedirs(self._save_dir, exist_ok=True)

        for sname in extractor.subsets():
            subset = extractor.get_subset(sname)
            path = osp.join(self._save_dir, sname + self.EXPORT_EXT)
            list_of_dicts: List[Dict[str, TableDtype]] = list()
            for item in subset:
                dicts = item.media.data()
                for ann in item.annotations:
                    if isinstance(ann, Tabular):
                        dicts.update(ann.values)  # update value
                list_of_dicts.append(dicts)

            table = Table.from_list(list_of_dicts)
            table.save(path)
