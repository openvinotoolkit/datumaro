# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import inspect
import os
import os.path as osp
import shutil
from typing import Dict, Iterable, Optional, Type, Union

from datumaro.cli.util.compare import DistanceCompareVisualizer
from datumaro.cli.util.project import generate_next_file_name
from datumaro.components.comparator import DistanceComparator, EqualityComparator, TableComparator
from datumaro.components.dataset import Dataset, DatasetItemStorageDatasetView, IDataset
from datumaro.components.environment import Environment
from datumaro.components.errors import DatasetError
from datumaro.components.exporter import Exporter
from datumaro.components.filter import XPathAnnotationsFilter, XPathDatasetFilter
from datumaro.components.launcher import Launcher
from datumaro.components.merge import DEFAULT_MERGE_POLICY, get_merger
from datumaro.components.transformer import ModelTransform, Transform
from datumaro.components.validator import TaskType, Validator
from datumaro.util import dump_json_file, parse_str_enum_value
from datumaro.util.scope import on_error_do, scoped

__all__ = ["HLOps"]


class HLOps:
    """High-level dataset operations for Python API."""

    @staticmethod
    def compare(
        first_dataset: IDataset,
        second_dataset: IDataset,
        report_dir: Optional[str] = None,
        method: str = "table",
        **kwargs,
    ) -> IDataset:
        """
        Compare two datasets and optionally save a comparison report.

        Args:
            first_dataset (IDataset): The first dataset to compare.
            second_dataset (IDataset): The second dataset to compare.
            report_dir (Optional[str], optional): The directory path to save the comparison report. Defaults to None.
            method (str, optional): The comparison method to use. Possible values are "table", "equality", "distance".
                Defaults to "table".
            **kwargs: Additional keyword arguments that can be passed to the comparison method.

        Returns:
            IDataset: The result of the comparison.

        Raises:
            ValueError: If the method is "distance" and report_dir is not specified.

        Example:
            comparator = Comparator()
            result = comparator.compare(first_dataset, second_dataset, report_dir="./comparison_report")
            print(result)
        """
        if method == "table":
            comparator = TableComparator()
            h_table, m_table, l_table, result_dict = comparator.compare_datasets(
                first_dataset, second_dataset
            )
            if report_dir:
                comparator.save_compare_report(h_table, m_table, l_table, result_dict, report_dir)

        elif method == "equality":
            comparator = EqualityComparator(**kwargs)
            output = comparator.compare_datasets(first_dataset, second_dataset)
            if report_dir:
                comparator.save_compare_report(output, report_dir)

        elif method == "distance":
            if not report_dir:
                raise ValueError(
                    "Please specify report_dir to save comparision result for DistanceComparator."
                )
            output_format = kwargs.pop("output_format", "simple")
            comparator = DistanceComparator(**kwargs)
            with DistanceCompareVisualizer(
                save_dir=report_dir,
                comparator=comparator,
                output_format=output_format,
            ) as visualizer:
                visualizer.save(first_dataset, second_dataset)

        return 0

    @staticmethod
    def transform(
        dataset: IDataset,
        method: Union[str, Type[Transform]],
        *,
        env: Optional[Environment] = None,
        **kwargs,
    ) -> IDataset:
        """
        Applies some function to dataset items.

        Results are computed lazily, if the transform supports this.

        Args:
            dataset: The dataset to be transformed
            method: The transformation to be applied to the dataset.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the environment
                set by the 'env' argument
            env: A plugin collection. If not set, the built-in plugins are used
            **kwargs: Parameters for the transformation

        Returns: a wrapper around the input dataset
        """

        if isinstance(method, str):
            if env is None:
                env = Environment()
            method = env.transforms[method]

        if not (inspect.isclass(method) and issubclass(method, Transform)):
            raise TypeError(f"Unexpected 'method' argument type: {type(method)}")

        produced = method(dataset, **kwargs)

        return Dataset(source=produced, env=env)

    @staticmethod
    def filter(
        dataset: IDataset,
        expr: str,
        *,  # pylint: disable=redefined-builtin
        filter_annotations: bool = False,
        remove_empty: bool = False,
    ) -> IDataset:
        """
        Filters out some dataset items or annotations, using a custom filter
        expression.

        Args:
            dataset: The dataset to be filtered
            expr: XPath-formatted filter expression
                (e.g. `/item[subset = 'train']`, `/item/annotation[label = 'cat']`)
            filter_annotations: Indicates if the filter should be
                applied to items or annotations
            remove_empty: When filtering annotations, allows to
                exclude empty items from the resulting dataset

        Returns: a wrapper around the input dataset, which is computed lazily
            during iteration
        """

        if filter_annotations:
            return HLOps.transform(
                dataset, XPathAnnotationsFilter, xpath=expr, remove_empty=remove_empty
            )
        else:
            if not expr:
                return dataset
            return HLOps.transform(dataset, XPathDatasetFilter, xpath=expr)

    @staticmethod
    def merge(
        *datasets: Dataset,
        merge_policy: str = DEFAULT_MERGE_POLICY,
        report_path: Optional[str] = None,
        **kwargs,
    ) -> Dataset:
        """
        Merge `datasets` according to `merge_policy`. You have to choose an appropriate `merge_policy`
        for your purpose. The available merge policies are "union", "intersect", and "exact".
        For more details about the merge policies, please refer to :func:`get_merger`.
        """

        merger = get_merger(merge_policy, **kwargs)
        merged = merger(*datasets)
        env = Environment.merge(
            dataset.env
            for dataset in datasets
            if hasattr(
                dataset, "env"
            )  # TODO: Sometimes, there is dataset which is not exactly "Dataset",
            # e.g., VocClassificationBase. this should be fixed and every object from
            # Dataset.import_from should have "Dataset" type.
        )
        if report_path:
            merger.save_merge_report(report_path)
        return Dataset(source=merged, env=env)

    @staticmethod
    def run_model(
        dataset: IDataset,
        model: Union[Launcher, Type[ModelTransform]],
        *,
        batch_size: int = 1,
        append_annotation: bool = False,
        num_workers: int = 0,
        **kwargs,
    ) -> IDataset:
        """
        Run the model on the dataset item media entities, such as images,
        to obtain pseudo labels and add them as dataset annotations.

        Args:
            dataset: The dataset to be transformed
            model: The model to be applied to the dataset
            batch_size: The number of dataset items processed
                simultaneously by the model
            append_annotation: Whether append new annotation to existed annotations
            num_workers: The number of worker threads to use for parallel inference.
                Set to 0 for single-process mode. Default is 0.
            **kwargs: Parameters for the model

        Returns: a wrapper around the input dataset, which is computed lazily
            during iteration
        """

        if isinstance(model, Launcher):
            return HLOps.transform(
                dataset,
                ModelTransform,
                launcher=model,
                batch_size=batch_size,
                append_annotation=append_annotation,
                num_workers=num_workers,
                **kwargs,
            )
        elif inspect.isclass(model) and issubclass(model, ModelTransform):
            return HLOps.transform(
                dataset,
                model,
                batch_size=batch_size,
                append_annotation=append_annotation,
                num_workers=num_workers,
                **kwargs,
            )
        else:
            raise TypeError(f"Unexpected model argument type: {type(model)}")

    @staticmethod
    @scoped
    def export(
        dataset: IDataset,
        path: str,
        format: Union[str, Type[Exporter]],
        *,
        env: Optional[Environment] = None,
        **kwargs,
    ) -> None:
        """
        Saves the input dataset in some format.

        Args:
            dataset: The dataset to be saved
            path: The output directory
            format: The desired output format for the dataset.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the environment set by the 'env' argument
            env: A plugin collection. If not set, the built-in plugins are used
            **kwargs: Parameters for the export format
        """

        if isinstance(format, str):
            if env is None:
                env = Environment()
            exporter = env.exporters[format]
        else:
            exporter = format

        if not (inspect.isclass(exporter) and issubclass(exporter, Exporter)):
            raise TypeError(f"Unexpected 'format' argument type: {type(exporter)}")

        path = osp.abspath(path)
        if not osp.exists(path):
            on_error_do(shutil.rmtree, path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

        exporter.convert(dataset, save_dir=path, **kwargs)

    @staticmethod
    def validate(
        dataset: IDataset,
        task: Union[str, TaskType],
        *,
        env: Optional[Environment] = None,
        **kwargs,
    ) -> Dict:
        """
        Checks dataset annotations for correctness relatively to a task type.

        Args:
            dataset: The dataset to check
            task: Target task type - classification, detection etc.
            env: A plugin collection. If not set, the built-in plugins are used
            **kwargs: Parameters for the validator

        Returns: a dictionary with validation results
        """

        task = parse_str_enum_value(task, TaskType).name

        if env is None:
            env = Environment()

        validator: Validator = env.validators[task](**kwargs)
        return validator.validate(dataset)

    @staticmethod
    def aggregate(dataset: Dataset, from_subsets: Iterable[str], to_subset: str) -> Dataset:
        subset_names = set(dataset.subsets().keys())

        for subset in from_subsets:
            if subset not in subset_names:
                raise DatasetError(
                    f"{subset} is not found in the subset names ({subset_names}) in the dataset."
                )

        return HLOps.transform(
            dataset, "map_subsets", mapping={subset: to_subset for subset in from_subsets}
        )
