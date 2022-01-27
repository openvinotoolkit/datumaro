# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Dict, Optional, Type, Union
import inspect
import os
import os.path as osp
import shutil

from datumaro.components.converter import Converter
from datumaro.components.dataset import (
    Dataset, DatasetItemStorageDatasetView, IDataset,
)
from datumaro.components.dataset_filter import (
    XPathAnnotationsFilter, XPathDatasetFilter,
)
from datumaro.components.environment import Environment
from datumaro.components.extractor import Transform
from datumaro.components.launcher import Launcher, ModelTransform
from datumaro.components.operations import ExactMerge
from datumaro.components.validator import TaskType, Validator
from datumaro.util import parse_str_enum_value
from datumaro.util.scope import on_error_do, scoped


def transform(dataset: IDataset, method: Union[str, Type[Transform]], *,
        env: Optional[Environment] = None, **kwargs) -> IDataset:
    """
    Applies some function to dataset items.

    Results are computed lazily, if the transform supports this.

    Args:
        dataset - The dataset to be transformed
        method - The transformation to be applied to the dataset.
            If a string is passed, it is treated as a plugin name,
            which is searched for in the environment
            set by the 'env' argument
        env - A plugin collection. If not set, the built-in plugins are used
        **kwargs - Parameters for the transformation

    Returns: a wrapper around the input dataset
    """

    if isinstance(method, str):
        if env is None:
            env = Environment()
        method = env.transforms[method]

    if not (inspect.isclass(method) and issubclass(method, Transform)):
        raise TypeError("Unexpected 'method' argument type: %s" % \
            type(method))

    produced = method(dataset, **kwargs)

    return Dataset(source=produced, env=env)

def filter(dataset: IDataset, expr: str, *, #pylint: disable=redefined-builtin
        filter_annotations: bool = False,
        remove_empty: bool = False) -> IDataset:
    """
    Filters out some dataset items or annotations, using a custom filter
    expression.

    Args:
        dataset - The dataset to be filtered
        expr - XPath-formatted filter expression
            (e.g. `/item[subset = 'train']`, `/item/annotation[label = 'cat']`)
        filter_annotations - Indicates if the filter should be
            applied to items or annotations
        remove_empty - When filtering annotations, allows to
            exclude empty items from the resulting dataset

    Returns: a wrapper around the input dataset, which is computed lazily
        during iteration
    """

    if filter_annotations:
        return transform(dataset, XPathAnnotationsFilter,
            xpath=expr, remove_empty=remove_empty)
    else:
        if not expr:
            return dataset
        return transform(dataset, XPathDatasetFilter, xpath=expr)

def merge(*datasets: IDataset) -> IDataset:
    """
    Merges several datasets using the "simple" (exact matching) algorithm:
    - items are matched by (id, subset) pairs
    - matching items share the fields available
        - nothing + nothing = nothing,
        - nothing + something = something
        - something A + something B = conflict
    - annotations are matched by value and shared
    - in case of conflicts, throws an error

    Returns: a wrapper around the input datasets
    """

    categories = ExactMerge.merge_categories(d.categories() for d in datasets)
    return DatasetItemStorageDatasetView(ExactMerge.merge(*datasets),
        categories=categories)

def run_model(dataset: IDataset,
        model: Union[Launcher, Type[ModelTransform]], *,
        batch_size: int = 1, **kwargs) -> IDataset:
    """
    Applies a model to dataset items' media and produces a dataset with
    media and annotations.

    Args:
        dataset - The dataset to be transformed
        model - The model to be applied to the dataset
        batch_size - The number of dataset items processed
            simultaneously by the model
        **kwargs - Parameters for the model

    Returns: a wrapper around the input dataset, which is computed lazily
        during iteration
    """

    if isinstance(model, Launcher):
        return transform(dataset, ModelTransform, launcher=model,
            batch_size=batch_size, **kwargs)
    elif inspect.isclass(model) and issubclass(model, ModelTransform):
        return transform(dataset, model,
            batch_size=batch_size, **kwargs)
    else:
        raise TypeError('Unexpected model argument type: %s' % type(model))

@scoped
def export(dataset: IDataset, path: str,
        format: Union[str, Type[Converter]], *,
        env: Optional[Environment] = None, **kwargs) -> None:
    """
    Saves the input dataset in some format.

    Args:
        dataset - The dataset to be saved
        path - The output directory
        format - The desired output format for the dataset.
            If a string is passed, it is treated as a plugin name,
            which is searched for in the environment set by the 'env' argument
        env - A plugin collection. If not set, the built-in plugins are used
        **kwargs - Parameters for the export format
    """

    if isinstance(format, str):
        if env is None:
            env = Environment()
        converter = env.converters[format]
    else:
        converter = format

    if not (inspect.isclass(converter) and issubclass(converter, Converter)):
        raise TypeError("Unexpected 'format' argument type: %s" % \
            type(converter))

    path = osp.abspath(path)
    if not osp.exists(path):
        on_error_do(shutil.rmtree, path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

    converter.convert(dataset, save_dir=path, **kwargs)

def validate(dataset: IDataset, task: Union[str, TaskType], *,
        env: Optional[Environment] = None, **kwargs) -> Dict:
    """
    Checks dataset annotations for correctness relatively to a task type.

    Args:
        dataset - The dataset to check
        task - Target task type - classification, detection etc.
        env - A plugin collection. If not set, the built-in plugins are used
        **kwargs - Parameters for the validator

    Returns: a dictionary with validation results
    """

    task = parse_str_enum_value(task, TaskType).name

    if env is None:
        env = Environment()

    validator: Validator = env.validators[task](**kwargs)
    return validator.validate(dataset)
