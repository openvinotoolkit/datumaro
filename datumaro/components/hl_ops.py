# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Dict, List, Optional, Union
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

# lazy, not inplace

def transform(dataset: IDataset, method: Union[str, Transform], *,
        env: Optional[Environment] = None, **kwargs) -> IDataset:
    if isinstance(method, str):
        if env is None:
            env = Environment()
        method = env.transforms[method]
    produced = method(dataset, **kwargs)
    return Dataset.from_extractors(produced, env=env)

def filter(dataset: IDataset, expr: str, *, #pylint: disable=redefined-builtin
        filter_annotations: bool = False,
        remove_empty: bool = False) -> IDataset:
    if filter_annotations:
        return transform(dataset, XPathAnnotationsFilter,
            xpath=expr, remove_empty=remove_empty)
    else:
        if not expr:
            return dataset
        return transform(dataset, XPathDatasetFilter, xpath=expr)

def merge(*datasets: IDataset) -> IDataset:
    categories = ExactMerge.merge_categories(d.categories() for d in datasets)
    return DatasetItemStorageDatasetView(ExactMerge.merge(*datasets),
        categories=categories)

def run_model(dataset: IDataset, model: Union[ModelTransform, Launcher], *,
        batch_size: int = 1, **kwargs) -> IDataset:
    if isinstance(model, Launcher):
        return transform(dataset, ModelTransform, launcher=model,
            batch_size=batch_size, **kwargs)
    elif isinstance(model, ModelTransform):
        return transform(model, batch_size=batch_size, **kwargs)
    else:
        raise TypeError('Unexpected model argument type: %s' % type(model))

# other operations

@scoped
def export(dataset: IDataset, path: str, format: Union[str, Converter], *,
        env: Optional[Environment] = None, **kwargs):
    if isinstance(format, str):
        if env is None:
            env = Environment()
        converter = env.converters[format]
    else:
        converter = format

    path = osp.abspath(path)
    if not osp.exists(path):
        on_error_do(shutil.rmtree, path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

    converter.convert(dataset, save_dir=path, **kwargs)

def detect(path: str, env: Optional[Environment] = None) -> List[str]:
    if env is None:
        env = Environment()
    return env.detect_dataset(path)

def validate(dataset: IDataset, domain: Union[str, TaskType], *,
        env: Optional[Environment] = None, **kwargs) -> Dict:
    if env is None:
        env = Environment()
    domain = parse_str_enum_value(domain, TaskType).name.lower()
    validator: Validator = env.validators[domain](**kwargs)
    return validator.validate(dataset)
