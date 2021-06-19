# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto
from typing import Dict, List

from datumaro.components.dataset import IDataset


class Severity(Enum):
    warning = auto()
    error = auto()


class TaskType(Enum):
    classification = auto()
    detection = auto()
    segmentation = auto()


class IValidator:
    def validate(self, dataset: IDataset) -> Dict:
        raise NotImplementedError()


class Validator(IValidator):
    def validate(self, dataset: IDataset) -> Dict:
        raise NotImplementedError()

    def compute_statistics(self, dataset: IDataset) -> Dict:
        raise NotImplementedError()

    def generate_reports(self, stats: Dict) -> List[Dict]:
        raise NotImplementedError()
