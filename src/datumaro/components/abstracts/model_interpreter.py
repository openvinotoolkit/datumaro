# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod

__all__ = ["IModelInterpreter"]


class IModelInterpreter(ABC):
    @abstractmethod
    def get_categories(self):
        raise NotImplementedError("Function should be implemented.")

    @abstractmethod
    def process_outputs(self, inputs, outputs):
        raise NotImplementedError("Function should be implemented.")

    @abstractmethod
    def normalize(self, inputs):
        raise NotImplementedError("Function should be implemented.")

    @abstractmethod
    def resize(self, inputs):
        raise NotImplementedError("Function should be implemented.")
