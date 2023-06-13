# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.abstracts import IModelInterpreter

__all__ = ["OTXAtssModelInterpreter"]


class OTXAtssModelInterpreter(IModelInterpreter):
    def get_categories(self):
        return None

    def process_outputs(self, inputs, outputs):
        return outputs

    def normalize(self, inputs):
        return inputs

    def resize(self, inputs):
        return inputs
