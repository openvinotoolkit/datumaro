# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import importlib.util
import sys
from types import ModuleType


def lazy_import(name: str) -> ModuleType:
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module
