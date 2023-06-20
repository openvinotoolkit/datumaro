# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.cli_plugin import CliPlugin


# pylint: disable=no-self-use
class Launcher(CliPlugin):
    def __init__(self, model_dir=None):
        pass

    def launch(self, inputs):
        raise NotImplementedError()

    def infos(self):
        return None

    def categories(self):
        return None

    def type_check(self, item):
        return True
