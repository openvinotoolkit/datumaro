
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from datumaro.cli.util import MultilineFormatter
from datumaro.util import to_snake_case


class CliPlugin:
    @staticmethod
    def _get_name(cls):
        return getattr(cls, 'NAME',
            remove_plugin_type(to_snake_case(cls.__name__)))

    @staticmethod
    def _get_doc(cls):
        return getattr(cls, '__doc__', "")

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        args = {
            'prog': cls._get_name(cls),
            'description': cls._get_doc(cls),
            'formatter_class': MultilineFormatter,
        }
        args.update(kwargs)

        return argparse.ArgumentParser(**args)

    @classmethod
    def parse_cmdline(cls, args=None):
        if args and args[0] == '--':
            args = args[1:]
        parser = cls.build_cmdline_parser()
        args = parser.parse_args(args)
        return vars(args)

def remove_plugin_type(s):
    for t in {'transform', 'extractor', 'converter', 'launcher', 'importer'}:
        s = s.replace('_' + t, '')
    return s
