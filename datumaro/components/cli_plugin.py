# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.cli.util import MultilineFormatter
from datumaro.util import to_snake_case


class CliPlugin:
    @staticmethod
    def _get_name(cls):
        return getattr(cls, 'NAME',
            remove_plugin_type(to_snake_case(cls.__name__)))

    @staticmethod
    def _get_doc(cls):
        doc = getattr(cls, '__doc__', "")
        if doc:
            from datumaro.components.converter import Converter
            from datumaro.components.extractor import (
                Extractor, Importer, Transform,
            )
            from datumaro.components.launcher import Launcher
            base_classes = [Launcher, Extractor, Transform, Importer, Converter]

            if any(getattr(t, '__doc__', '') == doc for t in base_classes):
                doc = ''
        return doc

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
        args = vars(args)

        log.debug("Parsed parameters: \n\t%s",
            '\n\t'.join('%s: %s' % (k, v) for k, v in args.items()))

        return args

def remove_plugin_type(s):
    for t in {'transform', 'extractor', 'converter', 'launcher', 'importer',
            'validator'}:
        s = s.replace('_' + t, '')
    return s
