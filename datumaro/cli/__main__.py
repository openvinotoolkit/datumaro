
# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import sys

from . import contexts, commands
from .util import CliException, add_subparser
from ..version import VERSION


_log_levels = {
    'debug': log.DEBUG,
    'info': log.INFO,
    'warning': log.WARNING,
    'error': log.ERROR,
    'critical': log.CRITICAL
}

def loglevel(name):
    return _log_levels[name]

class _LogManager:
    @classmethod
    def init_logger(cls, args=None):
        # Define minimalistic parser only to obtain loglevel
        parser = argparse.ArgumentParser(add_help=False)
        cls._define_loglevel_option(parser)
        args, _ = parser.parse_known_args(args)

        log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
            level=args.loglevel)

    @staticmethod
    def _define_loglevel_option(parser):
        parser.add_argument('--loglevel', type=loglevel, default='info',
            help="Logging level (options: %s; default: %s)" % \
                (', '.join(_log_levels.keys()), "%(default)s"))
        return parser


def _make_subcommands_help(commands, help_line_start=0):
    desc = ""
    for command_name, _, command_help in commands:
        desc += ("  %-" + str(max(0, help_line_start - 2 - 1)) + "s%s\n") % \
            (command_name, command_help)
    return desc

def make_parser():
    parser = argparse.ArgumentParser(prog="datumaro",
        description="Dataset Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--version', action='version', version=VERSION)
    _LogManager._define_loglevel_option(parser)

    known_contexts = [
        ('project', contexts.project, "Actions with project"),
        ('source', contexts.source, "Actions with data sources"),
        ('model', contexts.model, "Actions with models"),
    ]
    known_commands = [
        ("Project modification:", None, ''),
        ('create', commands.create, "Create empty project"),
        ('add', commands.add, "Add data source to project"),
        ('remove', commands.remove, "Remove data source from project"),

        ("", None, ''),
        ("Project versioning:", None, ''),
        ('checkout', commands.checkout, "Switch to another branch or revision"),
        ('commit', commands.commit, "Commit changes in tracked files"),

        ("", None, ''),
        ("Dataset and project operations:", None, ''),
        ('export', commands.export, "Export project in some format"),
        ('filter', commands.filter, "Filter project items"),
        ('transform', commands.transform, "Modify project items"),
        ('merge', commands.merge, "Merge projects"),
        ('convert', commands.convert, "Convert dataset between formats"),
        ('diff', commands.diff, "Compare projects with intersection"),
        ('ediff', commands.ediff, "Compare projects for equality"),
        ('stats', commands.stats, "Compute project statistics"),
        ('info', commands.info, "Print project info"),
        ('explain', commands.explain, "Run Explainable AI algorithm for model"),
        ('validate', commands.validate, "Validate project")
    ]

    # Argparse doesn't support subparser groups:
    # https://stackoverflow.com/questions/32017020/grouping-argparse-subparser-arguments
    help_line_start = max((len(e[0]) for e in known_contexts + known_commands),
        default=0)
    help_line_start = max((2 + help_line_start) // 4 + 1, 6) * 4 # align to tabs
    subcommands_desc = ""
    if known_contexts:
        subcommands_desc += "Contexts:\n"
        subcommands_desc += _make_subcommands_help(known_contexts,
            help_line_start)
    if known_commands:
        if subcommands_desc:
            subcommands_desc += "\n"
        subcommands_desc += "Commands:\n"
        subcommands_desc += _make_subcommands_help(known_commands,
            help_line_start)
    if subcommands_desc:
        subcommands_desc += \
            "\nRun '%s COMMAND --help' for more information on a command." % \
                parser.prog

    subcommands = parser.add_subparsers(title=subcommands_desc,
        description="", help=argparse.SUPPRESS)
    for command_name, command, _ in known_contexts + known_commands:
        if command is not None:
            add_subparser(subcommands, command_name, command.build_parser)

    return parser


def main(args=None):
    _LogManager.init_logger(args)

    parser = make_parser()
    args = parser.parse_args(args)

    if 'command' not in args:
        parser.print_help()
        return 1

    try:
        retcode = args.command(args)
        if retcode is None:
            retcode = 0
        return retcode
    except CliException as e:
        log.error(e)
        return 1
    except Exception as e:
        log.error(e)
        raise


if __name__ == '__main__':
    sys.exit(main())