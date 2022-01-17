# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os.path as osp
import sys

from ..util.telemetry_utils import (
    close_telemetry_session, init_telemetry_session,
    send_command_exception_info, send_command_failure_info,
    send_command_success_info,
)
from ..version import VERSION
from . import commands, contexts
from .util import add_subparser
from .util.errors import CliException

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

def _get_known_contexts():
    return [
        ('model', contexts.model, "Actions with models"),
        ('project', contexts.project, "Actions with projects"),
        ('source', contexts.source, "Actions with data sources"),
        ('util', contexts.util, "Auxillary tools and utilities"),
    ]

def _get_known_commands():
    return [
        ("Project modification:", None, ''),
        ('add', commands.add, "Add dataset"),
        ('create', commands.create, "Create empty project"),
        ('import', commands.import_, "Import dataset"),
        ('remove', commands.remove, "Remove dataset"),

        ("", None, ''),
        ("Project versioning:", None, ''),
        ('checkout', commands.checkout, "Switch to another branch or revision"),
        ('commit', commands.commit, "Commit changes in tracked files"),
        ('log', commands.log, "List history"),
        ('status', commands.status, "Display current status"),

        ("", None, ''),
        ("Dataset operations:", None, ''),
        ('convert', commands.convert, "Convert dataset between formats"),
        ('diff', commands.diff, "Compare datasets"),
        ('download', commands.download, "Download a publicly available dataset"),
        ('explain', commands.explain, "Run Explainable AI algorithm for model"),
        ('export', commands.export, "Export dataset in some format"),
        ('filter', commands.filter, "Filter dataset items"),
        ('info', commands.info, "Print dataset info"),
        ('merge', commands.merge, "Merge datasets"),
        ('patch', commands.patch, "Update dataset from another one"),
        ('stats', commands.stats, "Compute dataset statistics"),
        ('transform', commands.transform, "Modify dataset items"),
        ('validate', commands.validate, "Validate dataset")
    ]

def _get_sensitive_args():
    known_contexts = _get_known_contexts()
    known_commands = _get_known_commands()

    res = {}
    for _, command, _ in known_contexts + known_commands:
        if command is not None:
            res.update(command.get_sensitive_args())

    return res

def make_parser():
    parser = argparse.ArgumentParser(
        description="Dataset Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    if parser.prog == osp.basename(__file__): # python -m datumaro ...
        parser.prog = 'datumaro'

    parser.add_argument('--version', action='version', version=VERSION)
    _LogManager._define_loglevel_option(parser)

    known_contexts = _get_known_contexts()
    known_commands = _get_known_commands()

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

    sensitive_args = _get_sensitive_args()
    telemetry = init_telemetry_session(app_name='Datumaro', app_version=VERSION)

    try:
        retcode = args.command(args)
        if retcode is None:
            retcode = 0
    except CliException as e:
        log.error(e)
        send_command_exception_info(telemetry, args,
            sensitive_args=sensitive_args[args.command])
        return 1
    except Exception as e:
        log.error(e)
        send_command_exception_info(telemetry, args,
            sensitive_args=sensitive_args[args.command])
        raise
    else:
        if retcode:
            send_command_failure_info(telemetry, args,
                sensitive_args=sensitive_args[args.command])
        else:
            send_command_success_info(telemetry, args,
                sensitive_args=sensitive_args[args.command])
        return retcode
    finally:
        close_telemetry_session(telemetry)

if __name__ == '__main__':
    sys.exit(main())
