# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os.path as osp
import sys
import warnings

from ..util.telemetry_utils import (
    close_telemetry_session,
    init_telemetry_session,
    send_command_exception_info,
    send_command_failure_info,
    send_command_success_info,
)
from ..version import __version__
from . import contexts
from .commands import get_non_project_commands, get_project_commands
from .util import add_subparser, make_subcommands_help
from .util.errors import CliException

_log_levels = {
    "debug": log.DEBUG,
    "info": log.INFO,
    "warning": log.WARNING,
    "error": log.ERROR,
    "critical": log.CRITICAL,
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
        log_format = "%(asctime)s %(levelname)s: %(message)s"

        # Try setting up logging with basicConfig.
        # This does nothing, if other parts of the software
        # already configured handlers, i.e. during imports and when
        # main is called programmatically.
        log.basicConfig(format=log_format, level=args.loglevel)
        # Force-overwrite the log level and formatter
        log.root.setLevel(args.loglevel)
        for h in log.root.handlers:
            h.setFormatter(log.Formatter(log_format))

        # Suppress own deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"datumaro\..*")
        # We don't use sklearn directly, but it yells out too much about its deprecations.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sklearn\..*")

    @staticmethod
    def _define_loglevel_option(parser):
        parser.add_argument(
            "--loglevel",
            type=loglevel,
            default="info",
            help="Logging level (options: %s; default: %s)"
            % (", ".join(_log_levels.keys()), "%(default)s"),
        )
        return parser


# TODO: revisit during CLI refactoring
def _get_known_contexts():
    return [
        ("model", contexts.model, "Actions with models"),
        ("project", contexts.project, "Actions with projects"),
        ("source", contexts.source, "Actions with data sources"),
        ("util", contexts.util, "Auxillary tools and utilities"),
    ]


def _get_sensitive_args():
    known_contexts = _get_known_contexts()
    known_commands = get_project_commands() + get_non_project_commands()

    res = {}
    for _, command, _ in known_contexts + known_commands:
        if command is not None:
            res.update(command.get_sensitive_args())

    return res


def make_parser():
    parser = argparse.ArgumentParser(
        description="Dataset Framework", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    if parser.prog == osp.basename(__file__):  # python -m datumaro ...
        parser.prog = "datumaro"

    parser.add_argument("--version", action="version", version=__version__)
    _LogManager._define_loglevel_option(parser)

    known_contexts = _get_known_contexts()
    known_commands = get_non_project_commands()

    # Argparse doesn't support subparser groups:
    # https://stackoverflow.com/questions/32017020/grouping-argparse-subparser-arguments
    help_line_start = max((len(e[0]) for e in known_contexts + known_commands), default=0)
    help_line_start = max((2 + help_line_start) // 4 + 1, 6) * 4  # align to tabs
    subcommands_desc = ""
    if known_contexts:
        subcommands_desc += "Contexts:\n"
        subcommands_desc += make_subcommands_help(known_contexts, help_line_start)
    if known_commands:
        if subcommands_desc:
            subcommands_desc += "\n"
        subcommands_desc += "Context-free Commands:\n"
        subcommands_desc += make_subcommands_help(known_commands, help_line_start)
    if subcommands_desc:
        subcommands_desc += (
            "\nRun '%s COMMAND --help' for more information on a command." % parser.prog
        )

    subcommands = parser.add_subparsers(
        title=subcommands_desc, description="", help=argparse.SUPPRESS
    )
    for command_name, command, _ in known_contexts + known_commands:
        if command is not None:
            add_subparser(subcommands, command_name, command.build_parser)

    return parser


def main(args=None):
    _LogManager.init_logger(args)

    parser = make_parser()
    args = parser.parse_args(args)

    if "command" not in args:
        parser.print_help()
        return 1

    sensitive_args = _get_sensitive_args()
    telemetry = init_telemetry_session(app_name="Datumaro", app_version=__version__)

    try:
        retcode = args.command(args)
        if retcode is None:
            retcode = 0
    except CliException as e:
        log.error(e)
        send_command_exception_info(telemetry, args, sensitive_args=sensitive_args[args.command])
        return 1
    except Exception as e:
        log.error(e)
        send_command_exception_info(telemetry, args, sensitive_args=sensitive_args[args.command])
        raise
    else:
        if retcode:
            send_command_failure_info(telemetry, args, sensitive_args=sensitive_args[args.command])
        else:
            send_command_success_info(telemetry, args, sensitive_args=sensitive_args[args.command])
        return retcode
    finally:
        close_telemetry_session(telemetry)


if __name__ == "__main__":
    sys.exit(main())
