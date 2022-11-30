# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.components.environment import Environment
from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.validator import TaskType
from datumaro.util import dump_json_file
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import generate_next_file_name, load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Validate project",
        description="""
        Validates a dataset according to the task type and
        reports summary in a JSON file.|n
        Target dataset is specified by a revpath. The full syntax is:|n
        - Dataset paths:|n
        |s|s- <dataset path>[ :<format> ]|n
        - Revision paths:|n
        |s|s- <project path> [ @<rev> ] [ :<target> ]|n
        |s|s- <rev> [ :<target> ]|n
        |s|s- <target>|n
        |n
        Both forms use the -p/--project as a context for plugins. It can be
        useful for dataset paths in targets. When not specified, the current
        project's working tree is used.|n
        |n
        Examples:|n
        - Validate a project's subset as a classification dataset:|n |n
        |s|s%(prog)s -t classification -s train
        """,
        formatter_class=MultilineFormatter,
    )

    task_types = ", ".join(t.name for t in TaskType)

    def _parse_task_type(s):
        try:
            return TaskType[s.lower()].name
        except:
            raise argparse.ArgumentTypeError(
                "Unknown task type %s. Expected " "one of: %s" % (s, task_types)
            )

    parser.add_argument(
        "_positionals", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
    )  # workaround for -- eaten by positionals
    parser.add_argument(
        "target", default="project", nargs="?", help="Target dataset revpath (default: project)"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=_parse_task_type,
        required=True,
        help="Task type for validation, one of %s" % task_types,
    )
    parser.add_argument(
        "-s", "--subset", dest="subset_name", help="Subset to validate (default: whole dataset)"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to validate (default: current dir)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments for validator (pass '-- -h' for help)",
    )
    parser.set_defaults(command=validate_command)

    return parser


def get_sensitive_args():
    return {
        validate_command: ["target", "project_dir", "subset_name", "extra_args"],
    }


@scoped
def validate_command(args):
    has_sep = "--" in args._positionals
    if has_sep:
        pos = args._positionals.index("--")
        if 1 < pos:
            raise argparse.ArgumentError(None, message="Expected no more than 1 target argument")
    else:
        pos = 1
    args.target = (args._positionals[:pos] or ["project"])[0]
    args.extra_args = args._positionals[pos + has_sep :]

    show_plugin_help = "-h" in args.extra_args or "--help" in args.extra_args

    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if not show_plugin_help and args.project_dir:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    try:
        validator_type = env.validators[args.task]
    except KeyError:
        raise CliException("Validator type '%s' is not found" % args.task)

    extra_args = validator_type.parse_cmdline(args.extra_args)

    dataset, target_project = parse_full_revpath(args.target, project)
    if target_project:
        scope_add(target_project)

    dst_file_name = f"validation-report"
    if args.subset_name is not None:
        dataset = dataset.get_subset(args.subset_name)
        dst_file_name += f"-{args.subset_name}"

    validator = validator_type(**extra_args)
    report = validator.validate(dataset)

    def _make_serializable(d):
        for key, val in list(d.items()):
            # tuple key to str
            if isinstance(key, tuple):
                d[str(key)] = val
                d.pop(key)
            if isinstance(val, dict):
                _make_serializable(val)

    _make_serializable(report)

    dst_file = generate_next_file_name(dst_file_name, ext=".json")
    log.info("Writing project validation results to '%s'" % dst_file)
    dump_json_file(dst_file, report, indent=True, allow_numpy=True)
