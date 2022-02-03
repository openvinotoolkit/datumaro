# Copyright (C) 2021-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

import orjson

from datumaro.cli.util import MultilineFormatter
from datumaro.cli.util.project import load_project
from datumaro.components.environment import Environment
from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.format_detection import (
    RejectionReason, detect_dataset_format,
)
from datumaro.util.scope import scope_add, scoped


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Detect the format of a dataset",
        description="""
        Attempts to detect the format of a dataset in a directory.
        Currently, only local directories are supported.|n
        |n
        By default, this command shows a human-readable report with the ID
        of the format that was detected (if any). If Datumaro is unable to
        unambiguously determine a single format, all matching formats will
        be shown.|n
        |n
        To see why other formats were rejected, use --show-rejections. To get
        machine-readable output, use --json-report.|n
        |n
        The value of -p/--project is used as a context for plugins.|n
        |n
        Example:|n
        |s|s%(prog)s --show-rejections path/to/dataset
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('url',
        help="URL to the dataset; a path to a directory")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the project to use as the context "
            "(default: current dir)")
    parser.add_argument('--show-rejections', action='store_true',
        help="Describe why each supported format that wasn't detected "
            "was rejected")
    parser.add_argument('--json-report',
        help="Path to which to save a JSON report describing detected "
            "and rejected formats. By default, no report is saved.")
    parser.set_defaults(command=detect_format_command)

    return parser

def get_sensitive_args():
    return {
        detect_format_command: ['url'],
    }

@scoped
def detect_format_command(args):
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    report = {'rejected_formats': {}}

    def rejection_callback(
        format_name: str, reason: RejectionReason, human_message: str,
    ):
        report['rejected_formats'][format_name] = {
            'reason': reason.name,
            'message': human_message,
        }

    detected_formats = detect_dataset_format(
        ((format_name, importer.detect)
            for format_name, importer in env.importers.items.items()),
        args.url,
        rejection_callback=rejection_callback,
    )
    report['detected_formats'] = detected_formats

    if len(detected_formats) == 1:
        print(f"Detected format: {detected_formats[0]}")
    elif len(detected_formats) == 0:
        print("Unable to detect the format")
    else:
        print("Ambiguous dataset; detected the following formats:")
        print()
        for format_name in sorted(detected_formats):
            print(f"- {format_name}")

    if args.show_rejections:
        print()
        if report['rejected_formats']:
            print("The following formats were rejected:")
            print()

            for format_name, rejection in sorted(
                report['rejected_formats'].items()
            ):
                print(f"{format_name}:")
                for line in rejection['message'].split('\n'):
                    print(f"  {line}")
        else:
            print("No formats were rejected.")

    if args.json_report:
        with open(args.json_report, 'wb') as report_file:
            report_file.write(orjson.dumps(report, option=orjson.OPT_INDENT_2))
