# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from datumaro.components.errors import (
    DatasetMergeError, MissingObjectError, ProjectNotFoundError,
)
from datumaro.components.extractor import AnnotationType
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.project import load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Prints dataset overview",
        description="""
        Prints info about the dataset at <revpath>, or about the current
        project's combined dataset, if none is specified.

        <revpath> - either a dataset path or a revision path. The full
        syntax is:
        - Dataset paths:
          - <dataset path>[ :<format> ]
        - Revision paths:
          - <project path> [ @<rev> ] [ :<target> ]
          - <rev> [ :<target> ]
          - <target>

        Both forms use the -p/--project as a context for plugins. It can be
        useful for dataset paths in targets. When not specified, the current
        project's working tree is used.

        Examples:
        - Print dataset info for the current project's working tree:
          %(prog)s

        - Print dataset info for a path and a format name:
          %(prog)s path/to/dataset:voc

        - Print dataset info for a source from a past revision:
          %(prog)s HEAD~2:source-2
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('target', nargs='?', default='project',
        metavar='revpath',
        help="Target dataset revpath")
    parser.add_argument('--all', action='store_true',
        help="Print all information")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the current project (default: current dir)")
    parser.set_defaults(command=info_command)

    return parser

def get_sensitive_args():
    return {
        info_command: ['target', 'project_dir',],
    }

@scoped
def info_command(args):
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    try:
        # TODO: avoid computing working tree hashes
        dataset, target_project = parse_full_revpath(args.target, project)
        if target_project:
            scope_add(target_project)
    except DatasetMergeError as e:
        dataset = None
        dataset_problem = "Can't merge project sources automatically: %s " \
            "Conflicting sources are: %s" % (e, ', '.join(e.sources))
    except MissingObjectError as e:
        dataset = None
        dataset_problem = str(e)

    def print_dataset_info(dataset, indent=''):
        print("%slength:" % indent, len(dataset))

        categories = dataset.categories()
        print("%scategories:" % indent, ', '.join(c.name for c in categories))

        for cat_type, cat in categories.items():
            print("%s  %s:" % (indent, cat_type.name))
            if cat_type == AnnotationType.label:
                print("%s    count:" % indent, len(cat.items))

                count_threshold = 10
                if args.all:
                    count_threshold = len(cat.items)
                labels = ', '.join(c.name for c in cat.items[:count_threshold])
                if count_threshold < len(cat.items):
                    labels += " (and %s more)" % (
                        len(cat.items) - count_threshold)
                print("%s    labels:" % indent, labels)

    if dataset is not None:
        print_dataset_info(dataset)

        subsets = dataset.subsets()
        print("subsets:", ', '.join(subsets))
        for subset_name in subsets:
            subset = dataset.get_subset(subset_name)
            print("  '%s':" % subset_name)
            print_dataset_info(subset, indent="    ")
    else:
        print("Dataset info is not available: ", dataset_problem)

    return 0
