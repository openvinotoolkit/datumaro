# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import json
import logging as log

from datumaro.components.operations import ExactComparator

from ..util import MultilineFormatter
from ..util.project import generate_next_file_name, load_project


_ediff_default_if = ['id', 'group'] # avoid https://bugs.python.org/issue16399

def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Compare projects for equality",
        description="""
        Compares two projects for equality.|n
        |n
        Examples:|n
        - Compare two projects, exclude annotation group |n
        |s|s|sand the 'is_crowd' attribute from comparison:|n
        |s|sediff other/project/ -if group -ia is_crowd
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('other_project_dir',
        help="Directory of the second project to be compared")
    parser.add_argument('-iia', '--ignore-item-attr', action='append',
        help="Ignore item attribute (repeatable)")
    parser.add_argument('-ia', '--ignore-attr', action='append',
        help="Ignore annotation attribute (repeatable)")
    parser.add_argument('-if', '--ignore-field', action='append',
        help="Ignore annotation field (repeatable, default: %s)" % \
            _ediff_default_if)
    parser.add_argument('--match-images', action='store_true',
        help='Match dataset items by images instead of ids')
    parser.add_argument('--all', action='store_true',
        help="Include matches in the output")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the first project to be compared (default: current dir)")
    parser.set_defaults(command=ediff_command)

    return parser

def ediff_command(args):
    first_project = load_project(args.project_dir)

    try:
        second_project = load_project(args.other_project_dir)
    except FileNotFoundError:
        if first_project.vcs.is_ref(args.other_project_dir):
            raise NotImplementedError("It seems that you're trying to compare "
                "different revisions of the project. "
                "Comparisons between project revisions are not implemented yet.")
        raise

    if args.ignore_field:
        args.ignore_field = _ediff_default_if
    comparator = ExactComparator(
        match_images=args.match_images,
        ignored_fields=args.ignore_field,
        ignored_attrs=args.ignore_attr,
        ignored_item_attrs=args.ignore_item_attr)
    matches, mismatches, a_extra, b_extra, errors = \
        comparator.compare_datasets(
            first_project.make_dataset(), second_project.make_dataset())
    output = {
        "mismatches": mismatches,
        "a_extra_items": sorted(a_extra),
        "b_extra_items": sorted(b_extra),
        "errors": errors,
    }
    if args.all:
        output["matches"] = matches

    output_file = generate_next_file_name('diff', ext='.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4, sort_keys=True)

    print("Found:")
    print("The first project has %s unmatched items" % len(a_extra))
    print("The second project has %s unmatched items" % len(b_extra))
    print("%s item conflicts" % len(errors))
    print("%s matching annotations" % len(matches))
    print("%s mismatching annotations" % len(mismatches))

    log.info("Output has been saved to '%s'" % output_file)

    return 0
