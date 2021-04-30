# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import os.path as osp
from io import StringIO

from datumaro.components.config import Config
from datumaro.components.config_model import PROJECT_SCHEMA

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=status_command)

    return parser

def status_command(args):
    project = load_project(args.project_dir)

    data_status = project.vcs.dvc.status()
    for stage_name, stage_status in data_status.items():
        if stage_name.endswith('.dvc'):
            stage_name = osp.splitext(osp.basename(stage_name))[0]
        print(stage_status, stage_name)

    project_status = project.vcs.git.status()
    config_path = osp.join(project.config.env_dir,
        project.config.project_filename)
    if config_path in project_status:
        current_conf = Config.parse(config_path, schema=PROJECT_SCHEMA)

        prev_conf = project.vcs.git.show(config_path, rev='HEAD')
        prev_conf = Config.parse(StringIO(prev_conf), schema=PROJECT_SCHEMA)


        a_sources = set(prev_conf.sources)
        b_sources = set(current_conf.sources)

        added = b_sources - a_sources
        removed = a_sources - b_sources
        modified = set(s for s in a_sources & b_sources
            if (prev_conf.sources[s] != current_conf.sources[s]) or \
               (prev_conf.build_targets[s] != current_conf.build_targets[s])
            )

        for s in a_sources | b_sources:
            if s in added:
                print('A', s)
            if s in removed:
                print('D', s)
            if s in modified:
                print('M', s)

    for path, path_status in project_status.items():
        if path.endswith('.dvc'):
            path = osp.splitext(osp.basename(path))[0]
            print(path_status, path)

    return 0
