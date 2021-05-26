
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from enum import Enum

from datumaro.components.project import Project
from datumaro.util.image import load_image


TargetKinds = Enum('TargetKinds',
    ['project', 'source', 'external_dataset', 'inference', 'image'])

def is_project(value):
    if value:
        return Project.find_project_dir(value) != None
    return False

def is_source(value, project=None):
    if project is not None:
        return value in project.sources
    return False

def is_image_path(value):
    try:
        return load_image(value) is not None
    except Exception:
        return False


class Target:
    def __init__(self, kind, test, is_default=False, name=None):
        self.kind = kind
        self.test = test
        self.is_default = is_default
        self.name = name

    def _get_fields(self):
        return [self.kind, self.test, self.is_default, self.name]

    def __str__(self):
        return self.name or str(self.kind)

    def __len__(self):
        return len(self._get_fields())

    def __iter__(self):
        return iter(self._get_fields())

def ProjectTarget(kind=TargetKinds.project, test=is_project,
        is_default=False, name='project name or path'):
    return Target(kind, test, is_default, name)

def SourceTarget(kind=TargetKinds.source, test=None,
        is_default=False, name='source name',
        project=None):
    if test is None:
        test = lambda v: is_source(v, project=project)
    return Target(kind, test, is_default, name)

def ImageTarget(kind=TargetKinds.image, test=is_image_path,
        is_default=False, name='image path'):
    return Target(kind, test, is_default, name)


def target_selector(*targets):
    def selector(value):
        for (kind, test, is_default, _) in targets:
            if (is_default and (value == '' or value is None)) or test(value):
                return (kind, value)
        raise argparse.ArgumentTypeError('Value should be one of: %s' \
            % (', '.join([str(t) for t in targets])))
    return selector
