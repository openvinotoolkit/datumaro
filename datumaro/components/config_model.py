
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.config import Config, \
    DictConfig as _DictConfig, \
    SchemaBuilder as _SchemaBuilder

from datumaro.util import find


REMOTE_SCHEMA = _SchemaBuilder() \
    .add('url', str) \
    .add('type', str) \
    .add('options', dict) \
    .build()

class Remote(Config):
    def __init__(self, config=None):
        super().__init__(config, schema=REMOTE_SCHEMA)


SOURCE_SCHEMA = _SchemaBuilder() \
    .add('url', str) \
    .add('format', str) \
    .add('options', dict) \
    .add('hash', str) \
    .build()

class Source(Config):
    def __init__(self, config=None):
        super().__init__(config, schema=SOURCE_SCHEMA)

    @property
    def is_generated(self) -> bool:
        return not self.url


MODEL_SCHEMA = _SchemaBuilder() \
    .add('url', str) \
    .add('launcher', str) \
    .add('options', dict) \
    .add('hash', str) \
    .build()

class Model(Config):
    def __init__(self, config=None):
        super().__init__(config, schema=MODEL_SCHEMA)


BUILDSTAGE_SCHEMA = _SchemaBuilder() \
    .add('name', str) \
    .add('type', str) \
    .add('kind', str) \
    .add('hash', str) \
    .add('params', dict) \
    .build()

class BuildStage(Config):
    def __init__(self, config=None):
        super().__init__(config, schema=BUILDSTAGE_SCHEMA)

BUILDTARGET_SCHEMA = _SchemaBuilder() \
    .add('stages', list) \
    .add('parents', list) \
    .build()

class BuildTarget(Config):
    def __init__(self, config=None):
        super().__init__(config, schema=BUILDTARGET_SCHEMA)
        self.stages = [BuildStage(o) for o in self.stages]

    @property
    def root(self):
        return self.stages[0]

    @property
    def head(self):
        return self.stages[-1]

    @property
    def is_clean(self) -> bool:
        return len(self.stages) == 1

    def find_stage(self, stage):
        if stage == 'root':
            return self.root
        elif stage == 'head':
            return self.head
        return find(self.stages, lambda x: x.name == stage or x == stage)

    def get_stage(self, stage):
        res = self.find_stage(stage)
        if res is None:
            raise KeyError("Unknown stage '%s'" % stage)
        return res


TREE_SCHEMA = _SchemaBuilder() \
    .add('project_name', str) \
    .add('format_version', int) \
    \
    .add('default_repo', str) \
    .add('remotes', lambda: _DictConfig(lambda v=None: Remote(v))) \
    .add('sources', lambda: _DictConfig(lambda v=None: Source(v))) \
    .add('models', lambda: _DictConfig(lambda v=None: Model(v))) \
    .add('build_targets', lambda: _DictConfig(lambda v=None: BuildTarget(v))) \
    \
    .add('models_dir', str, internal=True) \
    .add('plugins_dir', str, internal=True) \
    .add('sources_dir', str, internal=True) \
    .add('dataset_dir', str, internal=True) \
    .add('dvc_aux_dir', str, internal=True) \
    .add('pipelines_dir', str, internal=True) \
    .add('build_dir', str, internal=True) \
    .add('cache_dir', str, internal=True) \
    .add('revisions_dir', str, internal=True) \
    \
    .add('config_path', str, internal=True) \
    .add('env_dir', str, internal=True) \
    .build()

TREE_DEFAULT_CONFIG = Config({
    'project_name': 'undefined',
    'format_version': 2,

    'default_repo': 'origin',

    'sources_dir': 'sources',
    'dataset_dir': 'dataset',
    'models_dir': 'models',
    'plugins_dir': 'plugins',
    'dvc_aux_dir': 'dvc_aux',
    'pipelines_dir': 'dvc_pipelines',
    'build_dir': 'build',
    'cache_dir': 'cache',
    'revisions_dir': 'revisions',

    'config_path': 'config.yaml',
    'env_dir': '.datumaro',
}, mutable=False, schema=TREE_SCHEMA)

class TreeConfig(Config):
    def __init__(self, config=None, mutable=True):
        super().__init__(config=config, mutable=mutable,
            fallback=TREE_DEFAULT_CONFIG, schema=TREE_SCHEMA)

class PipelineConfig(Config):
    pass

class ProjectLayout:
    aux_dir = '.datumaro'
    cache_dir = 'cache'
    index_dir = 'index'
    work_dir = 'work_dir'
    head_file = 'head'
    index_tree_dir = osp.join(index_dir, 'tree')
    index_cache_dir = osp.join(index_dir, 'cache')
    tmp_dir = 'tmp'

class TreeLayout:
    conf_file = 'config.yml'
    plugins_dir = 'plugins'
    models_dir = 'models'
    sources_dir = 'sources'