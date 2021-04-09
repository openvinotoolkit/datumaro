# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import logging as log
import networkx as nx
import os
import os.path as osp
import shutil
import unittest.mock
import urllib.parse
import yaml
from contextlib import ExitStack
from enum import Enum
from functools import partial
from glob import glob
from typing import Dict, List, Optional, Tuple, Union
from ruamel.yaml import YAML

from datumaro.components.config import Config
from datumaro.components.config_model import (PROJECT_DEFAULT_CONFIG,
    PROJECT_SCHEMA, BuildStage, Remote, Source)
from datumaro.components.environment import Environment
from datumaro.components.errors import (DatumaroError, DetachedProjectError,
    ReadonlyProjectError, SourceExistsError, VcsError)
from datumaro.components.dataset import Dataset, DEFAULT_FORMAT
from datumaro.util import find, error_rollback, parse_str_enum_value
from datumaro.util.os_util import make_file_name, generate_next_name
from datumaro.util.log_utils import logging_disabled, catch_logs


class ProjectSourceDataset(Dataset):
    @classmethod
    def from_source(cls, project: 'Project', source: Source):
        config = project.sources[source]

        path = osp.join(project.sources.data_dir(source), config.url)
        readonly = not path or not osp.exists(path)
        if path and not osp.exists(path) and not config.remote:
            # backward compatibility
            path = osp.join(project.config.project_dir, config.url)
            readonly = True

        dataset = cls.import_from(path, env=project.env,
            format=config.format, **config.options)
        dataset._project = project
        dataset._config = config
        dataset._readonly = readonly
        return dataset

    def save(self, save_dir=None, **kwargs):
        if save_dir is None:
            if self.readonly:
                raise ReadonlyProjectError("Can't update a read-only dataset")
        super().save(save_dir, **kwargs)

    @property
    def readonly(self):
        return not self._readonly and self.is_bound and \
            self._project.vcs.writeable

    @property
    def _env(self):
        return self._project.env

    @property
    def config(self):
        return self._config

    def run_model(self, model, batch_size=1):
        if isinstance(model, str):
            model = self._project.models.make_executable_model(model)
        return super().run_model(model, batch_size=batch_size)


MergeStrategy = Enum('MergeStrategy', ['ours', 'theirs', 'conflict'])

class CrudProxy:
    @property
    def _data(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, name):
        return self._data[name]

    def get(self, name, default=None):
        return self._data.get(name, default)

    def __iter__(self):
        return iter(self._data.keys())

    def items(self):
        return iter(self._data.items())

    def __contains__(self, name):
        return name in self._data

class ProjectRepositories(CrudProxy):
    def __init__(self, project_vcs):
        self._vcs = project_vcs

    def set_default(self, name):
        if name not in self:
            raise KeyError("Unknown repository name '%s'" % name)
        self._vcs._project.config.default_repo = name

    def get_default(self):
        return self._vcs._project.config.default_repo

    @CrudProxy._data.getter
    def _data(self):
        return self._vcs.git.list_remotes()

    def add(self, name, url):
        self._vcs.git.add_remote(name, url)

    def remove(self, name):
        self._vcs.git.remove_remote(name)

class ProjectRemotes(CrudProxy):
    SUPPORTED_PROTOCOLS = {'', 'remote', 's3', 'ssh', 'http', 'https'}

    def __init__(self, project_vcs):
        self._vcs = project_vcs

    def fetch(self, name=None):
        self._vcs.dvc.fetch_remote(name)

    def pull(self, name=None):
        self._vcs.dvc.pull_remote(name)

    def push(self, name=None):
        self._vcs.dvc.push_remote(name)

    def set_default(self, name):
        self._vcs.dvc.set_default_remote(name)

    def get_default(self):
        return self._vcs.dvc.get_default_remote()

    @CrudProxy._data.getter
    def _data(self):
        return self._vcs._project.config.remotes

    def add(self, name, value):
        url_parts = self.validate_url(value['url'])
        if not url_parts.scheme:
            value['url'] = osp.abspath(value['url'])
        if not value.get('type'):
            value['type'] = 'url'

        if not isinstance(value, Remote):
            value = Remote(value)
        value = self._data.set(name, value)

        assert value.type in {'url', 'git', 'dvc'}, value.type
        self._vcs.dvc.add_remote(name, value)
        return value

    def remove(self, name, force=False):
        try:
            self._vcs.dvc.remove_remote(name)
        except DvcWrapper.DvcError:
            if not force:
                raise

    @classmethod
    def validate_url(cls, url):
        url_parts = urllib.parse.urlsplit(url)
        if url_parts.scheme not in cls.SUPPORTED_PROTOCOLS and \
                not osp.exists(url):
            raise ValueError(
                "Invalid remote '%s': scheme '%s' is not supported, the only"
                "available are: %s" % \
                (url, url_parts.scheme, ', '.join(cls.SUPPORTED_PROTOCOLS))
            )
        if not (url_parts.hostname or url_parts.path):
            raise ValueError("URL must not be empty, url: '%s'" % url)
        return url_parts

class _DataSourceBase(CrudProxy):
    def __init__(self, project, config_field):
        self._project = project
        self._field = config_field

    @CrudProxy._data.getter
    def _data(self):
        return self._project.config[self._field]

    def pull(self, names=None, rev=None):
        if not self._project.vcs.writeable:
            raise ReadonlyProjectError("Can't pull in a read-only project")

        if not names:
            names = []
        elif isinstance(names, str):
            names = [names]
        else:
            names = list(names)

        for name in names:
            if name and name not in self:
                raise KeyError("Unknown source '%s'" % name)

        if rev and len(names) != 1:
            raise ValueError("A revision can only be specified for a "
                "single source invocation")

        self._project.vcs.dvc.update_imports(
            [self.dvcfile_path(name) for name in names])

    def fetch(self, names=None):
        if not self._project.vcs.readable:
            raise DetachedProjectError("Can't fetch in a detached project")

        if not names:
            names = []
        elif isinstance(names, str):
            names = [names]
        else:
            names = list(names)

        for name in names:
            if name and name not in self:
                raise KeyError("Unknown source '%s'" % name)

        self._project.vcs.dvc.fetch(
            [self.dvcfile_path(name) for name in names])

    def checkout(self, names=None):
        # TODO: need to add DVC cache interaction and checking of the
        # checked-out revision hash. In the case of mismatch, run rebuild

        if not self._project.vcs.writeable:
            raise ReadonlyProjectError("Can't checkout in a read-only project")

        if not names:
            names = []
        elif isinstance(names, str):
            names = [names]
        else:
            names = list(names)

        for name in names:
            if name and name not in self:
                raise KeyError("Unknown source '%s'" % name)

        self._project.vcs.dvc.checkout(
            [self.dvcfile_path(name) for name in names])

    def push(self, names=None):
        if not self._project.vcs.writeable:
            raise ReadonlyProjectError("Can't push in a read-only project")

        if not names:
            names = []
        elif isinstance(names, str):
            names = [names]
        else:
            names = list(names)

        for name in names:
            if name and name not in self:
                raise KeyError("Unknown source '%s'" % name)

        self._project.vcs.dvc.push([self.dvcfile_path(name) for name in names])

    @classmethod
    def _validate_url(cls, url):
        return ProjectRemotes.validate_url(url)

    @classmethod
    def _make_remote_name(cls, name):
        return name

    def data_dir(self, name):
        return osp.join(self._project.config.project_dir, name)

    def validate_name(self, name):
        valid_filename = make_file_name(name)
        if valid_filename != name:
            raise ValueError("Source name contains "
                "prohibited symbols: %s" % (set(name) - set(valid_filename)) )

        if name.startswith('.'):
            raise ValueError("Source name can't start with '.'")

    def dvcfile_path(self, name):
        return self._project.vcs.dvc_filepath(name)

    @classmethod
    def _fix_dvc_file(cls, source_path, dvc_path, dst_name):
        with open(dvc_path, 'r+') as dvc_file:
            yaml = YAML(typ='rt')
            dvc_data = yaml.load(dvc_file)
            dvc_data['wdir'] = osp.join(
                dvc_data['wdir'], osp.basename(source_path))
            dvc_data['outs'][0]['path'] = dst_name

            dvc_file.seek(0)
            yaml.dump(dvc_data, dvc_file)
            dvc_file.truncate()

    def _ensure_in_dir(self, source_path, dvc_path, dst_name):
        if not osp.isfile(source_path):
            return
        tmp_dir = osp.join(self._project.config.project_dir,
            self._project.config.env_dir, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        source_tmp = osp.join(tmp_dir, osp.basename(source_path))
        os.replace(source_path, source_tmp)
        os.makedirs(source_path)
        os.replace(source_tmp, osp.join(source_path, dst_name))

        self._fix_dvc_file(source_path, dvc_path, dst_name)

    @error_rollback('on_error', implicit=True)
    def add(self, name, value):
        self.validate_name(name)

        if name in self:
            raise SourceExistsError("Source '%s' already exists" % name)

        url = value.get('url', '')

        if self._project.vcs.writeable:
            if url:
                url_parts = self._validate_url(url)

            if not url:
                # a generated source
                remote_name = ''
                path = url
            elif url_parts.scheme == 'remote':
                # add a source with existing remote
                remote_name = url_parts.netloc
                remote_conf = self._project.vcs.remotes[remote_name]
                path = url_parts.path
                url = remote_conf.url + path
            else:
                # add a source and a new remote
                if not url_parts.scheme and not osp.exists(url):
                    raise FileNotFoundError(
                        "Can't find file or directory '%s'" % url)

                remote_name = self._make_remote_name(name)
                if remote_name not in self._project.vcs.remotes:
                    on_error.do(self._project.vcs.remotes.remove, remote_name,
                        ignore_errors=True)
                remote_conf = self._project.vcs.remotes.add(remote_name, {
                    'url': url,
                    'type': 'url',
                })
                path = ''

            source_dir = self.data_dir(name)

            dvcfile = self.dvcfile_path(name)
            if not osp.isfile(dvcfile):
                on_error.do(os.remove, dvcfile, ignore_errors=True)

            if not remote_name:
                pass
            elif remote_conf.type == 'url':
                self._project.vcs.dvc.import_url(
                    'remote://%s%s' % (remote_name, path),
                    out=source_dir, dvc_path=dvcfile, download=True)
                self._ensure_in_dir(source_dir, dvcfile, osp.basename(url))
            elif remote_conf.type == 'git':
                self._project.vcs.dvc.import_repo(remote_conf.url, path=path,
                    out=source_dir, dvc_path=dvcfile, download=True)
                self._ensure_in_dir(source_dir, dvcfile, osp.basename(url))
            else:
                raise ValueError("Unknown remote type '%s'" % remote_conf.type)

            path = osp.basename(path)
        else:
            if not url or osp.exists(url):
                # a local or a generated source
                # in a read-only or in-memory project
                remote_name = ''
                path = url
            else:
                raise VcsError("Can only add an existing local, or generated "
                    "source to a detached project")

        value['url'] = path
        value['remote'] = remote_name
        value = self._data.set(name, value)

        return value

    def remove(self, name, force=False, keep_data=True):
        """Force - ignores errors and tries to wipe remaining data"""

        if name not in self._data and not force:
            raise KeyError("Unknown source '%s'" % name)

        self._data.remove(name)

        if not self._project.vcs.writeable:
            return

        if force and not keep_data:
            source_dir = self.data_dir(name)
            if osp.isdir(source_dir):
                shutil.rmtree(source_dir, ignore_errors=True)

        dvcfile = self.dvcfile_path(name)
        if osp.isfile(dvcfile):
            try:
                self._project.vcs.dvc.remove(dvcfile, outs=not keep_data)
            except DvcWrapper.DvcError:
                if force:
                    os.remove(dvcfile)
                else:
                    raise

        self._project.vcs.remotes.remove(name, force=force)

class ProjectModels(_DataSourceBase):
    def __init__(self, project):
        super().__init__(project, 'models')

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError("Unknown model '%s'" % name)

    def data_dir(self, name):
        return osp.join(
            self._project.config.project_dir,
            self._project.config.env_dir,
            self._project.config.models_dir, name)

    def make_executable_model(self, name):
        model = self[name]
        return self._project.env.make_launcher(model.launcher,
            **model.options, model_dir=self.data_dir(name))

class ProjectSources(_DataSourceBase):
    def __init__(self, project):
        super().__init__(project, 'sources')

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError("Unknown source '%s'" % name)

    def make_dataset(self, name):
        return ProjectSourceDataset.from_source(self._project, name)

    def validate_name(self, name):
        super().validate_name(name)

        reserved_names = {'dataset', 'build', 'project'}
        if name.lower() in reserved_names:
            raise ValueError("Source name is reserved for internal use")

    def add(self, name, value):
        value = super().add(name, value)

        self._project.build_targets.add_target(name)

        return value

    def remove(self, name, force=False, keep_data=True):
        self._project.build_targets.remove_target(name)

        super().remove(name, force=force, keep_data=keep_data)


BuildStageType = Enum('BuildStageType',
    ['source', 'project', 'transform', 'filter', 'convert', 'inference'])

class ProjectBuildTargets(CrudProxy):
    def __init__(self, project):
        self._project = project

    @CrudProxy._data.getter
    def _data(self):
        data = self._project.config.build_targets

        if self.MAIN_TARGET not in data:
            data[self.MAIN_TARGET] = {
                'stages': [
                    BuildStage({
                        'name': self.BASE_STAGE,
                        'type': BuildStageType.project.name,
                    }),
                ]
            }

        for source in self._project.sources:
            if source not in data:
                data[source] = {
                    'stages': [
                        BuildStage({
                            'name': self.BASE_STAGE,
                            'type': BuildStageType.source.name,
                        }),
                    ]
                }

        return data

    def __contains__(self, key):
        if '.' in key:
            target, stage = self._split_target_name(key)
            return target in self._data and \
                self._data[target].find_stage(stage) is not None
        return key in self._data

    def add_target(self, name):
        return self._data.set(name, {
            'stages': [
                BuildStage({
                    'name': self.BASE_STAGE,
                    'type': BuildStageType.source.name,
                }),
            ]
        })

    def add_stage(self, target, value, prev=None,
            name=None) -> Tuple[BuildStage, str]:
        target_name = target
        target_stage_name = None
        if '.' in target:
            target_name, target_stage_name = self._split_target_name(target)

        if prev is None:
            prev = target_stage_name

        target = self._data[target_name]

        if prev:
            prev_stage = find(enumerate(target.stages),
                lambda e: e[1].name == prev)
            if prev_stage is None:
                raise KeyError("Can't find stage '%s'" % prev)
            prev_stage = prev_stage[0]
        else:
            prev_stage = len(target.stages) - 1

        name = value.get('name') or name
        if not name:
            name = generate_next_name((s.name for s in target.stages),
                value['type'], sep='-')
        else:
            if target.find_stage(name):
                raise VcsError("Stage '%s' already exists" % name)
        value['name'] = name

        value = BuildStage(value)
        assert BuildStageType[value.type]
        target.stages.insert(prev_stage + 1, value)
        return value, self._make_target_name(target_name, name)

    def remove_target(self, name):
        assert name != self.MAIN_TARGET, "Can't remove the main target"
        self._data.remove(name)

    def remove_stage(self, target, name):
        assert name not in {self.BASE_STAGE}, "Can't remove a default stage"

        target = self._data[target]
        idx = find(enumerate(target.stages), lambda e: e[1].name == name)
        if idx is None:
            raise KeyError("Can't find stage '%s'" % name)
        target.stages.remove(idx)

    def add_transform_stage(self, target, transform, params=None, name=None):
        if not transform in self._project.env.transforms:
            raise KeyError("Unknown transform '%s'" % transform)

        return self.add_stage(target, {
            'type': BuildStageType.transform.name,
            'kind': transform,
            'params': params or {},
        }, name=name)

    def add_inference_stage(self, target, model, name=None):
        if not model in self._project.config.models:
            raise KeyError("Unknown model '%s'" % model)

        return self.add_stage(target, {
            'type': BuildStageType.inference.name,
            'kind': model,
        }, name=name)

    def add_filter_stage(self, target, params=None, name=None):
        return self.add_stage(target, {
            'type': BuildStageType.filter.name,
            'params': params or {},
        }, name=name)

    def add_convert_stage(self, target, format, \
            params=None, name=None): # pylint: disable=redefined-builtin
        if not self._project.env.is_format_known(format):
            raise KeyError("Unknown format '%s'" % format)

        return self.add_stage(target, {
            'type': BuildStageType.convert.name,
            'kind': format,
            'params': params or {},
        }, name=name)

    MAIN_TARGET = 'project'
    BASE_STAGE = 'root'
    def _get_build_graph(self):
        graph = nx.DiGraph()
        for target_name, target in self.items():
            if target_name == self.MAIN_TARGET:
                # main target combines all the others
                prev_stages = [self._make_target_name(n, t.head.name)
                    for n, t in self.items() if n != self.MAIN_TARGET]
            else:
                prev_stages = [self._make_target_name(t, self[t].head.name)
                    for t in target.parents]

            for stage in target.stages:
                stage_name = self._make_target_name(target_name, stage['name'])
                graph.add_node(stage_name, config=stage)
                for prev_stage in prev_stages:
                    graph.add_edge(prev_stage, stage_name)
                prev_stages = [stage_name]

        return graph

    @staticmethod
    def _make_target_name(target, stage=None):
        if stage:
            return '%s.%s' % (target, stage)
        return target

    @classmethod
    def _split_target_name(cls, name):
        if '.' in name:
            target, stage = name.split('.', maxsplit=1)
            if not target:
                raise ValueError("Wrong target name '%s' - target name can't "
                    "be empty" % name)
            if not stage:
                raise ValueError("Wrong target name '%s' - expected "
                    "stage name after the separator" % name)
        else:
            target = name
            stage = cls.BASE_STAGE
        return target, stage

    def _get_target_subgraph(self, target):
        if '.' not in target:
            target = self._make_target_name(target, self[target].head.name)

        full_graph = self._get_build_graph()

        target_parents = set()
        visited = set()
        to_visit = {target}
        while to_visit:
            current = to_visit.pop()
            visited.add(current)
            for pred in full_graph.predecessors(current):
                target_parents.add(pred)
                if pred not in visited:
                    to_visit.add(pred)

        target_parents.add(target)

        return full_graph.subgraph(target_parents)

    def _get_target_config(self, name):
        """Returns a target or stage description"""
        target, stage = self._split_target_name(name)
        target_config = self._data[target]
        stage_config = target_config.get_stage(stage)
        return stage_config

    def make_pipeline(self, target):
        # a subgraph with all the target dependencies
        target_subgraph = self._get_target_subgraph(target)
        pipeline = []
        for node_name, node in target_subgraph.nodes.items():
            entry = {
                'name': node_name,
                'parents': list(target_subgraph.predecessors(node_name)),
                'config': dict(node['config']),
            }
            pipeline.append(entry)
        return pipeline

    def generate_pipeline(self, target):
        real_target = self._normalize_target(target)

        pipeline = self.make_pipeline(real_target)
        path = osp.join(self._project.config.project_dir,
            self._project.config.env_dir, self._project.config.pipelines_dir)
        os.makedirs(path, exist_ok=True)
        path = osp.join(path, make_file_name(target) + '.yml')
        self.write_pipeline(pipeline, path)

        return path

    @classmethod
    def _read_pipeline_graph(cls, pipeline):
        graph = nx.DiGraph()
        for entry in pipeline:
            target_name = entry['name']
            parents = entry['parents']
            target = BuildStage(entry['config'])

            graph.add_node(target_name, config=target)
            for prev_stage in parents:
                graph.add_edge(prev_stage, target_name)

        return graph

    def apply_pipeline(self, pipeline):
        def _join_parent_datasets():
            if 1 < len(parent_datasets):
                dataset = Dataset.from_extractors(*parent_datasets,
                    env=self._project.env)
            else:
                dataset = parent_datasets[0]
            return dataset

        if len(pipeline) == 0:
            raise Exception("Can't run empty pipeline")

        graph = self._read_pipeline_graph(pipeline)

        head = None
        for node in graph.nodes:
            if graph.out_degree(node) == 0:
                assert head is None, "A pipeline can have only one " \
                    "main target, but it has at least 2: %s, %s" % \
                    (head, node)
                head = node
        assert head is not None, "A pipeline must have a finishing node"

        # Use DFS to traverse the graph and initialize nodes from roots to tops
        to_visit = [head]
        while to_visit:
            current_name = to_visit.pop()
            current = graph.nodes[current_name]

            assert current.get('dataset') is None

            parents_uninitialized = []
            parent_datasets = []
            for p_name in graph.predecessors(current_name):
                parent = graph.nodes[p_name]
                dataset = parent.get('dataset')
                if dataset is None:
                    parents_uninitialized.append(p_name)
                else:
                    parent_datasets.append(dataset)

            if parents_uninitialized:
                to_visit.append(current_name)
                to_visit.extend(parents_uninitialized)
                continue

            type_ = BuildStageType[current['config'].type]
            params = current['config'].params
            if type_ == BuildStageType.transform:
                kind = current['config'].kind
                try:
                    transform = self._project.env.transforms[kind]
                except KeyError:
                    raise KeyError("Unknown transform '%s'" % kind)

                dataset = _join_parent_datasets()
                dataset = dataset.transform(transform, **params)

            elif type_ == BuildStageType.filter:
                dataset = _join_parent_datasets()
                dataset = dataset.filter(**params)

            elif type_ == BuildStageType.inference:
                kind = current['config'].kind
                model = self._project.models.make_executable_model(kind)

                dataset = _join_parent_datasets()
                dataset = dataset.run_model(model)

            elif type_ == BuildStageType.source:
                assert len(parent_datasets) == 0, current_name
                source, _ = self._split_target_name(current_name)
                dataset = self._project.sources.make_dataset(source)

            elif type_ == BuildStageType.project:
                dataset = Dataset.from_extractors(*parent_datasets,
                    env=self._project.env)

            elif type_ == BuildStageType.convert:
                dataset = _join_parent_datasets()

            else:
                raise NotImplementedError("Unknown stage type '%s'" % type_)

            current['dataset'] = dataset

        return graph, head

    @staticmethod
    def write_pipeline(pipeline, path):
        # force encoding and newline to produce same files on different OSes
        # this should be used by DVC later, which checks file hashes
        with open(path, 'w', encoding='utf-8', newline='') as f:
            yaml.safe_dump(pipeline, f)

    @staticmethod
    def read_pipeline(path):
        with open(path) as f:
            return yaml.safe_load(f)

    def make_dataset(self, target):
        if len(self._data) == 1 and self.MAIN_TARGET in self._data:
            raise DatumaroError("Can't create dataset from an empty project.")

        target = self._normalize_target(target)

        pipeline = self.make_pipeline(target)
        graph, head = self.apply_pipeline(pipeline)
        return graph.nodes[head]['dataset']

    def _normalize_target(self, target):
        if '.' not in target:
            real_target = self._make_target_name(target, self[target].head.name)
        else:
            t, s = self._split_target_name(target)
            assert self[t].get_stage(s), target
            real_target = target
        return real_target

    @classmethod
    def pipeline_sources(cls, pipeline):
        sources = set()
        for item in pipeline:
            if item['config']['type'] == BuildStageType.source.name:
                s, _ = cls._split_target_name(item['name'])
                sources.add(s)
        return list(sources)

    def build(self, target, out_dir=None, force=False, reset=True):
        def _rpath(p):
            return osp.relpath(p, self._project.config.project_dir)

        def _source_dvc_path(source):
            return _rpath(self._project.vcs.dvc_filepath(source))

        def _reset_sources(sources):
            for source in sources:
                dvc_path = _source_dvc_path(source)
                project_dir = self._project.config.project_dir
                repo = self._project.vcs.dvc.repo
                stage = repo.stage.load_file(osp.join(project_dir, dvc_path))[0]
                try:
                    logs = None
                    with repo.lock, catch_logs('dvc') as logs:
                        stage.frozen = False
                        stage.run(force=True, no_commit=True)
                except Exception:
                    if logs:
                        log.debug(logs.getvalue())
                    raise

        def _restore_sources(sources):
            if not self._project.vcs.has_commits() or not sources:
                return
            self._project.vcs.git.checkout(None,
                [_source_dvc_path(s) for s in sources])
            self._project.sources.checkout(sources)

        _is_modified = partial(self._project.vcs.dvc.check_stage_status,
            status='modified')


        if not self._project.vcs.writeable:
            raise VcsError("Can't build project in read-only or detached mode")

        if '.' in target:
            raw_target, target_stage = self._split_target_name(target)
        else:
            raw_target = target
            target_stage = None

        if raw_target not in self:
            raise KeyError("Unknown target '%s'" % raw_target)

        if target_stage and target_stage != self[raw_target].head.name:
            # build is not inplace, need to generate or ask output dir
            inplace = False
        else:
            inplace = not out_dir

        if inplace:
            if target == self.MAIN_TARGET:
                out_dir = osp.join(self._project.config.project_dir,
                    self._project.config.build_dir)
            elif target == raw_target:
                out_dir = self._project.sources.data_dir(target)

        if not out_dir:
            raise Exception("Output directory is not specified.")

        pipeline = self.make_pipeline(target)
        related_sources = self.pipeline_sources(pipeline)

        if not force:
            if inplace:
                stages = [_source_dvc_path(s) for s in related_sources]
                status = self._project.vcs.dvc.status(stages)
                for stage, source in zip(stages, related_sources):
                    if _is_modified(status, stage):
                        raise VcsError("Can't build when there are "
                            "uncommitted changes in the source '%s'" % source)
            elif osp.isdir(out_dir) and os.listdir(out_dir):
                raise Exception("Can't build when output directory"
                    "is not empty")

        try:
            if reset:
                _reset_sources(related_sources)

            self.run_pipeline(pipeline, out_dir=out_dir)

            if raw_target != self.MAIN_TARGET:
                related_sources.remove(raw_target)

        finally:
            if reset:
                _restore_sources(related_sources)

    def run_pipeline(self, pipeline, out_dir):
        graph, head = self.apply_pipeline(pipeline)
        head_node = graph.nodes[head]
        raw_target, _ = self._split_target_name(head)

        dataset = head_node['dataset']
        dst_format = DEFAULT_FORMAT
        options = {'save_images': True}
        if raw_target in self._project.sources:
            dst_format = self._project.sources[raw_target].format
        elif head_node['config']['type'] == BuildStageType.convert.name:
            dst_format = head_node['config'].kind
            options.update(head_node['config'].params)
        dataset.export(format=dst_format, save_dir=out_dir, **options)


class GitWrapper:
    @staticmethod
    def import_module():
        import git
        return git

    try:
        module = import_module.__func__()
    except ImportError:
        module = None

    def _git_dir(self):
        return osp.join(self._project_dir, '.git')

    def __init__(self, project_dir):
        self._project_dir = project_dir
        self.repo = None

        if osp.isdir(project_dir) and osp.isdir(self._git_dir()):
            self.repo = self.module.Repo(project_dir)

    @property
    def initialized(self):
        return self.repo is not None

    def init(self):
        if self.initialized:
            return

        repo = self.module.Repo.init(path=self._project_dir)
        repo.config_writer() \
            .set_value("user", "name", "User") \
            .set_value("user", "email", "<>") \
            .release()
        # gitpython does not support init, use git directly
        repo.git.init()

        self.repo = repo

    @property
    def refs(self) -> List[str]:
        return [t.name for t in self.repo.refs]

    @property
    def tags(self) -> List[str]:
        return [t.name for t in self.repo.tags]

    def push(self, remote=None):
        args = [remote] if remote else []
        remote = self.repo.remote(*args)
        branch = self.repo.head.ref.name
        if not self.repo.head.ref.tracking_branch():
            self.repo.git.push('--set-upstream', remote, branch)
        else:
            remote.push(branch)

    def pull(self, remote=None):
        args = [remote] if remote else []
        return self.repo.remote(*args).pull()

    def check_updates(self, remote=None) -> List[str]:
        args = [remote] if remote else []
        remote = self.repo.remote(*args)
        prev_refs = {r.name: r.commit.hexsha for r in remote.refs}
        remote.update()
        new_refs = {r.name: r.commit.hexsha for r in remote.refs}
        updated_refs = [(prev_refs.get(n), new_refs.get(n))
            for n, _ in (set(prev_refs.items()) ^ set(new_refs.items()))]
        return updated_refs

    def fetch(self, remote=None):
        args = [remote] if remote else []
        self.repo.remote(*args).fetch()

    def tag(self, name):
        self.repo.create_tag(name)

    def checkout(self, ref=None, paths=None):
        args = []
        if ref:
            args.append(ref)
        if paths:
            args.append('--')
            args.extend(paths)
        self.repo.git.checkout(*args)

    def add(self, paths, all=False): # pylint: disable=redefined-builtin
        if not all:
            paths = [
                p2 for p in paths
                for p2 in glob(osp.join(p, '**', '*'), recursive=True)
                if osp.isdir(p)
            ] + [
                p for p in paths if osp.isfile(p)
            ]
            self.repo.index.add(paths)
        else:
            self.repo.git.add(all=True)

    def commit(self, message):
        self.repo.index.commit(message)

    def status(self):
        # R[everse] flag is needed for index to HEAD comparison
        # to avoid inversed output in gitpython, which adds this flag
        # git diff --cached HEAD [not not R]
        diff = self.repo.index.diff(R=True)
        return {
            osp.relpath(d.a_rawpath.decode(), self._project_dir): d.change_type
            for d in diff
        }

    def list_remotes(self):
        return { r.name: r.url for r in self.repo.remotes }

    def add_remote(self, name, url):
        self.repo.create_remote(name, url)

    def remove_remote(self, name):
        self.repo.delete_remote(name)

    def is_ref(self, rev):
        try:
            self.repo.commit(rev)
            return True
        except (ValueError, self.module.exc.BadName):
            return False

    def has_commits(self):
        return self.is_ref('HEAD')

    IgnoreMode = Enum('IgnoreMode', ['rewrite', 'append', 'remove'])

    def ignore(self, paths, filepath=None, mode=None):
        repo_root = self._project_dir

        def _make_ignored_path(path):
            path = osp.join(repo_root, osp.normpath(path))
            assert path.startswith(repo_root), path
            return osp.relpath(path, repo_root)

        IgnoreMode = self.IgnoreMode
        mode = parse_str_enum_value(mode, IgnoreMode, IgnoreMode.append)

        if not filepath:
            filepath = '.gitignore'
        filepath = osp.abspath(osp.join(repo_root, filepath))
        assert filepath.startswith(repo_root), filepath

        paths = [_make_ignored_path(p) for p in paths]

        openmode = 'r+'
        if not osp.isfile(filepath):
            openmode = 'w+' # r+ cannot create, w+ truncates
        with open(filepath, openmode) as f:
            if mode in {IgnoreMode.append, IgnoreMode.remove}:
                paths_to_write = set(
                    line.split('#', maxsplit=1)[0] \
                        .split('/', maxsplit=1)[-1].strip()
                    for line in f
                )
                f.seek(0)
            else:
                paths_to_write = set()

            if mode in {IgnoreMode.append, IgnoreMode.rewrite}:
                paths_to_write.update(paths)
            elif mode == IgnoreMode.remove:
                for p in paths:
                    paths_to_write.discard(p)

            paths_to_write = sorted(p for p in paths_to_write if p)
            f.write('# The file is autogenerated by Datumaro\n')
            f.writelines('\n'.join(paths_to_write))
            f.truncate()

    def show(self, path, rev=None):
        return self.repo.git.show('%s:%s' % (rev or '', path))

class DvcWrapper:
    @staticmethod
    def import_module():
        import dvc
        import dvc.repo
        import dvc.main
        return dvc

    try:
        module = import_module.__func__()
    except ImportError:
        module = None

    def _dvc_dir(self):
        return osp.join(self._project_dir, '.dvc')

    class DvcError(Exception):
        pass

    def __init__(self, project_dir):
        self._project_dir = project_dir
        self._repo = None

        if osp.isdir(project_dir) and osp.isdir(self._dvc_dir()):
            with logging_disabled():
                self._repo = self.module.repo.Repo(project_dir)

    @property
    def initialized(self):
        return self._repo is not None

    @property
    def repo(self):
        self._repo = self.module.repo.Repo(self._project_dir)
        return self._repo

    def init(self):
        if self.initialized:
            return

        with logging_disabled():
            self._repo = self.module.repo.Repo.init(self._project_dir)

    def push(self, targets=None, remote=None):
        args = ['push']
        if remote:
            args.append('--remote')
            args.append(remote)
        if targets:
            args.extend(targets)
        self._exec(args)

    def pull(self, targets=None, remote=None):
        args = ['pull']
        if remote:
            args.append('--remote')
            args.append(remote)
        if targets:
            args.extend(targets)
        self._exec(args)

    def check_updates(self, targets=None, remote=None):
        args = ['fetch'] # no other way now?
        if remote:
            args.append('--remote')
            args.append(remote)
        if targets:
            args.extend(targets)
        self._exec(args)

    def fetch(self, targets=None, remote=None):
        args = ['fetch']
        if remote:
            args.append('--remote')
            args.append(remote)
        if targets:
            args.extend(targets)
        self._exec(args)

    def import_repo(self, url, path, out=None, dvc_path=None, rev=None,
            download=True):
        args = ['import']
        if dvc_path:
            args.append('--file')
            args.append(dvc_path)
            os.makedirs(osp.dirname(dvc_path), exist_ok=True)
        if rev:
            args.append('--rev')
            args.append(rev)
        if out:
            args.append('-o')
            args.append(out)
        if not download:
            args.append('--no-exec')
        args.append(url)
        args.append(path)
        self._exec(args)

    def import_url(self, url, out=None, dvc_path=None, download=True):
        args = ['import-url']
        if dvc_path:
            args.append('--file')
            args.append(dvc_path)
            os.makedirs(osp.dirname(dvc_path), exist_ok=True)
        if not download:
            args.append('--no-exec')
        args.append(url)
        if out:
            args.append(out)
        self._exec(args)

    def update_imports(self, targets=None, rev=None):
        args = ['update']
        if rev:
            args.append('--rev')
            args.append(rev)
        if targets:
            args.extend(targets)
        self._exec(args)

    def checkout(self, targets=None):
        args = ['checkout']
        if targets:
            args.extend(targets)
        self._exec(args)

    def add(self, paths, dvc_path=None):
        args = ['add']
        if dvc_path:
            args.append('--file')
            args.append(dvc_path)
            os.makedirs(osp.dirname(dvc_path), exist_ok=True)
        if paths:
            if isinstance(paths, str):
                args.append(paths)
            else:
                args.extend(paths)
        self._exec(args)

    def remove(self, paths, outs=False):
        args = ['remove']
        if outs:
            args.append('--outs')
        if paths:
            if isinstance(paths, str):
                args.append(paths)
            else:
                args.extend(paths)
        self._exec(args)

    def commit(self, paths):
        args = ['commit', '--recursive', '--force']
        if paths:
            args.extend(paths)
        self._exec(args)

    def add_remote(self, name, config):
        self._exec(['remote', 'add', name, config['url']])

    def remove_remote(self, name):
        self._exec(['remote', 'remove', name])

    def list_remotes(self):
        out = self._exec(['remote', 'list'])
        return dict(line.split() for line in out.split('\n') if line)

    def get_default_remote(self):
        out = self._exec(['remote', 'default'])
        if out == 'No default remote set' or 1 < len(out.split()):
            return None
        return out

    def set_default_remote(self, name):
        assert name and 1 == len(name.split()), "Invalid remote name '%s'" % name
        self._exec(['remote', 'default', name])

    def list_stages(self):
        return set(s.addressing for s in self.repo.stages)

    def run(self, name, cmd, deps=None, outs=None, force=False):
        args = ['run', '-n', name]
        if force:
            args.append('--force')
        for d in deps:
            args.append('-d')
            args.append(d)
        for o in outs:
            args.append('--outs')
            args.append(o)
        args.extend(cmd)
        self._exec(args, hide_output=False)

    def repro(self, targets=None, force=False, pull=False):
        args = ['repro']
        if force:
            args.append('--force')
        if pull:
            args.append('--pull')
        if targets:
            args.extend(targets)
        self._exec(args)

    def status(self, targets=None):
        args = ['status', '--show-json']
        if targets:
            args.extend(targets)
        out = self._exec(args).splitlines()[-1]
        return json.loads(out)

    @staticmethod
    def check_stage_status(data, stage, status):
        assert status in {'deleted', 'modified'}
        return status in [s
            for d in data.get(stage, []) if 'changed outs' in d
            for co in d.values()
            for s in co.values()
        ]

    def _exec(self, args, hide_output=True, answer_on_input='y'):
        contexts = ExitStack()

        args = ['--cd', self._project_dir] + args
        contexts.callback(os.chdir, os.getcwd()) # restore cd after DVC

        if answer_on_input is not None:
            def _input(*args): return answer_on_input
            contexts.enter_context(unittest.mock.patch(
                'dvc.prompt.input', new=_input))

        log.debug("Calling DVC main with args: %s", args)

        logs = contexts.enter_context(catch_logs('dvc'))

        with contexts:
            retcode = self.module.main.main(args)

        logs = logs.getvalue()
        if retcode != 0:
            raise self.DvcError(logs)
        if not hide_output:
            print(logs)
        return logs

class ProjectVcs:
    def __init__(self, project: 'Project', readonly: bool = False):
        self._project = project
        self.readonly = readonly

        if not project.config.detached:
            try:
                GitWrapper.import_module()
                DvcWrapper.import_module()
                self._git = GitWrapper(project.config.project_dir)
                self._dvc = DvcWrapper(project.config.project_dir)
            except ImportError as e:
                log.warning("Failed to init VCS for the project: %s", e)
                self._git = None
                self._dvc = None

        self._remotes = ProjectRemotes(self)
        self._repos = ProjectRepositories(self)

    @property
    def git(self) -> GitWrapper:
        if not self._git:
            raise ImportError("Git is not available.")
        return self._git

    @property
    def dvc(self) -> DvcWrapper:
        if not self._dvc:
            raise ImportError("DVC is not available.")
        return self._dvc

    @property
    def detached(self):
        return self._project.config.detached or not self._git or not self._dvc

    @property
    def writeable(self):
        return not self.detached and not self.readonly and self.initialized

    @property
    def readable(self):
        return not self.detached and self.initialized

    @property
    def initialized(self):
        return not self.detached and \
            self.git.initialized and self.dvc.initialized

    @property
    def remotes(self) -> ProjectRemotes:
        return self._remotes

    @property
    def repositories(self) -> ProjectRepositories:
        return self._repos

    @property
    def refs(self) -> List[str]:
        return self.git.refs

    @property
    def tags(self) -> List[str]:
        return self.git.tags

    def push(self, remote: Union[None, str] = None):
        if not self.writeable:
            raise ReadonlyProjectError(
                "Can't push in a detached or read-only repository")

        self.dvc.push()
        self.git.push(remote=remote)

    def pull(self, remote=None):
        if not self.writeable:
            raise ReadonlyProjectError(
                "Can't pull in a detached or read-only repository")

        # order matters
        self.git.pull(remote=remote)
        self.dvc.pull()

    def check_updates(self,
            targets: Union[None, str, List[str]] = None) -> List[str]:
        if not self.writeable:
            raise ReadonlyProjectError(
                "Can't check updates in a detached or read-only repository")

        updated_refs = self.git.check_updates()
        updated_remotes = self.remotes.check_updates(targets)
        return updated_refs, updated_remotes

    def fetch(self, remote: Union[None, str] = None):
        if not self.writeable:
            raise ReadonlyProjectError(
                "Can't fetch in a detached or read-only repository")

        self.git.fetch(remote=remote)
        self.dvc.fetch()

    def tag(self, name: str):
        if not self.writeable:
            raise ReadonlyProjectError(
                "Can't tag in a detached or read-only repository")

        self.git.tag(name)

    def checkout(self, rev: Union[None, str] = None,
            targets: Union[None, str, List[str]] = None):
        if not self.writeable:
            raise ReadonlyProjectError(
                "Can't checkout in a detached or read-only repository")

        # order matters
        targets = targets or []
        dvc_paths = [self.dvc_filepath(t) for t in targets]
        self.git.checkout(rev, dvc_paths)

        if not targets:
            self._dvc.checkout()
        else:
            sources = [t for t in targets if t in self._project.sources]
            if sources:
                self._project.sources.checkout(sources)

            models = [t for t in targets if t in self._project.models]
            if models:
                self._project.models.checkout(models)

    def add(self, paths: List[str]):
        if not self.writeable:
            raise ReadonlyProjectError(
                "Can't track files in a detached or read-only repository")

        if not paths:
            raise ValueError("Expected at least one file path to add")
        for p in paths:
            self.dvc.add(p, dvc_path=self.dvc_aux_path(osp.basename(p)))
        self.ensure_gitignored()

    def commit(self, paths: Union[None, List[str]], message):
        if not self.writeable:
            raise ReadonlyProjectError(
                "Can't commit in a detached or read-only repository")

        # order matters
        if not paths:
            paths = glob(
                osp.join(self._project.config.project_dir, '**', '*.dvc'),
                recursive=True)
        self.dvc.commit(paths)
        self.ensure_gitignored()

        project_dir = self._project.config.project_dir
        env_dir = self._project.config.env_dir
        self.git.add([
            osp.join(project_dir, env_dir),
            osp.join(project_dir, '.dvc', 'config'),
            osp.join(project_dir, '.dvc', '.gitignore'),
            osp.join(project_dir, '.gitignore'),
            osp.join(project_dir, '.dvcignore'),
        ] + list(self.git.status()))
        self.git.commit(message)

    def init(self):
        if self.readonly or self.detached:
            raise ReadonlyProjectError(
                "Can't init in a detached or read-only repository")

        # order matters
        self.git.init()
        self.dvc.init()
        os.makedirs(self.dvc_aux_dir(), exist_ok=True)

    def status(self) -> Dict:
        if not self.readable:
            raise DetachedProjectError(
                "Can't check status in a detached repository")

        # check status of files and remotes
        uncomitted = {}
        uncomitted.update(self.git.status())
        uncomitted.update(self.dvc.status())
        return uncomitted

    def ensure_gitignored(self, paths: Union[None, str, List[str]] = None):
        if not self.writeable:
            raise ReadonlyProjectError(
                "Can't update a detached or read-only repository")

        if paths is None:
            paths = [self._project.sources.data_dir(source)
                    for source in self._project.sources] + \
                [self._project.config.build_dir]
        self.git.ignore(paths, mode='append')

    def dvc_aux_dir(self) -> str:
        return osp.join(self._project.config.project_dir,
            self._project.config.env_dir,
            self._project.config.dvc_aux_dir)

    def dvc_filepath(self, target: str) -> str:
        return osp.join(self.dvc_aux_dir(), target + '.dvc')

    def is_ref(self, ref: str) -> bool:
        if not self.readable:
            raise DetachedProjectError("Can't read in a detached repository")

        return self.git.is_ref(ref)

    def has_commits(self) -> bool:
        return self.git.has_commits()

class Project:
    @classmethod
    def import_from(cls, path: str, dataset_format: Optional[str] = None,
            env: Optional[Environment] = None, **format_options) -> 'Project':
        if env is None:
            env = Environment()

        if not dataset_format:
            matches = env.detect_dataset(path)
            if not matches:
                raise DatumaroError(
                    "Failed to detect dataset format automatically")
            if 1 < len(matches):
                raise DatumaroError(
                    "Failed to detect dataset format automatically:"
                    " data matches more than one format: %s" % \
                    ', '.join(matches))
            dataset_format = matches[0]
        elif not env.is_format_known(dataset_format):
            raise KeyError("Unknown format '%s'. To make it "
                "available, add the corresponding Extractor implementation "
                "to the environment" % dataset_format)

        project = Project(env=env)
        project.sources.add('source', {
            'url': path,
            'format': dataset_format,
            'options': format_options,
        })
        return project

    @classmethod
    def generate(cls, save_dir: str,
            config: Optional[Config] = None) -> 'Project':
        config = Config(config)
        config.project_dir = save_dir
        project = Project(config)
        project.save(save_dir)
        return project

    @classmethod
    def load(cls, path: str) -> 'Project':
        path = osp.abspath(path)
        config_path = osp.join(path, PROJECT_DEFAULT_CONFIG.env_dir,
            PROJECT_DEFAULT_CONFIG.project_filename)
        config = Config.parse(config_path)
        config.project_dir = path
        config.project_filename = osp.basename(config_path)
        return Project(config)

    @error_rollback('on_error', implicit=True)
    def save(self, save_dir: Union[None, str] = None):
        config = self.config
        if save_dir and config.project_dir and save_dir != config.project_dir:
            raise NotImplementedError("Can't copy or resave project "
                "to another directory.")

        config.project_dir = save_dir or config.project_dir
        assert config.project_dir
        project_dir = config.project_dir
        save_dir = osp.join(project_dir, config.env_dir)

        if not osp.exists(project_dir):
            on_error.do(shutil.rmtree, project_dir, ignore_errors=True)
        if not osp.exists(save_dir):
            on_error.do(shutil.rmtree, save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)

        config.dump(osp.join(save_dir, config.project_filename))

        if self.vcs.detached:
            return

        if not self.vcs.initialized and not self.vcs.readonly:
            self._vcs = ProjectVcs(self)
            self.vcs.init()
        if self.vcs.writeable:
            self.vcs.ensure_gitignored()
            self.vcs.git.add([
                osp.join(project_dir, config.env_dir),
                osp.join(project_dir, '.dvc', 'config'),
                osp.join(project_dir, '.dvc', '.gitignore'),
                osp.join(project_dir, '.gitignore'),
                osp.join(project_dir, '.dvcignore'),
            ])

    def __init__(self, config: Optional[Config] = None,
            env: Optional[Environment] = None):
        self._config = self._read_config(config)
        if env is None:
            env = Environment(self._config)
        elif config is not None:
            raise ValueError("env can only be provided when no config provided")
        self._env = env
        self._vcs = ProjectVcs(self)
        self._sources = ProjectSources(self)
        self._models = ProjectModels(self)
        self._build_targets = ProjectBuildTargets(self)

    @property
    def sources(self) -> ProjectSources:
        return self._sources

    @property
    def models(self) -> ProjectModels:
        return self._models

    @property
    def build_targets(self) -> ProjectBuildTargets:
        return self._build_targets

    @property
    def vcs(self) -> ProjectVcs:
        return self._vcs

    @property
    def config(self) -> Config:
        return self._config

    @property
    def env(self) -> Environment:
        return self._env

    def make_dataset(self,
            target: Union[None, str, List[str]] = None) -> Dataset:
        if target is None:
            target = 'project'
        return self.build_targets.make_dataset(target)

    def publish(self):
        # build + tag + push?
        raise NotImplementedError()

    def build(self, target: Union[None, str, List[str]] = None,
            force: bool = False, out_dir: Union[None, str] = None):
        if target is None:
            target = 'project'
        return self.build_targets.build(target, force=force, out_dir=out_dir)

    @classmethod
    def _read_config_v1(cls, config):
        config = Config(config)
        config.remove('subsets')
        config.remove('format_version')

        config = cls._read_config_v2(config)
        name = generate_next_name(list(config.sources), 'source',
            sep='-', default='1')
        config.sources[name] = {
            'url': config.dataset_dir,
            'format': DEFAULT_FORMAT,
        }
        return config

    @classmethod
    def _read_config_v2(cls, config):
        return Config(config,
            fallback=PROJECT_DEFAULT_CONFIG, schema=PROJECT_SCHEMA)

    @classmethod
    def _read_config(cls, config):
        if config:
            version = config.get('format_version')
        else:
            version = None
        if version == 1:
            return cls._read_config_v1(config)
        elif version in {None, 2}:
            return cls._read_config_v2(config)
        else:
            raise ValueError("Unknown project config file format version '%s'. "
                "The only known are: 1, 2" % version)

def merge_projects(a, b, strategy: MergeStrategy = None):
    raise NotImplementedError()

def compare_projects(a, b, **options):
    raise NotImplementedError()


def load_project_as_dataset(url):
    return Project.load(url).make_dataset()
