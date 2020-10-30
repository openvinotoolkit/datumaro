# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import logging as log
import networkx as nx
import os
import os.path as osp
import shutil
import urllib.parse
import yaml
from enum import Enum
from glob import glob
from typing import List

from datumaro.components.config import Config
from datumaro.components.config_model import (PROJECT_DEFAULT_CONFIG,
    PROJECT_SCHEMA, BuildStage)
from datumaro.components.environment import Environment
from datumaro.components.dataset import Dataset
from datumaro.components.launcher import ModelTransform
from datumaro.util import make_file_name, find, generate_next_name
from datumaro.util.log_utils import logging_disabled, catch_logs


def load_project_as_dataset(url):
    # symbol forward declaration
    raise NotImplementedError()

class ProjectSourceDataset(Dataset):
    def __init__(self, project, source):
        super().__init__()

        self._project = project
        self._env = project.env
        env = project.env

        config = project.sources[source]
        self._config = config
        self._local_dir = project.sources.source_dir(source)

        dataset = Dataset.from_extractors(
            env.make_extractor(config.format,
                self._local_dir, **config.options))

        self._subsets = dataset._subsets
        self._categories = dataset._categories

    def save(self, save_dir=None, **kwargs):
        if save_dir is None:
            save_dir = self._local_dir
        super().export(self.config.format, save_dir=save_dir, **kwargs)

    @Dataset.env.getter
    def env(self):
        return self._project.env

    @property
    def config(self):
        return self._config

    def apply_model(self, model, batch_size=1):
        # NOTE: probably this function should be in the ViewModel layer
        if isinstance(model, str):
            launcher = self._project.make_executable_model(model)

        return self.transform(ModelTransform, launcher=launcher,
            batch_size=batch_size)

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

        value = self._data.set(name, value)

        if value.type == 'url':
            self._vcs.dvc.add_remote(name, value)
        return value

    def remove(self, name, force=False):
        try:
            self._vcs.dvc.remove_remote(name)
        except Exception:
            if not force:
                raise

    @classmethod
    def validate_url(cls, url):
        url_parts = urllib.parse.urlsplit(url)
        if url_parts.scheme not in cls.SUPPORTED_PROTOCOLS:
            raise NotImplementedError(
                "Invalid remote '%s': scheme '%s' is not supported, the only"
                "available are: %s" % \
                (url, url_parts.scheme, ', '.join(cls.SUPPORTED_PROTOCOLS))
            )
        return url_parts

class _RemotesProxy(CrudProxy):
    def __init__(self, project, config_field):
        self._project = project
        self._field = config_field

    @CrudProxy._data.getter
    def _data(self):
        return self._project.config[self._field]

    def pull(self, names=None):
        if not self._project.vcs.writeable:
            raise Exception("Can't pull in read-only repository")

        if not names:
            names = []

        for name in names:
            if name and name not in self:
                raise KeyError("Unknown source '%s'" % name)

        self._project.vcs.dvc.update_imports(
            [self.aux_path(name) for name in names])

    def fetch(self, names=None):
        if not self._project.vcs.readable:
            raise Exception("Can't fetch in read-only repository")

        if not names:
            names = []

        for name in names:
            if name and name not in self:
                raise KeyError("Unknown source '%s'" % name)

        self._project.vcs.dvc.fetch(
            [self.aux_path(name) for name in names])

    def checkout(self, names=None):
        if not self._project.vcs.writeable:
            raise Exception("Can't checkout in read-only repository")

        if not names:
            names = []

        for name in names:
            if name and name not in self:
                raise KeyError("Unknown source '%s'" % name)

        self._project.vcs.dvc.checkout(
            [self.aux_path(name) for name in names])

    def push(self, names=None):
        if not self._project.vcs.writeable:
            raise Exception("Can't push in read-only repository")

        if not names:
            names = []

        for name in names:
            if name and name not in self:
                raise KeyError("Unknown source '%s'" % name)

        self._project.vcs.dvc.push([self.aux_path(name) for name in names])

    def add(self, name, value):
        return self._data.set(name, value)

    @classmethod
    def _make_remote_name(cls, name):
        raise NotImplementedError("Should be implemented in a subclass")

    @classmethod
    def _validate_url(cls, url):
        url_parts = ProjectRemotes.validate_url(url)
        if not url_parts.path:
            raise ValueError("URL must contain path, url: '%s'" % url)
        return url_parts

    def aux_dir(self):
        return osp.join(self._project.config.project_dir,
            self._project.config.env_dir, self._project.config.dvc_aux_dir)

    def aux_path(self, name):
        return osp.join(self.aux_dir(), name + '.dvc')

class ProjectModels(_RemotesProxy):
    def __init__(self, project):
        super().__init__(project, 'models')

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError("Unknown model '%s'" % name)

    def model_dir(self, name):
        return osp.join(self.config.env_dir, self.config.models_dir, name)

    def make_executable_model(self, name):
        model = self.get_model(name)
        return self.env.make_launcher(model.launcher,
            **model.options, model_dir=self.local_model_dir(name))

class ProjectSources(_RemotesProxy):
    def __init__(self, project):
        super().__init__(project, 'sources')

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError("Unknown source '%s'" % name)

    def add(self, name, value):
        if name in self:
            raise Exception("Source '%s' already exists" % name)

        url_parts = self._validate_url(value['url'])

        if url_parts.scheme == 'remote':
            remote_name = url_parts.netloc
            remote_conf = self._project.vcs.remotes[remote_name]
            path = url_parts.path
            if path.startswith('/') and len(path) != 1:
                path = path[1:]
        elif self._project.vcs.writeable:
            remote_name = self._make_remote_name(name)
            remote_conf = self._project.vcs.remotes.add(remote_name, {
                'url': value['url'],
                'type': 'url',
            })
            path = '' # all goes to the remote
        else:
            raise Exception("Can't update read-only project")

        source_dir = osp.relpath(self.source_dir(name),
            self._project.config.project_dir)

        if self._project.vcs.writeable:
            os.makedirs(source_dir, exist_ok=True)
            os.makedirs(self.aux_dir(), exist_ok=True)

            aux_path = self.aux_path(name)
            if remote_conf.type == 'url':
                self._project.vcs.dvc.import_url(
                    urllib.parse.urlunsplit(('remote', remote_name, path, '', '')),
                    out=source_dir, dvc_path=aux_path, download=False)
            elif remote_conf.type == 'git':
                self._project.vcs.dvc.import_(remote_conf.url, path=path,
                    out=source_dir, dvc_path=aux_path)
            else:
                raise Exception("Unknown remote type '%s'" % remote_conf.type)

        value['url'] = source_dir
        value['remote'] = remote_name
        value = super().add(name, value)

        self._project.build_targets.add_target(name)

        return value

    def remove(self, name, force=False, keep_data=True):
        """Force - ignores errors and tries to wipe remaining data"""

        if name not in self._data and not force:
            raise KeyError("Unknown source '%s'" % name)

        self._project.build_targets.remove_target(name)
        self._data.remove(name)

        if not self._project.vcs.writeable:
            return

        if force and not keep_data:
            source_dir = self.source_dir(name)
            if osp.isdir(source_dir):
                shutil.rmtree(source_dir, ignore_errors=True)

        aux_file = self.aux_path(name)
        if osp.isfile(aux_file):
            try:
                self._project.vcs.dvc.remove(aux_file, outs=not keep_data)
            except Exception:
                if force:
                    os.remove(aux_file)
                else:
                    raise

        self._project.vcs.remotes.remove(name, force=force)

    @classmethod
    def _make_remote_name(cls, name):
        return name

    def make_dataset(self, name):
        return ProjectSourceDataset(self._project, name)

    def source_dir(self, name):
        return osp.join(self._project.config.project_dir, name)

    def aux_path(self, name):
        return osp.join(self.aux_dir(), name + '.dvc')


BuildStageType = Enum('BuildStageType',
    ['source', 'project', 'transform', 'filter', 'convert'])

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
        return data

    def add_target(self, name):
        return self._data.set(name, {
            'stages': [
                BuildStage({
                    'name': self.BASE_STAGE,
                    'type': BuildStageType.source.name,
                }),
            ]
        })

    def add_stage(self, target, value, prev=None, name=None):
        target = self._data[target]

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
            value['name'] = generate_next_name((s.name for s in target.stages),
                value['type'], sep='-')

        target.stages.insert(prev_stage + 1, BuildStage(value))
        return value

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
            if type_ in {BuildStageType.transform, BuildStageType.filter}:
                if type_ == BuildStageType.transform:
                    kind = current['config'].kind
                    try:
                        transform = self._project.env.transforms.get(kind)
                    except KeyError:
                        raise CliException("Unknown transform '%s'" % kind)

                    # fused, unless required multiple times
                    dataset = transform(*parent_datasets, **params)
                elif type_ == BuildStageType.filter:
                    if 1 < len(parent_datasets):
                        dataset = Dataset.from_extractors(*parent_datasets)
                    else:
                        dataset = parent_datasets[0]
                    dataset = dataset.filter(**params)

                if 1 < graph.out_degree(current_name):
                    # if multiple consumers, avoid reapplying the whole stack
                    # for each one
                    dataset = Dataset.from_extractors(*parent_datasets)

            elif type_ == BuildStageType.source:
                source, _ = self._split_target_name(current_name)
                dataset = self._project.sources.make_dataset(source)

            elif type_ == BuildStageType.project:
                if 1 < len(parent_datasets):
                    dataset = Dataset.from_extractors(*parent_datasets)
                else:
                    dataset = parent_datasets[0]

            else:
                raise NotImplementedError("Unknown stage type '%s'" % type_)

            if head == current_name and not isinstance(dataset, Dataset):
                dataset = Dataset.from_extractors(dataset)
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

    def build(self, target, force=False):
        if not self._project.vcs.readable:
            raise Exception("Can't build a project without VCS support")

        def _rpath(p):
            return osp.relpath(p, self._project.config.project_dir)

        try:
            status = self._project.vcs.dvc.status([make_file_name(target)])
            if not status:
                return
        except Exception:
            pass

        pipeline = self.make_pipeline(target)
        sources = self.pipeline_sources(pipeline)
        self._project.sources.checkout(sources)

        out_dir = osp.join(self._project.config.project_dir,
            self._project.config.build_dir)

        pipeline_file = _rpath(self.generate_pipeline(target))

        out_dir = _rpath(out_dir)
        self._project.vcs.dvc.run(
            cmd=['datum', 'apply', '--build', '--overwrite',
                '-o', out_dir, pipeline_file],
            deps=[pipeline_file] + [
                self._project.sources.source_dir(s) for s in sources],
            outs=[out_dir], name=target, force=True)

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
        self.repo.remote(*args).push()

    def pull(self, remote=None):
        args = [remote] if remote else []
        self.repo.remote(*args).pull()

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

    def checkout(self, ref):
        self.repo.git.checkout(ref)

    def add(self, paths, all=False):
        if not all:
            paths = [p for p in paths if osp.isfile(p)]
            self.repo.index.add(paths)
        else:
            self.repo.git.add(all=True)

    def commit(self, message):
        self.repo.index.commit(message)

    def status(self):
        diff = self.repo.index.diff(None)
        return {
            osp.relpath(d.a_rawpath.decode(), self._project_dir): d.change_type
            for d in diff
        }

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

    def __init__(self, project_dir):
        self._project_dir = project_dir
        self.repo = None

        if osp.isdir(project_dir) and osp.isdir(self._dvc_dir()):
            with logging_disabled():
                self.repo = self.module.repo.Repo(project_dir)

    @property
    def initialized(self):
        return self.repo is not None

    def init(self):
        if self.initialized:
            return

        with logging_disabled():
            self.repo = self.module.repo.Repo.init(self._project_dir)

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

    def import_(self, url, path, out=None, dvc_path=None, rev=None):
        args = ['import']
        if dvc_path:
            args.append('--file')
            args.append(dvc_path)
        if rev:
            args.append('--rev')
            args.append(rev)
        if out:
            args.append('-o')
            args.append(out)
        args.append(url)
        args.append(path)
        self._exec(args)

    def import_url(self, url, out=None, dvc_path=None, download=True):
        args = ['import-url']
        if dvc_path:
            args.append('--file')
            args.append(dvc_path)
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

    def add(self, paths):
        args = ['add']
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
        args = ['commit', '--recursive']
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

    def repro(self, targets=None, force=False):
        args = ['repro']
        if force:
            args.append('--force')
        if targets:
            args.extend(targets)
        self._exec(args)

    def status(self, targets=None):
        args = ['status', '--show-json']
        if targets:
            args.extend(targets)
        out = self._exec(args).splitlines()[-1]
        return json.loads(out)

    def _exec(self, args, hide_output=True):
        log.debug("Calling DVC main with args: %s", args)

        with catch_logs('dvc') as logs:
            retcode = self.module.main.main(args)
        logs = logs.getvalue()
        if retcode != 0:
            raise Exception(logs)
        if not hide_output:
            print(logs)
        return logs

class ProjectVcs:
    def __init__(self, project, readonly=False):
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
    def refs(self) -> List[str]:
        return self.git.refs

    @property
    def tags(self) -> List[str]:
        return self.git.tags

    def push(self, remote=None):
        self.dvc.push()
        self.git.push(remote=remote)

    def pull(self, remote=None):
        # order matters
        self.git.pull(remote=remote)
        self.dvc.pull()

    def check_updates(self, targets=None) -> List[str]:
        updated_refs = self.git.check_updates()
        updated_remotes = self.remotes.check_updates(targets)
        return updated_refs, updated_remotes

    def fetch(self, remote=None):
        self.git.fetch(remote=remote)
        self.dvc.fetch()

    def tag(self, name):
        self.git.tag(name)

    def checkout(self, rev=None, targets=None):
        # order matters
        if rev:
            self.git.checkout(rev)
        self.dvc.checkout(targets)

    def add(self, paths):
        if not paths:
            paths = [self._project.config.project_dir]
        self.dvc.add(paths)

    def commit(self, paths, message):
        # order matters
        if not paths:
            paths = glob(
                osp.join(self._project.config.project_dir, '**', '*.dvc'),
                recursive=True)
        self.dvc.commit(paths)
        self.git.add(None, all=True)
        self.git.commit(message)

    def init(self):
        # order matters
        self.git.init()
        self.dvc.init()

    def status(self):
        # check status of files and remotes
        uncomitted = {}
        uncomitted.update(self.git.status())
        uncomitted.update(self.dvc.status())
        return uncomitted

class Project:
    @classmethod
    def import_from(cls, path, dataset_format=None, env=None, **format_options):
        if env is None:
            env = Environment()
        if not dataset_format:
            dataset_format = env.detect_dataset(path)
        importer = env.make_importer(dataset_format)
        return importer(path, **format_options)

    @classmethod
    def generate(cls, save_dir, config=None):
        config = Config(config)
        config.project_dir = save_dir
        project = Project(config)
        project.save(save_dir)
        return project

    @classmethod
    def load(cls, path):
        path = osp.abspath(path)
        config_path = osp.join(path, PROJECT_DEFAULT_CONFIG.env_dir,
            PROJECT_DEFAULT_CONFIG.project_filename)
        config = Config.parse(config_path)
        config.project_dir = path
        config.project_filename = osp.basename(config_path)
        return Project(config)

    def save(self, save_dir=None):
        config = self.config

        if save_dir is None:
            assert config.project_dir
            project_dir = config.project_dir
        else:
            project_dir = save_dir

        env_dir = osp.join(project_dir, config.env_dir)
        save_dir = osp.abspath(env_dir)

        project_dir_existed = osp.exists(project_dir)
        env_dir_existed = osp.exists(env_dir)
        try:
            os.makedirs(save_dir, exist_ok=True)

            config_path = osp.join(save_dir, config.project_filename)
            config.dump(config_path)

            if not self.vcs.detached and not self.vcs.readonly and \
                    not self.vcs.initialized:
                self._vcs = ProjectVcs(self) # TODO: handle different save_dir
                self.vcs.init()
            if self.vcs.writeable:
                self.vcs.git.add([
                    osp.join(project_dir, config.env_dir),
                    osp.join(project_dir, '.dvc', 'config'),
                    osp.join(project_dir, '.gitignore'),
                ])
        except BaseException:
            if not env_dir_existed:
                shutil.rmtree(save_dir, ignore_errors=True)
            if not project_dir_existed:
                shutil.rmtree(project_dir, ignore_errors=True)
            raise

    def __init__(self, config=None):
        self._config = Config(config,
            fallback=PROJECT_DEFAULT_CONFIG, schema=PROJECT_SCHEMA)
        self._env = Environment(self.config)
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

    def make_dataset(self, target=None) -> Dataset:
        if target is None:
            target = 'project'
        return self.build_targets.make_dataset(target)

    def publish(self):
        # build + tag + push?
        raise NotImplementedError()

    def build(self, target=None, force=False):
        if target is None:
            target = 'project'
        return self.build_targets.build(target, force=force)

def merge_projects(a, b, strategy: MergeStrategy = None):
    raise NotImplementedError()

def compare_projects(a, b, **options):
    raise NotImplementedError()


# pylint: disable=function-redefined
def load_project_as_dataset(url):
    # implement the function declared above
    return Project.load(url).make_dataset()
# pylint: enable=function-redefined