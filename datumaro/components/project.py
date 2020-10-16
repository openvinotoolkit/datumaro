# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

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
    PROJECT_SCHEMA)
from datumaro.components.environment import Environment
from datumaro.components.dataset import Dataset
from datumaro.components.launcher import ModelTransform
from datumaro.util import make_file_name, find, generate_next_name


def load_project_as_dataset(url):
    # symbol forward declaration
    raise NotImplementedError()

class ProjectSourceDataset(Dataset):
    def __init__(self, project, source):
        super().__init__()

        self._project = project
        self._env = project.env

        config = project.sources[source]
        self._config = config
        self._local_dir = project.sources.source_dir(source)

        dataset = Dataset.from_extractors(
            env.make_extractor(config.format, config.url, **config.options))

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

    def add(self, name, value):
        raise NotImplementedError()

    def remove(self, name):
        raise NotImplementedError()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, name):
        return self._data[name]

    def get(self, name, default=None):
        return self._data.get(name, default)

    def __iter__(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __contains__(self, name):
        return name in self._data

class ProjectRemotes(CrudProxy):
    SUPPORTED_PROTOCOLS = {'local', 'remote', 'git', 's3', 'ssh'}

    def __init__(self, project_vcs):
        self._vcs = project_vcs

    def check_updates(self, name=None):
        self._vcs.dvc.check_remote(name)

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
        return self._vcs.dvc.list_remotes()

    def add(self, name, value):
        if osp.isdir(value['url']):
            value['url'] = 'local://' + value['url']
        self.validate_url(value['url'])

        return self._vcs.dvc.add_remote(name, value)

    def remove(self, name):
        self._vcs.dvc.remove_remote(name)

    @classmethod
    def validate_url(cls, url):
        url_parts = urllib.parse.urlsplit(value['url'])
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

    def check_updates(self, name=None):
        if self._project.vcs.readable:
            self._project.vcs.remotes.check_remote(name)

    def pull(self, name=None):
        if self._project.vcs.writeable:
            self._project.vcs.dvc.update_imports(name)

    def add(self, name, value):
        return self._data.set(name, value)

    def remove(self, name):
        self._data.remove(name)
        if self._project.vcs.writeable:
            self._project.vcs.remotes.remove(name)

    @classmethod
    def _validate_url(cls, url):
        url_parts = ProjectRemotes._validate_url(url)
        if not url_parts.path:
            raise ValueError("URL must contain path, url: '%s'" % url)
        return url_parts

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
            if not remote_name in self._project.vcs.remotes:
                raise Exception("Can't find remote '%s'" % remote_name)
        elif self._project.vcs.writeable:
            remote_name = self._make_remote_name(name)
            self._project.vcs.remotes.add(remote_name, { 'url': value['url'] })

        if self._project.vcs.writeable:
            self._project.vcs.dvc.import_url(urllib.parse.urlunsplit(
                url_parts._replace(scheme='remote', netloc=remote_name)
            ), out=self.source_dir(name))

        value['url'] = osp.normpath(url_parts.path)
        value['remote'] = remote_name
        value = super().add(name, value)

        self._project.build_targets.add_target(name)

        return value

    def remove(self, name):
        super().remove(name)
        self._project.build_targets.remove_target(name)

    @classmethod
    def _make_remote_name(cls, name):
        return 'source-%s'

    def make_dataset(self, name):
        return ProjectSourceDataset(self._project, name)

    def source_dir(self, name):
        return osp.join(self._project.config.project_dir, name)


BuildStageType = Enum('BuildStageType',
    ['source', 'project', 'export', 'transform', 'filter'])

class ProjectBuildTargets(CrudProxy):
    def __init__(self, project):
        self._project = project

    @CrudProxy._data.getter
    def _data(self):
        return self._project.config.build_targets

    def add_target(self, name):
        return self._data.set(name, {
            'stages': [
                {
                    'name': self.BASE_STAGE,
                    'type': BuildStageType.source.name,
                }
            ]
        })

    def add_stage(self, target, value, prev=None, name=None):
        if prev is None:
            prev = self.BASE_STAGE

        target = self._data[target]

        prev_stage = find(enumerate(target.stages), lambda e: e[1].name == prev)
        if prev_stage is None:
            raise KeyError("Can't find stage '%s'" % prev)
        prev_stage = prev_stage[0]

        name = value.get('name') or name
        if not name:
            value['name'] = generate_next_name((s.name for s in target.stages),
                value['type'], sep='-')

        target.stages.insert(prev_stage + 1, value)
        return value

    def remove_target(self, name):
        assert name != self.MAIN_TARGET, "Can't remove the main target"
        self._data.remove(name)

    def remove_stage(self, target, name):
        assert name not in {self.BASE_STAGE}, "Can't remove a default stage"

        target = self._data[target]
        prev_stage = find(enumerate(target.stages), lambda e: e[1].name == prev)
        if prev_stage is None:
            raise KeyError("Can't find stage '%s'" % prev)
        target.stages.remove()

    MAIN_TARGET = 'project'
    BASE_STAGE = 'root'
    def _get_build_graph(self):
        graph = nx.DiGraph()
        for target_name, target in self._data.items():
            if target_name == self.MAIN_TARGET:
                # main target combines all the others
                prev_stages = [t.head.name for t in self._data
                    if t != self.MAIN_TARGET]
            else:
                prev_stages = [t.head.name for t in target.parents]

            for stage_name, stage in target.stages:
                stage_name = self._make_target_name(target_name, stage_name)
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
        def _is_root(g, n):
            return g.in_degree(n) == 0

        full_graph = self._get_build_graph()

        target_parents = set()
        visited = set()
        to_visit = set()
        while to_visit:
            current = to_visit.pop()
            visited.add(current)

            for pred in current.predecessors:
                if _is_root(full_graph, pred):
                    target_parents.add(pred)
                elif pred not in visited:
                    to_visit.add(pred)

        target_parents.add(target)

        return full_graph.subgraph(target_parents)

    def _get_target_config(self, name):
        """Returns a target or stage description"""
        target, stage = self._split_target_name(name)
        target_config = self._data[target]
        stage_config = find(target_config.stages, lambda s: s.name == stage)
        return stage_config

    def make_pipeline(self, target):
        # a subgraph with all the target dependencies
        target_subgraph = self._get_target_subgraph(target)
        pipeline = []
        for node_name, node in target_subgraph.nodes.items():
            entry = {
                'name': node_name,
                'parents': list(target_subgraph.predecessors(node_name)),
                'type': node.type,
                'params': node.parameters,
            }
            pipeline.append(entry)
        return pipeline

    def generate_pipeline(self, target):
        pipeline = self.make_pipeline(target)
        path = self._project.config.pipelines_dir
        dir_existed = osp.isdir(path)
        try:
            os.makedirs(path, exist_ok=True)
            self.write_pipeline(pipeline,
                osp.join(path, make_file_name(target) + '.yml'))
        except BaseException:
            if not dir_existed:
                shutil.rmtree(path, ignore_errors=True)
            raise

        return path

    @classmethod
    def _read_pipeline_graph(cls, pipeline):
        graph = nx.DiGraph()
        for entry in pipeline:
            target_name = entry['name']
            target = {
                'type': entry['type'],
                'params': entry['params'],
            }
            parents = entry['parents']

            graph.add_node(target_name, config=target)
            for prev_stage in parents:
                graph.add_edge(prev_stage, target_name)

        return graph

    def apply_pipeline(self, pipeline):
        def _is_root(g, n):
            return g.in_degree(n) == 0

        graph = self._read_pipeline_graph(pipeline)

        head = None

        # Use DFS to traverse the graph and initialize nodes from roots to tops
        to_visit = list()
        while to_visit:
            current_name = to_visit.pop()
            current = graph.nodes[current_name]

            assert current_name.get('dataset') is None

            if _is_root(graph, current_name):
                assert current['config']['type'] == BuildStageType.source.name, \
                    "A pipeline root can only be a source"
                source, _ = self._split_target_name(current_name)
                current['dataset'] = self._project.sources.make_dataset(source)
                continue

            parents_uninitialized = []
            parent_datasets = []
            for p_name, parent in graph.nodes[current_name].predecessors.items():
                dataset = parent.get('dataset')
                if dataset is None:
                    parents_uninitialized.append(p_name)
                else:
                    parent_datasets.append(dataset)

            if parents_uninitialized:
                to_visit.append(current_name)
                to_visit.extend(parents_uninitialized)
                continue

            type_ = BuildStageType[current['config']['type']]
            params = current['config']['params']
            if type_ in {BuildStageType.transform, BuildStageType.filter}:
                if type_ == BuildStageType.transform:
                    name = current['config']['kind']
                    try:
                        transform = project.env.transforms.get(name)
                    except KeyError:
                        raise CliException("Unknown transform '%s'" % name)

                    # fused, unless required multiple times
                    dataset = transform(*parent_datasets, **params)
                elif type_ == BuildStageType.filter:
                    if 1 < len(parent_datasets):
                        dataset = Dataset.from_extractors(parent_datasets)
                    else:
                        dataset = parent_datasets[0]
                    dataset = dataset.filter(**params)

                if 1 < graph.out_degree(current_name):
                    # if multiple consumers, avoid reapplying the whole stack
                    # for each one
                    dataset = Dataset.from_extractors(parent_datasets)

            elif type_ == BuildStageType.export:
                name = current['config']['kind']
                try:
                    converter = project.env.converters.get(name)
                except KeyError:
                    raise CliException("Unknown converter '%s'" % name)

                if 1 < len(parent_datasets):
                    dataset = Dataset.from_extractors(parent_datasets)
                else:
                    dataset = parent_datasets[0]
                converter(dataset, **params)

            else:
                raise NotImplementedError("Unknown stage type '%s'")

            current['dataset'] = dataset

            if graph.out_degree(current_name) == 0:
                assert head is None, "A pipeline can have only one " \
                    "main target, but it has at least 2: %s, %s" % \
                    (head, current_name)
                head = current_name

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
        assert target in self

        pipeline = self.make_pipeline(target)
        graph, head = self.apply_pipeline(pipeline)
        return graph.nodes[head].dataset

    def build(self, target):
        assert target in self

        if not self._project.vcs.readable:
            raise Exception("Can't build a project without VCS support")

        def _rpath(p):
            return osp.relpath(p, self._project.config.project_dir)

        pipeline_file = _rpath(self.generate_pipeline(target))
        out_dir = _rpath(osp.join(self._project.config.build_dir,
            make_file_name(target)))
        self._project.vcs.dvc.run(cmd=['datum', 'process', pipeline_file],
            deps=[pipeline_file], outs=[out_dir], name=target)
        self._project.vcs.dvc.repro(target)

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
        self.repo.head.reference = self.repo.refs[ref]

    def add(self, paths):
        self.repo.index.add(paths)

    def commit(self, message):
        self.repo.index.commit(message)

class DvcWrapper:
    @staticmethod
    def import_module():
        import dvc
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
            self.repo = self.module.repo.Repo(project_dir)

    @property
    def initialized(self):
        return self.repo is not None

    def init(self):
        if self.initialized:
            return

        self.repo = self.module.repo.Repo.init(self._project_dir)

    def push(self, remote=None):
        self.repo.push(remote=remote)

    def pull(self, remote=None):
        self.repo.pull(remote=remote)

    def check_updates(self, remote=None):
        self.fetch(remote) # can't be done other way now?

    def fetch(self, remote=None):
        self.repo.fetch(remote=remote)

    def import_url(self, url, out=None):
        self.repo.import_url(url, out=out)
        return self.repo.scm.files_to_track

    def update_imports(self, targets=None):
        self.repo.update(targets)

    def checkout(self):
        self.repo.checkout()

    def add(self, paths):
        self.repo.add(paths)
        return self.repo.scm.files_to_track

    def commit(self, paths):
        self.repo.commit(paths, recursive=True)
        return self.repo.scm.files_to_track

    def add_remote(self, name, config=None):
        self.module.main.main('remote', 'add', name)

    def remove_remote(self, name):
        raise NotImplementedError()

    def run(self, name, cmd, deps=None, outs=None):
        args = []
        for d in deps:
            args.append('-d')
            args.append(d)
        for o in outs:
            args.append('-o')
            args.append(o)
        self.module.main.main('run', '-n', name, *args, *cmd)

    def repro(self, target):
        self.module.main.main('repro', target)

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
        return self.git.refs()

    @property
    def tags(self) -> List[str]:
        return self.git.tags()

    def push(self, remote=None):
        self.dvc.push()
        self.git.push()

    def pull(self, remote=None):
        # order matters
        self.git.pull()
        self.dvc.pull()

    def check_updates(self, remote=None) -> List[str]:
        updated_refs = self.git.check_updates()
        return updated_refs

    def fetch(self, remote=None):
        self.git.fetch()
        self.dvc.fetch()

    def tag(self, name):
        self.git.tag(name)

    def checkout(self, ref):
        # order matters
        self.git.checkout(ref)
        self.dvc.checkout()

    def add(self, paths):
        if not paths:
            paths = [self._project.config.project_dir]
        updated_paths = self.dvc.add(paths)
        self.git.add(updated_paths)

    def commit(self, paths, message):
        # order matters
        if not paths:
            paths = glob(
                osp.join(self._project.config.project_dir, '**', '*.dvc'),
                recursive=True)
        self.dvc.commit(paths)
        self.git.commit(message)

    def init(self):
        # order matters
        self.git.init()
        self.dvc.init()

    def status(self):
        # check status of the files and remotes
        raise NotImplementedError()

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
        except BaseException:
            if not env_dir_existed:
                shutil.rmtree(save_dir, ignore_errors=True)
            if not project_dir_existed:
                shutil.rmtree(project_dir, ignore_errors=True)
            raise

        if not self.vcs.detached and not self.vcs.readonly and \
                not self.vcs.initialized:
            self._vcs = ProjectVcs(self)
            self.vcs.init()

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

    def make_dataset(self, target=None) -> ProjectDataset:
        if target is None:
            target = 'project'
        return self.build_targets.make_dataset(target)

    def publish(self):
        # build + tag + push?
        raise NotImplementedError()

    def build(self, target=None):
        if target is None:
            target = 'project'
        return self.build_targets.build(target)

def merge_projects(a, b, strategy: MergeStrategy = None):
    raise NotImplementedError()

def compare_projects(a, b, **options):
    raise NotImplementedError()


# pylint: disable=function-redefined
def load_project_as_dataset(url):
    # implement the function declared above
    return Project.load(url).make_dataset()
# pylint: enable=function-redefined