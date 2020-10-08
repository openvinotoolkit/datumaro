# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
import shutil
import urllib.parse
from collections import defaultdict
from enum import Enum
from glob import glob
from typing import List

from datumaro.components.config import DEFAULT_FORMAT, Config
from datumaro.components.config_model import (PROJECT_DEFAULT_CONFIG,
    PROJECT_SCHEMA)
from datumaro.components.environment import Environment
from datumaro.components.dataset import Dataset
from datumaro.components.launcher import ModelTransform


def load_project_as_dataset(url):
    # symbol forward declaration
    raise NotImplementedError()

class ProjectDataset(Dataset):
    def __init__(self, project, only_own=False):
        super().__init__()

        self._project = project
        config = self.config
        env = self.env

        sources = {}
        if not only_own:
            for s_name, source in config.sources.items():
                s_format = source.format or env.PROJECT_EXTRACTOR_NAME
                options = {}
                options.update(source.options)

                url = source.url
                if not source.url:
                    url = osp.join(config.project_dir, config.sources_dir, s_name)
                sources[s_name] = env.make_extractor(s_format, url, **options)
        self._sources = sources

        own_source = None
        own_source_dir = osp.join(config.project_dir, config.dataset_dir)
        if config.project_dir and osp.isdir(own_source_dir):
            log.disable(log.INFO)
            own_source = env.make_importer(DEFAULT_FORMAT)(own_source_dir) \
                .make_dataset()
            log.disable(log.NOTSET)

        # merge categories
        # TODO: implement properly with merging and annotations remapping
        categories = self._merge_categories(s.categories()
            for s in self._sources.values())
        # ovewrite with own categories
        if own_source is not None and (not categories or len(own_source) != 0):
            categories.update(own_source.categories())
        self._categories = categories

        # merge items
        subsets = defaultdict(lambda: self.Subset(self))
        for source_name, source in self._sources.items():
            log.debug("Loading '%s' source contents..." % source_name)
            for item in source:
                existing_item = subsets[item.subset].items.get(item.id)
                if existing_item is not None:
                    path = existing_item.path
                    if item.path != path:
                        path = None # NOTE: move to our own dataset
                    item = self._merge_items(existing_item, item, path=path)
                else:
                    s_config = config.sources[source_name]
                    if s_config and \
                            s_config.format != env.PROJECT_EXTRACTOR_NAME:
                        # NOTE: consider imported sources as our own dataset
                        path = None
                    else:
                        path = [source_name] + (item.path or [])
                    item = item.wrap(path=path)

                subsets[item.subset].items[item.id] = item

        # override with our items, fallback to existing images
        if own_source is not None:
            log.debug("Loading own dataset...")
            for item in own_source:
                existing_item = subsets[item.subset].items.get(item.id)
                if existing_item is not None:
                    item = item.wrap(path=None,
                        image=self._merge_images(existing_item, item))

                subsets[item.subset].items[item.id] = item

        # TODO: implement subset remapping when needed
        subsets_filter = config.subsets
        if len(subsets_filter) != 0:
            subsets = { k: v for k, v in subsets.items() if k in subsets_filter}
        self._subsets = dict(subsets)

        self._length = None

    def iterate_own(self):
        return self.select(lambda item: not item.path)

    def get(self, item_id, subset=None, path=None):
        if path:
            source = path[0]
            rest_path = path[1:]
            return self._sources[source].get(
                item_id=item_id, subset=subset, path=rest_path)
        return super().get(item_id, subset)

    def put(self, item, item_id=None, subset=None, path=None):
        if path is None:
            path = item.path

        if path:
            source = path[0]
            rest_path = path[1:]
            # TODO: reverse remapping
            self._sources[source].put(item,
                item_id=item_id, subset=subset, path=rest_path)

        if item_id is None:
            item_id = item.id
        if subset is None:
            subset = item.subset

        item = item.wrap(path=path)
        if subset not in self._subsets:
            self._subsets[subset] = self.Subset(self)
        self._subsets[subset].items[item_id] = item
        self._length = None

        return item

    def save(self, save_dir=None, merge=False, recursive=True,
            save_images=False):
        if save_dir is None:
            assert self.config.project_dir
            save_dir = self.config.project_dir
            project = self._project
        else:
            merge = True

        if merge:
            project = Project(Config(self.config))
            project.config.remove('sources')

        save_dir = osp.abspath(save_dir)
        dataset_save_dir = osp.join(save_dir, project.config.dataset_dir)

        converter_kwargs = {
            'save_images': save_images,
        }

        save_dir_existed = osp.exists(save_dir)
        try:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(dataset_save_dir, exist_ok=True)

            if merge:
                # merge and save the resulting dataset
                self.env.converters.get(DEFAULT_FORMAT).convert(
                    self, dataset_save_dir, **converter_kwargs)
            else:
                if recursive:
                    # children items should already be updated
                    # so we just save them recursively
                    for source in self._sources.values():
                        if isinstance(source, ProjectDataset):
                            source.save(**converter_kwargs)

                self.env.converters.get(DEFAULT_FORMAT).convert(
                    self.iterate_own(), dataset_save_dir, **converter_kwargs)

            project.save(save_dir)
        except BaseException:
            if not save_dir_existed and osp.isdir(save_dir):
                shutil.rmtree(save_dir, ignore_errors=True)
            raise

    @Dataset.env.getter
    def env(self):
        return self._project.env

    @property
    def config(self):
        return self._project.config

    @property
    def sources(self):
        return self._sources

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
    SUPPORTED_PROTOCOLS = {'local', 'remote', 'git'}

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

    # TODO:
    # def make_executable_model(self, name):
    #     model = self.get_model(name)
    #     return self.env.make_launcher(model.launcher,
    #         **model.options, model_dir=self.local_model_dir(name))

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
        raise NotImplementedError()

    def source_dir(self, name):
        return osp.join(self._project.config.project_dir, name)

class ProjectBuildTargets(CrudProxy):
    def __init__(self, project):
        self._project = project

    @CrudProxy._data.getter
    def _data(self):
        return self._project.config.build_targets

    def add_target(self, name):
        raise NotImplementedError()

    def add_stage(self, target, name, value, prev=None):
        raise NotImplementedError()

    def remove_target(self, name):
        raise NotImplementedError()

    def remove_stage(self, target, name):
        raise NotImplementedError()


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
        return ProjectDataset(self)

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