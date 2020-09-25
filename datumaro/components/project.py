# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import inspect
import logging as log
import os
import os.path as osp
import shutil
from collections import defaultdict
from enum import Enum
from glob import glob
from typing import List, Union

from datumaro.components.config import DEFAULT_FORMAT, Config
from datumaro.components.config_model import (PROJECT_DEFAULT_CONFIG,
    PROJECT_SCHEMA)
from datumaro.components.dataset import Dataset
from datumaro.components.launcher import ModelTransform
from datumaro.util.os_util import import_foreign_module


class Registry:
    def __init__(self, config=None, item_type=None):
        self.item_type = item_type

        self.items = {}

        if config is not None:
            self.load(config)

    def load(self, config):
        pass

    def register(self, name, value):
        if self.item_type:
            value = self.item_type(value)
        self.items[name] = value
        return value

    def unregister(self, name):
        return self.items.pop(name, None)

    def get(self, key):
        return self.items[key] # returns a class / ctor

class PluginRegistry(Registry):
    def __init__(self, config=None, builtin=None, local=None):
        super().__init__(config)

        from datumaro.components.cli_plugin import CliPlugin

        if builtin is not None:
            for v in builtin:
                k = CliPlugin._get_name(v)
                self.register(k, v)
        if local is not None:
            for v in local:
                k = CliPlugin._get_name(v)
                self.register(k, v)

def load_project_as_dataset(url):
    # symbol forward declaration
    raise NotImplementedError()

class Environment:
    _builtin_plugins = None
    PROJECT_EXTRACTOR_NAME = 'datumaro_project'

    def __init__(self, config=None):
        config = Config(config,
            fallback=PROJECT_DEFAULT_CONFIG, schema=PROJECT_SCHEMA)

        env_dir = osp.join(config.project_dir, config.env_dir)
        builtin = self._load_builtin_plugins()
        custom = self._load_plugins2(osp.join(env_dir, config.plugins_dir))
        select = lambda seq, t: [e for e in seq if issubclass(e, t)]
        from datumaro.components.extractor import Transform
        from datumaro.components.extractor import SourceExtractor
        from datumaro.components.extractor import Importer
        from datumaro.components.converter import Converter
        from datumaro.components.launcher import Launcher
        self.extractors = PluginRegistry(
            builtin=select(builtin, SourceExtractor),
            local=select(custom, SourceExtractor)
        )
        self.extractors.register(self.PROJECT_EXTRACTOR_NAME,
            load_project_as_dataset)

        self.importers = PluginRegistry(
            builtin=select(builtin, Importer),
            local=select(custom, Importer)
        )
        self.launchers = PluginRegistry(
            builtin=select(builtin, Launcher),
            local=select(custom, Launcher)
        )
        self.converters = PluginRegistry(
            builtin=select(builtin, Converter),
            local=select(custom, Converter)
        )
        self.transforms = PluginRegistry(
            builtin=select(builtin, Transform),
            local=select(custom, Transform)
        )

    @staticmethod
    def _find_plugins(plugins_dir):
        plugins = []
        if not osp.exists(plugins_dir):
            return plugins

        for plugin_name in os.listdir(plugins_dir):
            p = osp.join(plugins_dir, plugin_name)
            if osp.isfile(p) and p.endswith('.py'):
                plugins.append((plugins_dir, plugin_name, None))
            elif osp.isdir(p):
                plugins += [(plugins_dir,
                        osp.splitext(plugin_name)[0] + '.' + osp.basename(p),
                        osp.splitext(plugin_name)[0]
                    )
                    for p in glob(osp.join(p, '*.py'))]
        return plugins

    @classmethod
    def _import_module(cls, module_dir, module_name, types, package=None):
        module = import_foreign_module(osp.splitext(module_name)[0], module_dir,
            package=package)

        exports = []
        if hasattr(module, 'exports'):
            exports = module.exports
        else:
            for symbol in dir(module):
                if symbol.startswith('_'):
                    continue
                exports.append(getattr(module, symbol))

        exports = [s for s in exports
            if inspect.isclass(s) and issubclass(s, types) and not s in types]

        return exports

    @classmethod
    def _load_plugins(cls, plugins_dir, types):
        types = tuple(types)

        plugins = cls._find_plugins(plugins_dir)

        all_exports = []
        for module_dir, module_name, package in plugins:
            try:
                exports = cls._import_module(module_dir, module_name, types,
                    package)
            except Exception as e:
                module_search_error = ImportError
                try:
                    module_search_error = ModuleNotFoundError # python 3.6+
                except NameError:
                    pass

                message = ["Failed to import module '%s': %s", module_name, e]
                if isinstance(e, module_search_error):
                    log.debug(*message)
                else:
                    log.warning(*message)
                continue

            log.debug("Imported the following symbols from %s: %s" % \
                (
                    module_name,
                    ', '.join(s.__name__ for s in exports)
                )
            )
            all_exports.extend(exports)

        return all_exports

    @classmethod
    def _load_builtin_plugins(cls):
        if not cls._builtin_plugins:
            plugins_dir = osp.join(
                __file__[: __file__.rfind(osp.join('datumaro', 'components'))],
                osp.join('datumaro', 'plugins')
            )
            assert osp.isdir(plugins_dir), plugins_dir
            cls._builtin_plugins = cls._load_plugins2(plugins_dir)
        return cls._builtin_plugins

    @classmethod
    def _load_plugins2(cls, plugins_dir):
        from datumaro.components.extractor import Transform
        from datumaro.components.extractor import Extractor
        from datumaro.components.extractor import Importer
        from datumaro.components.converter import Converter
        from datumaro.components.launcher import Launcher
        types = [Extractor, Converter, Importer, Launcher, Transform]

        return cls._load_plugins(plugins_dir, types)

    def make_extractor(self, name, *args, **kwargs):
        return self.extractors.get(name)(*args, **kwargs)

    def make_importer(self, name, *args, **kwargs):
        return self.importers.get(name)(*args, **kwargs)

    def make_launcher(self, name, *args, **kwargs):
        return self.launchers.get(name)(*args, **kwargs)

    def make_converter(self, name, *args, **kwargs):
        return self.converters.get(name)(*args, **kwargs)

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

    @property
    def env(self):
        return self._project.env

    @property
    def config(self):
        return self._project.config

    @property
    def sources(self):
        return self._sources

    def _save_branch_project(self, extractor, save_dir=None):
        extractor = Dataset.from_extractors(extractor) # apply lazy transforms

        # NOTE: probably this function should be in the ViewModel layer
        save_dir = osp.abspath(save_dir)
        if save_dir:
            dst_project = Project()
        else:
            if not self.config.project_dir:
                raise Exception("Either a save directory or a project "
                    "directory should be specified")
            save_dir = self.config.project_dir

            dst_project = Project(Config(self.config))
            dst_project.config.remove('project_dir')
            dst_project.config.remove('sources')
        dst_project.config.project_name = osp.basename(save_dir)

        dst_dataset = dst_project.make_dataset()
        dst_dataset.define_categories(extractor.categories())
        dst_dataset.update(extractor)

        dst_dataset.save(save_dir=save_dir, merge=True)

    def transform_project(self, method, save_dir=None, **method_kwargs):
        # NOTE: probably this function should be in the ViewModel layer
        if isinstance(method, str):
            method = self.env.make_transform(method)

        transformed = self.transform(method, **method_kwargs)
        self._save_branch_project(transformed, save_dir=save_dir)

    def apply_model(self, model, save_dir=None, batch_size=1):
        # NOTE: probably this function should be in the ViewModel layer
        if isinstance(model, str):
            launcher = self._project.make_executable_model(model)

        self.transform_project(ModelTransform, launcher=launcher,
            save_dir=save_dir, batch_size=batch_size)

    def export_project(self, save_dir, converter,
            filter_expr=None, filter_annotations=False, remove_empty=False):
        # NOTE: probably this function should be in the ViewModel layer
        dataset = self
        if filter_expr:
            dataset = dataset.filter(filter_expr,
                filter_annotations=filter_annotations,
                remove_empty=remove_empty)

        save_dir = osp.abspath(save_dir)
        save_dir_existed = osp.exists(save_dir)
        try:
            os.makedirs(save_dir, exist_ok=True)
            converter(dataset, save_dir)
        except BaseException:
            if not save_dir_existed:
                shutil.rmtree(save_dir)
            raise

    def filter_project(self, filter_expr, filter_annotations=False,
            save_dir=None, remove_empty=False):
        # NOTE: probably this function should be in the ViewModel layer
        dataset = self
        if filter_expr:
            dataset = dataset.filter(filter_expr,
                filter_annotations=filter_annotations,
                remove_empty=remove_empty)
        self._save_branch_project(dataset, save_dir=save_dir)

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

    def __iter__(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __contains__(self, name):
        return name in self._data

class ProjectRemotes(CrudProxy):
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

    @CrudProxy._data
    def _data(self):
        return self._vcs.dvc.list_remotes()

    def add(self, name, value):
        self._vcs.dvc.add_remote(name, value)

    def remove(self, name):
        self._vcs.dvc.remove_remote(name)

class _RemotesProxy(CrudProxy):
    def __init__(self, project, config_field):
        self._project = project
        self._field = config_field

    @CrudProxy._data
    def _data(self):
        return self._project.config[self._field]

    def check_updates(self, name=None):
        self._project.vcs.remotes.check_remote(name)

    def fetch(self, name=None):
        self._project.vcs.remotes.fetch_remote(name)

    def pull(self, name=None):
        self._project.vcs.remotes.pull_remote(name)

    def add(self, name, value):
        value = self._data.set(name, value)
        self._project.vcs.remotes.add(name, value)

    def remove(self, name):
        self._data.remove(name)
        self._project.vcs.remotes.remove(name)

class ProjectModels(_RemotesProxy):
    def __init__(self, project):
        super().__init__(project, 'models')

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError("Unknown model '%s'" % name)

class ProjectSources(_RemotesProxy):
    def __init__(self, project):
        super().__init__(project, 'sources')

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError("Unknown source '%s'" % name)

class GitWrapper:
    @staticmethod
    def import_module():
        import git
        return git

    try:
        module = import_module()
    except ImportError:
        module = None

    def _git_dir(self):
        return osp.join(self._project_dir, '.git')

    def __init__(self, project_dir):
        self._project_dir = project_dir
        self.repo = None

        if osp.isdir(project_dir) and osp.isdir(self._git_dir()):
            self.repo = self.module.Repo(project_dir)

    def init(self):
        assert self.repo is None
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
        module = import_module()
    except ImportError:
        module = None

    def _dvc_dir(self):
        return osp.join(self._project_dir, '.dvc')

    def __init__(self, project_dir):
        self._project_dir = project_dir
        self.repo = None

        if osp.isdir(project_dir) and osp.isdir(self._dvc_dir()):
            self.repo = self.module.repo.Repo(project_dir)

    def init(self):
        assert self.repo is None
        self.repo = self.module.repo.Repo.init(self._project_dir)

    def push(self, remote=None):
        self.repo.push(remote=remote)

    def pull(self, remote=None):
        self.repo.pull(remote=remote)

    def check_updates(self, remote=None):
        self.fetch(remote) # can't be done other way now?

    def fetch(self, remote=None):
        self.repo.fetch(remote=remote)

    def checkout(self):
        self.repo.checkout()

    def add(self, path):
        self.repo.add(path)
        return self.repo.scm.files_to_track

    def commit(self, message=None):
        return self.repo.scm.files_to_track

class ProjectVcs:
    def __init__(self, project, readonly=False):
        self._project = project
        self._readonly = readonly

        if not self._project.config.detached:
            try:
                GitWrapper.import_module()
                DvcWrapper.import_module()
                self._git = GitWrapper(project.config.project_dir)
                self._dvc = DvcWrapper(project.config.project_dir)
            except ImportError as e:
                log.warning("Failed to init versioning for the project: %s", e)
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
        return not self._detached and not self._readonly

    @property
    def readable(self):
        return not self._detached

    @property
    def remotes(self) -> ProjectRemotes:
        return self._remotes

    @property
    def refs(self) -> List[str]:
        return self.git.refs()

    @property
    def tags(self) -> List[str]:
        return self.git.tags()

    def push(self):
        self.dvc.push()
        self.git.push()

    def pull(self):
        # order matters
        self.git.pull()
        self.dvc.pull()

    def check_updates(self) -> List[str]:
        updated_refs = self.git.check_updates()
        return updated_refs

    def fetch(self):
        self.git.fetch()
        self.dvc.fetch()

    def tag(self, name):
        self.git.tag(name)

    def checkout(self, ref):
        # order matters
        self.git.checkout(ref)
        self.dvc.checkout()

    def add(self, path):
        updated_paths = self.dvc.add(path)
        self.git.add(updated_paths)

    def commit(self, message):
        # order matters
        self.dvc.commit()
        self.git.commit(message)

    def init(self):
        # order matters
        self.git.init()
        self.dvc.init()

class Project:
    @classmethod
    def import_from(cls, path, dataset_format=None, env=None, **format_options):
        if env is None:
            env = Environment()
        if not dataset_format:
            dataset_format = env.detect_dataset(path)
        importer = env.make_importer(dataset_format)
        return importer(path, **kwargs)

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

    def __init__(self, config=None):
        self._config = Config(config,
            fallback=PROJECT_DEFAULT_CONFIG, schema=PROJECT_SCHEMA)
        self._env = Environment(self.config)
        self._vcs = None
        if not self._config.detached and osp.isdir(self._config.project_dir):
            self._vcs = ProjectVcs(self)
        self._sources = ProjectSources(self)
        self._models = ProjectModels(self)
        self._dataset = None

    @property
    def sources(self) -> ProjectSources:
        return self._sources

    @property
    def models(self) -> ProjectModels:
        return self._models

    @property
    def vcs(self) -> Union[None, ProjectVcs]:
        return self._vcs

    @property
    def own_dataset(self) -> ProjectDataset:
        if self._dataset is None:
            self._dataset = ProjectDataset(self, only_own=True)
        return self._dataset

    @property
    def config(self) -> Config:
        return self._config

    @property
    def env(self) -> Environment:
        return self._env

    def make_dataset(self) -> ProjectDataset:
        return ProjectDataset(self)

    def publish(self):
        # build + tag + push?
        raise NotImplementedError()

    @property
    def build_targets(self):  # -> Dict-like proxy (CRUD)
        raise NotImplementedError()

    def build(self, target=None):
        raise NotImplementedError()


def merge_projects(a, b, strategy: MergeStrategy = None):
    raise NotImplementedError()

def compare_projects(a, b, **options):
    raise NotImplementedError()


# class Project:

#     def make_executable_model(self, name):
#         model = self.get_model(name)
#         return self.env.make_launcher(model.launcher,
#             **model.options, model_dir=self.local_model_dir(name))

#     def local_model_dir(self, model_name):
#         return osp.join(
#             self.config.env_dir, self.config.models_dir, model_name)

#     def local_source_dir(self, source_name):
#         return osp.join(self.config.sources_dir, source_name)

# pylint: disable=function-redefined
def load_project_as_dataset(url):
    # implement the function declared above
    return Project.load(url).make_dataset()
# pylint: enable=function-redefined