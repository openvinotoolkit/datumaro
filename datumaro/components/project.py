# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
import logging as log
import os
import os.path as osp
import shutil

from datumaro.components.config import Config
from datumaro.components.config_model import (Model, Source,
    PROJECT_DEFAULT_CONFIG, PROJECT_SCHEMA)
from datumaro.components.environment import Environment
from datumaro.components.launcher import ModelTransform
from datumaro.components.dataset import Dataset, DEFAULT_FORMAT


class ProjectDataset(Dataset):
    def __init__(self, project):
        super().__init__()

        self._project = project
        self._env = project.env
        config = self.config
        env = self.env

        sources = {}
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
    def config(self):
        return self._project.config

    @property
    def sources(self):
        return self._sources

    def _save_branch_project(self, extractor, save_dir=None):
        if not isinstance(extractor, Dataset):
            extractor = Dataset.from_extractors(
                extractor) # apply lazy transforms to avoid repeating traversals

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

class Project:
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

    @staticmethod
    def generate(save_dir, config=None):
        config = Config(config)
        config.project_dir = save_dir
        project = Project(config)
        project.save(save_dir)
        return project

    @staticmethod
    def import_from(path, dataset_format=None, env=None, **format_options):
        if env is None:
            env = Environment()

        if not dataset_format:
            matches = env.detect_dataset(path)
            if not matches:
                raise Exception("Failed to detect dataset format automatically")
            if 1 < len(matches):
                raise Exception("Failed to detect dataset format automatically:"
                    " data matches more than one format: %s" % \
                    ', '.join(matches))
            dataset_format = matches[0]
        elif not env.is_format_known(dataset_format):
            raise KeyError("Unknown dataset format '%s'" % dataset_format)

        if dataset_format in env.importers:
            project = env.make_importer(dataset_format)(path, **format_options)
        elif dataset_format in env.extractors:
            project = Project(env=env)
            project.add_source('source', {
                'url': path,
                'format': dataset_format,
                'options': format_options,
            })
        else:
            raise Exception("Unknown format '%s'. To make it "
                "available, add the corresponding Extractor implementation "
                "to the environment" % dataset_format)
        return project

    def __init__(self, config=None, env=None):
        self.config = Config(config,
            fallback=PROJECT_DEFAULT_CONFIG, schema=PROJECT_SCHEMA)
        if env is None:
            env = Environment(self.config)
        elif config is not None:
            raise ValueError("env can only be provided when no config provided")
        self.env = env

    def make_dataset(self):
        return ProjectDataset(self)

    def add_source(self, name, value=None):
        if value is None or isinstance(value, (dict, Config)):
            value = Source(value)
        self.config.sources[name] = value
        self.env.sources.register(name, value)

    def remove_source(self, name):
        self.config.sources.remove(name)
        self.env.sources.unregister(name)

    def get_source(self, name):
        try:
            return self.config.sources[name]
        except KeyError:
            raise KeyError("Source '%s' is not found" % name)

    def get_subsets(self):
        return self.config.subsets

    def set_subsets(self, value):
        if not value:
            self.config.remove('subsets')
        else:
            self.config.subsets = value

    def add_model(self, name, value=None):
        if value is None or isinstance(value, (dict, Config)):
            value = Model(value)
        self.env.register_model(name, value)
        self.config.models[name] = value

    def get_model(self, name):
        try:
            return self.env.models.get(name)
        except KeyError:
            raise KeyError("Model '%s' is not found" % name)

    def remove_model(self, name):
        self.config.models.remove(name)
        self.env.unregister_model(name)

    def make_executable_model(self, name):
        model = self.get_model(name)
        return self.env.make_launcher(model.launcher,
            **model.options, model_dir=self.local_model_dir(name))

    def make_source_project(self, name):
        source = self.get_source(name)

        config = Config(self.config)
        config.remove('sources')
        config.remove('subsets')
        project = Project(config)
        project.add_source(name, source)
        return project

    def local_model_dir(self, model_name):
        return osp.join(
            self.config.env_dir, self.config.models_dir, model_name)

    def local_source_dir(self, source_name):
        return osp.join(self.config.sources_dir, source_name)

def load_project_as_dataset(url):
    return Project.load(url).make_dataset()
