from unittest import TestCase
import os
import os.path as osp

import numpy as np

from datumaro.components.config import Config
from datumaro.components.config_model import Model, Source
from datumaro.components.dataset import DEFAULT_FORMAT, Dataset
from datumaro.components.extractor import (
    AnnotationType, DatasetItem, Extractor, Label, LabelCategories,
)
from datumaro.components.launcher import Launcher, ModelTransform
from datumaro.components.project import Environment, Project
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement


class ProjectTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_project_generate(self):
        src_config = Config({
            'project_name': 'test_project',
            'format_version': 1,
        })

        with TestDir() as test_dir:
            project_path = test_dir
            Project.generate(project_path, src_config)

            self.assertTrue(osp.isdir(project_path))

            result_config = Project.load(project_path).config
            self.assertEqual(
                src_config.project_name, result_config.project_name)
            self.assertEqual(
                src_config.format_version, result_config.format_version)

    @staticmethod
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_default_ctor_is_ok():
        Project()

    @staticmethod
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_empty_config_is_ok():
        Project(Config())

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_add_source(self):
        source_name = 'source'
        origin = Source({
            'url': 'path',
            'format': 'ext'
        })
        project = Project()

        project.add_source(source_name, origin)

        added = project.get_source(source_name)
        self.assertIsNotNone(added)
        self.assertEqual(added, origin)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_added_source_can_be_saved(self):
        source_name = 'source'
        origin = Source({
            'url': 'path',
        })
        project = Project()
        project.add_source(source_name, origin)

        saved = project.config

        self.assertEqual(origin, saved.sources[source_name])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_added_source_can_be_dumped(self):
        source_name = 'source'
        origin = Source({
            'url': 'path',
        })
        project = Project()
        project.add_source(source_name, origin)

        with TestDir() as test_dir:
            project.save(test_dir)

            loaded = Project.load(test_dir)
            loaded = loaded.get_source(source_name)
            self.assertEqual(origin, loaded)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_with_custom_importer(self):
        class TestImporter:
            def __call__(self, path, subset=None):
                return Project({
                    'project_filename': path,
                    'subsets': [ subset ]
                })

        path = 'path'
        importer_name = 'test_importer'

        env = Environment()
        env.importers.register(importer_name, TestImporter)

        project = Project.import_from(path, importer_name, env,
            subset='train')

        self.assertEqual(path, project.config.project_filename)
        self.assertListEqual(['train'], project.config.subsets)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_dump_added_model(self):
        model_name = 'model'

        project = Project()
        saved = Model({ 'launcher': 'name' })
        project.add_model(model_name, saved)

        with TestDir() as test_dir:
            project.save(test_dir)

            loaded = Project.load(test_dir)
            loaded = loaded.get_model(model_name)
            self.assertEqual(saved, loaded)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_have_project_source(self):
        with TestDir() as test_dir:
            Project.generate(test_dir)

            project2 = Project()
            project2.add_source('project1', {
                'url': test_dir,
            })
            dataset = project2.make_dataset()

            self.assertTrue('project1' in dataset.sources)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_batch_launch_custom_model(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=i, subset='train', image=np.array([i]))
            for i in range(5)
        ], categories=['label'])

        class TestLauncher(Launcher):
            def launch(self, inputs):
                for i, inp in enumerate(inputs):
                    yield [ Label(0, attributes={'idx': i, 'data': inp.item()}) ]

        model_name = 'model'
        launcher_name = 'custom_launcher'

        project = Project()
        project.env.launchers.register(launcher_name, TestLauncher)
        project.add_model(model_name, { 'launcher': launcher_name })
        model = project.make_executable_model(model_name)

        batch_size = 3
        executor = ModelTransform(dataset, model, batch_size=batch_size)

        for item in executor:
            self.assertEqual(1, len(item.annotations))
            self.assertEqual(int(item.id) % batch_size,
                item.annotations[0].attributes['idx'])
            self.assertEqual(int(item.id),
                item.annotations[0].attributes['data'])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_do_transform_with_custom_model(self):
        class TestExtractorSrc(Extractor):
            def __iter__(self):
                for i in range(2):
                    yield DatasetItem(id=i, image=np.ones([2, 2, 3]) * i,
                        annotations=[Label(i)])

            def categories(self):
                label_cat = LabelCategories()
                label_cat.add('0')
                label_cat.add('1')
                return { AnnotationType.label: label_cat }

        class TestLauncher(Launcher):
            def launch(self, inputs):
                for inp in inputs:
                    yield [ Label(inp[0, 0, 0]) ]

        class TestExtractorDst(Extractor):
            def __init__(self, url):
                super().__init__()
                self.items = [osp.join(url, p) for p in sorted(os.listdir(url))]

            def __iter__(self):
                for path in self.items:
                    with open(path, 'r') as f:
                        index = osp.splitext(osp.basename(path))[0]
                        label = int(f.readline().strip())
                        yield DatasetItem(id=index, annotations=[Label(label)])

        model_name = 'model'
        launcher_name = 'custom_launcher'
        extractor_name = 'custom_extractor'

        project = Project()
        project.env.launchers.register(launcher_name, TestLauncher)
        project.env.extractors.register(extractor_name, TestExtractorSrc)
        project.add_model(model_name, { 'launcher': launcher_name })
        project.add_source('source', { 'format': extractor_name })

        with TestDir() as test_dir:
            project.make_dataset().apply_model(model=model_name,
                save_dir=test_dir)

            result = Project.load(test_dir)
            result.env.extractors.register(extractor_name, TestExtractorDst)
            it = iter(result.make_dataset())
            item1 = next(it)
            item2 = next(it)
            self.assertEqual(0, item1.annotations[0].label)
            self.assertEqual(1, item2.annotations[0].label)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_source_datasets_can_be_merged(self):
        class TestExtractor(Extractor):
            def __init__(self, url, n=0, s=0):
                super().__init__(length=n)
                self.n = n
                self.s = s

            def __iter__(self):
                for i in range(self.n):
                    yield DatasetItem(id=self.s + i, subset='train')

        e_name1 = 'e1'
        e_name2 = 'e2'
        n1 = 2
        n2 = 4

        project = Project()
        project.env.extractors.register(e_name1, lambda p: TestExtractor(p, n=n1))
        project.env.extractors.register(e_name2, lambda p: TestExtractor(p, n=n2, s=n1))
        project.add_source('source1', { 'format': e_name1 })
        project.add_source('source2', { 'format': e_name2 })

        dataset = project.make_dataset()

        self.assertEqual(n1 + n2, len(dataset))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_merge_different_categories(self):
        class TestExtractor1(Extractor):
            def __iter__(self):
                return iter([])

            def categories(self):
                return { AnnotationType.label:
                    LabelCategories.from_iterable(['a', 'b']) }

        class TestExtractor2(Extractor):
            def __iter__(self):
                return iter([])

            def categories(self):
                return { AnnotationType.label:
                    LabelCategories.from_iterable(['b', 'a']) }

        e_name1 = 'e1'
        e_name2 = 'e2'

        project = Project()
        project.env.extractors.register(e_name1, TestExtractor1)
        project.env.extractors.register(e_name2, TestExtractor2)
        project.add_source('source1', { 'format': e_name1 })
        project.add_source('source2', { 'format': e_name2 })

        with self.assertRaisesRegex(Exception, "different categories"):
            project.make_dataset()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_project_filter_can_be_applied(self):
        class TestExtractor(Extractor):
            def __iter__(self):
                for i in range(10):
                    yield DatasetItem(id=i, subset='train')

        e_type = 'type'
        project = Project()
        project.env.extractors.register(e_type, TestExtractor)
        project.add_source('source', { 'format': e_type })

        dataset = project.make_dataset().filter('/item[id < 5]')

        self.assertEqual(5, len(dataset))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_own_dataset(self):
        with TestDir() as test_dir:
            src_project = Project()
            src_dataset = src_project.make_dataset()
            item = DatasetItem(id=1)
            src_dataset.put(item)
            src_dataset.save(test_dir)

            loaded_project = Project.load(test_dir)
            loaded_dataset = loaded_project.make_dataset()

            self.assertEqual(list(src_dataset), list(loaded_dataset))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_project_own_dataset_can_be_modified(self):
        project = Project()
        dataset = project.make_dataset()

        item = DatasetItem(id=1)
        dataset.put(item)

        self.assertEqual(item, next(iter(dataset)))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_project_compound_child_can_be_modified_recursively(self):
        with TestDir() as test_dir:
            child1 = Project({
                'project_dir': osp.join(test_dir, 'child1'),
            })
            child1.save()

            child2 = Project({
                'project_dir': osp.join(test_dir, 'child2'),
            })
            child2.save()

            parent = Project()
            parent.add_source('child1', {
                'url': child1.config.project_dir
            })
            parent.add_source('child2', {
                'url': child2.config.project_dir
            })
            dataset = parent.make_dataset()

            item1 = DatasetItem(id='ch1', path=['child1'])
            item2 = DatasetItem(id='ch2', path=['child2'])
            dataset.put(item1)
            dataset.put(item2)

            self.assertEqual(2, len(dataset))
            self.assertEqual(1, len(dataset.sources['child1']))
            self.assertEqual(1, len(dataset.sources['child2']))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_project_can_merge_item_annotations(self):
        class TestExtractor1(Extractor):
            def __iter__(self):
                yield DatasetItem(id=1, subset='train', annotations=[
                    Label(2, id=3),
                    Label(3, attributes={ 'x': 1 }),
                ])

        class TestExtractor2(Extractor):
            def __iter__(self):
                yield DatasetItem(id=1, subset='train', annotations=[
                    Label(3, attributes={ 'x': 1 }),
                    Label(4, id=4),
                ])

        project = Project()
        project.env.extractors.register('t1', TestExtractor1)
        project.env.extractors.register('t2', TestExtractor2)
        project.add_source('source1', { 'format': 't1' })
        project.add_source('source2', { 'format': 't2' })

        merged = project.make_dataset()

        self.assertEqual(1, len(merged))

        item = next(iter(merged))
        self.assertEqual(3, len(item.annotations))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_and_import(self):
        env = Environment()
        env.importers.items = {DEFAULT_FORMAT: env.importers[DEFAULT_FORMAT]}
        env.extractors.items = {DEFAULT_FORMAT: env.extractors[DEFAULT_FORMAT]}

        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, annotations=[ Label(2) ]),
        ], categories=['a', 'b', 'c'])

        with TestDir() as test_dir:
            source_dataset.save(test_dir)

            project = Project.import_from(test_dir, env=env)
            imported_dataset = project.make_dataset()

            self.assertEqual(next(iter(project.config.sources.values())).format,
                DEFAULT_FORMAT)
            compare_datasets(self, source_dataset, imported_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_custom_extractor_can_be_created(self):
        class CustomExtractor(Extractor):
            def __iter__(self):
                return iter([
                    DatasetItem(id=0, subset='train'),
                    DatasetItem(id=1, subset='train'),
                    DatasetItem(id=2, subset='train'),

                    DatasetItem(id=3, subset='test'),
                    DatasetItem(id=4, subset='test'),

                    DatasetItem(id=1),
                    DatasetItem(id=2),
                    DatasetItem(id=3),
                ])

        extractor_name = 'ext1'
        project = Project()
        project.env.extractors.register(extractor_name, CustomExtractor)
        project.add_source('src1', {
            'url': 'path',
            'format': extractor_name,
        })

        dataset = project.make_dataset()

        compare_datasets(self, CustomExtractor(), dataset)
