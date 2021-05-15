import numpy as np
import os
import os.path as osp
import shutil

from unittest import TestCase, skipIf, skip
from datumaro.components.config import Config
from datumaro.components.config_model import Source, Model
from datumaro.components.dataset import Dataset, DEFAULT_FORMAT
from datumaro.components.extractor import (Bbox, Extractor, DatasetItem,
    Label, LabelCategories, AnnotationType, Transform)
from datumaro.components.errors import VcsError
from datumaro.components.environment import Environment
from datumaro.components.launcher import Launcher, ModelTransform
from datumaro.components.project import (Project, BuildStageType,
    GitWrapper, DvcWrapper, Tree)
from datumaro.util.test_utils import TestDir, compare_datasets, compare_dirs


class ProjectTest(TestCase):
    def test_can_init_and_load(self):
        with TestDir() as test_dir:
            Project.init(test_dir)

            Project(test_dir)

    def test_can_import_local_model(self):
        with TestDir() as test_dir:
            source_name = 'source'
            origin = Model({
                'url': test_dir,
                'launcher': 'test',
                'options': { 'a': 5, 'b': 'hello' }
            })
            project = Project()

            project.models.add(source_name, origin)

            added = project.models[source_name]
            self.assertEqual(added.url, origin.url)
            self.assertEqual(added.launcher, origin.launcher)
            self.assertEqual(added.options, origin.options)

    def test_can_import_generated_model(self):
        model_name = 'model'
        origin = Model({
            # no url
            'launcher': 'test',
            'options': { 'c': 5, 'd': 'hello' }
        })
        project = Project()

        project.models.add(model_name, origin)

        added = project.models[model_name]
        self.assertEqual(added.launcher, origin.launcher)
        self.assertEqual(added.options, origin.options)

    def test_can_transform_source_with_model(self):
        class TestExtractor(Extractor):
            def __iter__(self):
                yield DatasetItem(0, image=np.ones([2, 2, 3]) * 0)
                yield DatasetItem(1, image=np.ones([2, 2, 3]) * 1)

            def categories(self):
                label_cat = LabelCategories().from_iterable(['0', '1'])
                return { AnnotationType.label: label_cat }

        class TestLauncher(Launcher):
            def launch(self, inputs):
                for inp in inputs:
                    yield [ Label(inp[0, 0, 0]) ]

        expected = Dataset.from_iterable([
            DatasetItem(0, image=np.zeros([2, 2, 3]), annotations=[Label(0)]),
            DatasetItem(1, image=np.ones([2, 2, 3]), annotations=[Label(1)])
        ], categories=['0', '1'])

        launcher_name = 'custom_launcher'
        extractor_name = 'custom_extractor'

        project = Project()
        project.env.launchers.register(launcher_name, TestLauncher)
        project.env.extractors.register(extractor_name, TestExtractor)
        project.models.add('model', { 'launcher': launcher_name })
        project.sources.add('source', { 'format': extractor_name })
        project.build_targets.add_inference_stage('source', 'model')

        result = project.make_dataset()

        compare_datasets(self, expected, result)

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

            self.assertEqual(next(iter(project.sources.items()))[1].format,
                DEFAULT_FORMAT)
            compare_datasets(self, source_dataset, imported_dataset)

    def test_can_import_local_source(self):
        with TestDir() as test_dir:
            source_base_url = osp.join(test_dir, 'test_repo')
            source_file_path = osp.join(source_base_url, 'x', 'y.txt')
            os.makedirs(osp.dirname(source_file_path), exist_ok=True)
            with open(source_file_path, 'w') as f:
                f.write('hello')

            project = Project.init(osp.join(test_dir, 'proj'))
            project.import_source('s1', url=source_base_url, format='fmt')

            source = project.working_tree.sources['s1']
            self.assertEqual('fmt', source.format)
            compare_dirs(self, source_base_url, project.source_data_dir('s1'))

    def test_can_import_generated_source(self):
        with TestDir() as test_dir:
            source_name = 'source'
            origin = Source({
                # no url
                'format': 'fmt',
                'options': { 'c': 5, 'd': 'hello' }
            })
            project = Project.init(test_dir)

            project.import_source(source_name, url='',
                format=origin.format, options=origin.options)

            added = project.working_tree.sources[source_name]
            self.assertEqual(added.format, origin.format)
            self.assertEqual(added.options, origin.options)

    def test_cant_import_source_with_wrong_name(self):
        with TestDir() as test_dir:
            project = Project.init(test_dir)

            for name in {'dataset', 'project', 'build', '.any'}:
                with self.subTest(name=name), \
                        self.assertRaisesRegex(ValueError, "Source name"):
                    project.import_source(name, url='', format='fmt')

    def test_can_remove_source_and_keep_data(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_source.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.init(osp.join(test_dir, 'proj'))
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)

            project.remove_source('s1', keep_data=True)

            self.assertFalse('s1' in project.working_tree.sources)
            compare_dirs(self, source_url, project.source_data_dir('s1'))

    def test_can_remove_source_and_wipe_data(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_source.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.init(osp.join(test_dir, 'proj'))
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)

            project.remove_source('s1', keep_data=False)

            self.assertFalse('s1' in project.working_tree.sources)
            self.assertFalse(osp.exists(project.source_data_dir('s1')))

    def test_can_checkout(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_source.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.init(osp.join(test_dir, 'proj'))
            project.import_source('s1', source_url, format=DEFAULT_FORMAT)
            project.commit("First commit")

            source_path = osp.join(
                project.source_data_dir('s1'),
                osp.basename(source_url))
            with open(source_path, 'w') as f:
                f.write('world')
            project.commit("Second commit")

            project.checkout('HEAD~1')

            compare_dirs(self, source_url, project.source_data_dir('s1'))

    @skip('Source data status checks are not implemented yet')
    def test_can_checkout_source_rev_noncached(self):
        raise NotImplementedError()

    def test_can_read_working_copy_of_source(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'source')
            source_dataset = Dataset.from_iterable([
                DatasetItem(0, image=np.ones((2, 3, 3)),
                    annotations=[ Bbox(1, 2, 3, 4, label=0) ]),
                DatasetItem(1, subset='s', image=np.ones((1, 2, 3)),
                    annotations=[ Bbox(1, 2, 3, 4, label=1) ]),
            ], categories=['a', 'b'])
            source_dataset.save(source_url, save_images=True)

            project = Project.init(osp.join(test_dir, 'proj'))
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)

            read_dataset = project.working_tree.sources.make_dataset('s1')

            compare_datasets(self, source_dataset, read_dataset)
            compare_dirs(self, source_url, project.source_data_dir('s1'))

    def test_can_read_current_revision_of_source(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'source')
            source_dataset = Dataset.from_iterable([
                DatasetItem(0, image=np.ones((2, 3, 3)),
                    annotations=[ Bbox(1, 2, 3, 4, label=0) ]),
                DatasetItem(1, subset='s', image=np.ones((1, 2, 3)),
                    annotations=[ Bbox(1, 2, 3, 4, label=1) ]),
            ], categories=['a', 'b'])
            source_dataset.save(source_url, save_images=True)

            project = Project.init(osp.join(test_dir, 'proj'))
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)
            project.commit("A commit")

            shutil.rmtree(project.source_data_dir('s1'))

            read_dataset = project.head.sources.make_dataset('s1')

            compare_datasets(self, source_dataset, read_dataset)
            self.assertFalse(osp.isdir(project.source_data_dir('s1')))
            compare_dirs(self, source_url, project.head.sources.data_dir('s1'))

    def test_can_make_dataset_from_project(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            source_dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
            ], categories=['a', 'b'])
            source_dataset.save(source_url)

            project = Project.init(test_dir)
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)

            read_dataset = project.working_tree.make_dataset()

            compare_datasets(self, source_dataset, read_dataset)

    def test_can_make_dataset_from_source(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.init(test_dir)
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)
            project.working_tree.build_targets.add_filter_stage('s1', '/item')

            built_dataset = project.working_tree.make_dataset('s1')

            compare_datasets(self, dataset, built_dataset)

    def test_can_add_filter_stage(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.init(test_dir)
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)

            stage = project.working_tree.build_targets.add_filter_stage('s1',
                '/item/annotation[label="b"]'
            )

            self.assertTrue(stage in project.working_tree.build_targets)

    def test_can_add_convert_stage(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.init(test_dir)
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)

            stage = project.working_tree.build_targets.add_convert_stage('s1',
                DEFAULT_FORMAT)

            self.assertTrue(stage in project.working_tree.build_targets)

    def test_can_add_transform_stage(self):
        class TestTransform(Transform):
            def __init__(self, extractor, p1=None, p2=None):
                super().__init__(extractor)
                self.p1 = p1
                self.p2 = p2

            def transform_item(self, item):
                return item

        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.init(test_dir)
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)
            project.working_tree.env.transforms.register('tr', TestTransform)

            stage = project.working_tree.build_targets.add_transform_stage('s1',
                'tr', params={'p1': 5, 'p2': ['1', 2, 3.5]}
            )

            self.assertTrue(stage in project.working_tree.build_targets)

    def test_can_make_dataset_from_stage(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.init(test_dir)
            project.import_source('s1', url=source_url, format=DEFAULT_FORMAT)
            stage = project.working_tree.build_targets.add_filter_stage('s1',
                '/item/annotation[label="b"]')

            built_dataset = project.working_tree.make_dataset(stage)

            expected_dataset = Dataset.from_iterable([
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            compare_datasets(self, expected_dataset, built_dataset)

    def test_can_commit_repo(self):
        with TestDir() as test_dir:
            project = Project.init(test_dir)

            commit_hash = project.commit("First commit")

            self.assertTrue(project.is_ref(commit_hash))


class BackwardCompatibilityTests_v0_1(TestCase):
    def test_can_load_old_project(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(0, subset='train', annotations=[Label(0)]),
            DatasetItem(1, subset='test', annotations=[Label(1)]),
        ], categories=['a', 'b'])

        project_dir = osp.join(osp.dirname(__file__),
            'assets', 'compat', 'v0.1', 'project')

        project = Project(project_dir)
        loaded_dataset = project.working_tree.make_dataset()

        compare_datasets(self, expected_dataset, loaded_dataset)


class ModelsTest(TestCase):
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
        project.models.add(model_name, { 'launcher': launcher_name })
        model = project.models.make_executable_model(model_name)

        batch_size = 3
        executor = ModelTransform(dataset, model, batch_size=batch_size)

        for item in executor:
            self.assertEqual(1, len(item.annotations))
            self.assertEqual(int(item.id) % batch_size,
                item.annotations[0].attributes['idx'])
            self.assertEqual(int(item.id),
                item.annotations[0].attributes['data'])
