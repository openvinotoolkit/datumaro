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


class TreeTest(TestCase):
    @staticmethod
    def test_default_ctor_is_ok():
        Project()

    @staticmethod
    def test_empty_config_is_ok():
        Project(Config())

    def test_inmemory_tree_is_detached(self):
        tree = Tree()

        self.assertTrue(tree.detached)

    def test_can_add_existing_local_source(self):
        # Reasons to exist:
        # - Backward compatibility
        # - In-memory and detached projects

        with TestDir() as test_dir:
            source_name = 'source'
            origin = Source({
                'url': test_dir,
                'format': 'fmt',
                'options': { 'a': 5, 'b': 'hello' }
            })
            project = Project()

            project.sources.add(source_name, origin)

            added = project.sources[source_name]
            self.assertEqual(added.url, origin.url)
            self.assertEqual(added.format, origin.format)
            self.assertEqual(added.options, origin.options)

    def test_cant_add_nonexisting_local_source(self):
        project = Project()

        with self.assertRaisesRegex(Exception, 'Can only add an existing'):
            project.sources.add('source', { 'url': '_p_a_t_h_' })

    def test_can_add_generated_source(self):
        source_name = 'source'
        origin = Source({
            # no url
            'format': 'fmt',
            'options': { 'c': 5, 'd': 'hello' }
        })
        project = Project()

        project.sources.add(source_name, origin)

        added = project.sources[source_name]
        self.assertEqual(added.format, origin.format)
        self.assertEqual(added.options, origin.options)

    def test_can_make_dataset(self):
        class CustomExtractor(Extractor):
            def __iter__(self):
                yield DatasetItem(42)

        extractor_name = 'ext1'
        project = Project()
        project.env.extractors.register(extractor_name, CustomExtractor)
        project.sources.add('src1', { 'format': extractor_name })

        dataset = project.make_dataset()

        compare_datasets(self, CustomExtractor(), dataset)

    def test_can_save_added_source(self):
        with TestDir() as test_dir:
            project = Project()
            project.sources.add('s', { 'format': 'fmt' })

            project.save(test_dir)

            loaded = Project.load(test_dir)
            self.assertEqual('fmt', loaded.sources['s'].format)

    def test_can_add_existing_local_model(self):
        # Reasons to exist:
        # - Backward compatibility
        # - In-memory and detached projects

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

    def test_cant_add_nonexisting_local_model(self):
        project = Project()

        with self.assertRaisesRegex(Exception, 'Can only add an existing'):
            project.models.add('m', { 'url': '_p_a_t_h_', 'launcher': 'test' })

    def test_can_add_generated_model(self):
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

    def test_can_save_added_model(self):
        project = Project()

        saved = Model({ 'launcher': 'test' })
        project.models.add('model', saved)

        with TestDir() as test_dir:
            project.save(test_dir)

            loaded = Project.load(test_dir)
            loaded = loaded.models['model']
            self.assertEqual(saved, loaded)

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

    def test_can_filter_source(self):
        class TestExtractor(Extractor):
            def __iter__(self):
                yield DatasetItem(0)
                yield DatasetItem(10)
                yield DatasetItem(2)
                yield DatasetItem(15)

        project = Project()
        project.env.extractors.register('f', TestExtractor)
        project.sources.add('source', { 'format': 'f' })
        project.build_targets.add_filter_stage('source', {
            'expr': '/item[id < 5]'
        })

        dataset = project.make_dataset()

        self.assertEqual(2, len(dataset))

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


no_vcs_installed = False
try:
    import git # pylint: disable=unused-import
    import dvc # pylint: disable=unused-import
except ImportError:
    no_vcs_installed = True

@skipIf(no_vcs_installed, "No VCS modules (Git, DVC) installed")
class ProjectTest(TestCase):
    def test_can_init_and_load(self):
        with TestDir() as test_dir:
            Project.init(save_dir=test_dir)

            Project(test_dir)

    def test_can_add_local_source_by_url(self):
        with TestDir() as test_dir:
            source_base_url = osp.join(test_dir, 'test_repo')
            source_file_path = osp.join(source_base_url, 'x', 'y.txt')
            os.makedirs(osp.dirname(source_file_path), exist_ok=True)
            with open(source_file_path, 'w') as f:
                f.write('hello')

            project = Project.init(test_dir)
            project.working_tree.sources.add('s1', {
                'url': source_base_url,
                'format': 'fmt',
            })

            source = project.working_tree.sources['s1']
            self.assertEqual('fmt', source.format)
            compare_dirs(self, source_base_url,
                project.working_tree.sources.data_dir('s1'))

    def test_can_add_generated_source(self):
        with TestDir() as test_dir:
            source_name = 'source'
            origin = Source({
                'format': 'fmt',
                'options': { 'c': 5, 'd': 'hello' }
            })
            project = Project.init(test_dir)

            project.working_tree.sources.add(source_name, origin)

            added = project.working_tree.sources[source_name]
            self.assertEqual(added.format, origin.format)
            self.assertEqual(added.options, origin.options)

    def test_cant_add_source_with_wrong_name(self):
        with TestDir() as test_dir:
            project = Project.init(test_dir)

            for name in {'dataset', 'project', 'build', '.any'}:
                with self.subTest(name=name), \
                        self.assertRaisesRegex(ValueError, "Source name"):
                    project.working_tree.sources.add(name, { 'format': 'fmt' })

    def test_can_remove_source_and_keep_data(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_source.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.init(test_dir)
            project.working_tree.sources.add('s1', { 'url': source_url })

            project.working_tree.sources.remove('s1', keep_data=True)

            self.assertFalse('s1' in project.working_tree.sources)
            compare_dirs(self, source_url,
                project.working_tree.sources.data_dir('s1'))

    def test_can_remove_source_and_wipe_data(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_source.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.init(test_dir)
            project.working_tree.sources.add('s1', { 'url': source_url })

            project.working_tree.sources.remove('s1', keep_data=False)

            self.assertFalse('s1' in project.working_tree.sources)
            self.assertFalse(osp.isfile(osp.join(
                project.working_tree.sources.data_dir('s1'),
                osp.basename(source_url))))

    def test_can_checkout(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_source.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.init(test_dir)
            project.working_tree.sources.add('s1', { 'url': source_url })
            project.add('s1')
            project.commit("First commit")

            local_source_path = osp.join(
                project.working_tree.sources.data_dir('s1'),
                osp.basename(source_url))
            with open(local_source_path, 'w') as f:
                f.write('world')
            project.add('s1')
            project.commit("Second commit")

            project.checkout('HEAD~1')

            compare_dirs(self, source_url,
                project.working_tree.sources.data_dir('s1'))

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
            project.working_tree.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })

            read_dataset = project.working_tree.sources.make_dataset('s1')

            compare_datasets(self, source_dataset, read_dataset)
            compare_dirs(self, source_url,
                project.working_tree.sources.data_dir('s1'))

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
            project.working_tree.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })
            project.add('s1')
            project.commit("A commit")

            shutil.rmtree(project.working_tree.sources.data_dir('s1'))

            read_dataset = project.head.sources.make_dataset('s1')

            compare_datasets(self, source_dataset, read_dataset)
            self.assertFalse(osp.isdir(
                project.working_tree.sources.data_dir('s1')))
            compare_dirs(self, source_url, project.head.sources.data_dir('s1'))

    def test_can_make_dataset_from_project(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.init(test_dir)
            project.working_tree.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })

            built_dataset = project.working_tree.make_dataset()

            compare_datasets(self, dataset, built_dataset)

    def test_can_make_dataset_from_source(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.init(test_dir)
            project.working_tree.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })
            project.working_tree.build_targets.add_filter_stage('s1', {
                'expr': '/item'
            })

            built_dataset = project.working_tree.make_dataset('s1')

            compare_datasets(self, dataset, built_dataset)

    def test_can_add_stage_directly(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.init(test_dir)
            project.working_tree.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })

            project.working_tree.build_targets.add_stage('s1', {
                'type': BuildStageType.filter.name,
                'params': {'expr': '/item/annotation[label="b"]'},
            }, name='f1')

            self.assertTrue('s1.f1' in project.working_tree.build_targets)

    def test_can_add_filter_stage(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.init(test_dir)
            project.working_tree.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })

            _, stage = project.working_tree.build_targets.add_filter_stage('s1',
                {'expr': '/item/annotation[label="b"]'}
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

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })

            _, stage = project.build_targets.add_convert_stage('s1',
                DEFAULT_FORMAT)
            project.save()

            self.assertTrue(stage in project.build_targets)

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

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })
            project.env.transforms.register('tr', TestTransform)

            _, stage = project.build_targets.add_transform_stage('s1',
                'tr', params={'p1': 5, 'p2': ['1', 2, 3.5]}
            )
            project.save()

            self.assertTrue(stage in project.build_targets)

    def test_can_build_stage(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })
            project.build_targets.add_stage('s1', {
                'type': BuildStageType.filter.name,
                'params': {'expr': '/item/annotation[label="b"]'},
            }, name='f1')

            project.build('s1.f1', out_dir=osp.join(test_dir, 'test_build'))

            built_dataset = Dataset.load(osp.join(test_dir, 'test_build'))
            expected_dataset = Dataset.from_iterable([
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            compare_datasets(self, expected_dataset, built_dataset)

    def test_can_make_dataset_from_stage(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', {
                'url': source_url,
                'format': DEFAULT_FORMAT,
            })
            project.build_targets.add_stage('s1', {
                'type': BuildStageType.filter.name,
                'params': {'expr': '/item/annotation[label="b"]'},
            }, name='f1')

            built_dataset = project.make_dataset('s1.f1')

            expected_dataset = Dataset.from_iterable([
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            compare_datasets(self, expected_dataset, built_dataset)

    def test_can_commit_repo(self):
        with TestDir() as test_dir:
            project = Project.generate(save_dir=test_dir)

            project.vcs.commit(None, message="First commit")

    def test_can_checkout_repo(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo', 'x', 'y.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.vcs.commit(None, message="First commit")

            project.sources.add('s1', { 'url': source_url })
            project.save()
            project.vcs.commit(None, message="Second commit")

            project.vcs.checkout('HEAD~1')

            project = Project.load(test_dir)
            self.assertFalse('s1' in project.sources)

    def test_can_push_repo(self):
        with TestDir() as test_dir:
            git_repo_dir = osp.join(test_dir, 'git_repo')
            os.makedirs(git_repo_dir, exist_ok=True)
            GitWrapper.module.Repo.init(git_repo_dir, bare=True)

            dvc_repo_dir = osp.join(test_dir, 'dvc_repo')
            os.makedirs(dvc_repo_dir, exist_ok=True)
            git = GitWrapper(dvc_repo_dir)
            git.init()
            dvc = DvcWrapper(dvc_repo_dir)
            dvc.init()

            project = Project.generate(save_dir=osp.join(test_dir, 'proj'))
            project.vcs.repositories.add('origin', git_repo_dir)
            project.vcs.remotes.add('data', {
                'url': dvc_repo_dir,
                'type': 'dvc',
            })
            project.vcs.remotes.set_default('data')
            project.save()
            project.vcs.commit(None, message="First commit")

            project.vcs.push()

            git = GitWrapper.module.Repo.init(git_repo_dir, bare=True)
            self.assertEqual('First commit', next(git.iter_commits()).summary)

    def test_can_tag_repo(self):
        with TestDir() as test_dir:
            project = Project.generate(save_dir=test_dir)

            project.vcs.commit(None, message="First commit")
            project.vcs.tag('r1')

            self.assertEqual(['r1'], project.vcs.tags)


class BackwardCompatibilityTests_v0_1(TestCase):
    def test_can_load_old_project(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(0, subset='train', annotations=[Label(0)]),
            DatasetItem(1, subset='test', annotations=[Label(1)]),
        ], categories=['a', 'b'])

        project_dir = osp.join(osp.dirname(__file__),
            'assets', 'compat', 'v0.1', 'project')

        project = Project.load(project_dir)
        loaded_dataset = project.make_dataset()

        compare_datasets(self, expected_dataset, loaded_dataset)

    @skip("Not actual")
    def test_project_compound_child_can_be_modified_recursively(self):
        with TestDir() as test_dir:
            child1 = Project.generate(osp.join(test_dir, 'child1'))
            child2 = Project.generate(osp.join(test_dir, 'child2'))

            parent = Project()
            parent.sources.add('child1', {
                'url': child1.config.project_dir,
                'format': 'datumaro_project'
            })
            parent.sources.add('child2', {
                'url': child2.config.project_dir,
                'format': 'datumaro_project'
            })
            dataset = parent.make_dataset()

            item1 = DatasetItem(id='ch1', path=['child1'])
            item2 = DatasetItem(id='ch2', path=['child2'])
            dataset.put(item1)
            dataset.put(item2)

            self.assertEqual(2, len(dataset))
            self.assertEqual(1, len(dataset.sources['child1']))
            self.assertEqual(1, len(dataset.sources['child2']))

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
