import numpy as np
import os
import os.path as osp
import shutil

from unittest import TestCase, skipIf, skip

from datumaro.components.project import Project, Environment, Dataset
from datumaro.components.config import Config
from datumaro.components.config_model import Source, Model
from datumaro.components.launcher import Launcher, ModelTransform
from datumaro.components.extractor import (Extractor, DatasetItem,
    Label, LabelCategories, AnnotationType
)
from datumaro.util.test_utils import TestDir, compare_datasets


class ProjectTest(TestCase):
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
    def test_default_ctor_is_ok():
        Project()

    @staticmethod
    def test_empty_config_is_ok():
        Project(Config())

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

    def test_added_source_can_be_saved(self):
        source_name = 'source'
        origin = Source({
            'url': 'path',
        })
        project = Project()
        project.add_source(source_name, origin)

        saved = project.config

        self.assertEqual(origin, saved.sources[source_name])

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

    def test_can_have_project_source(self):
        with TestDir() as test_dir:
            Project.generate(test_dir)

            project2 = Project()
            project2.add_source('project1', {
                'url': test_dir,
            })
            dataset = project2.make_dataset()

            self.assertTrue('project1' in dataset.sources)

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

    def test_project_own_dataset_can_be_modified(self):
        project = Project()
        dataset = project.make_dataset()

        item = DatasetItem(id=1)
        dataset.put(item)

        self.assertEqual(item, next(iter(dataset)))

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


no_vcs_installed = False
try:
    import git # pylint: disable=unused-import
    import dvc # pylint: disable=unused-import
except ImportError:
    no_vcs_installed = True

@skipIf(no_vcs_installed, "No VCS modules (Git, DVC) installed")
class AttachedProjectTest(TestCase):
    def tearDown(self):
        # cleanup DVC module to avoid
        pass

    def test_can_create(self):
        with TestDir() as test_dir:
            Project.generate(save_dir=test_dir)

            Project.load(test_dir)

            self.assertTrue(osp.isdir(osp.join(test_dir, '.git')))
            self.assertTrue(osp.isdir(osp.join(test_dir, '.dvc')))

    def test_can_add_source_by_url(self):
        with TestDir() as test_dir:
            source_base_url = osp.join(test_dir, 'test_repo')
            source_file_path = osp.join(source_base_url, 'x', 'y.txt')
            os.makedirs(osp.dirname(source_file_path), exist_ok=True)
            with open(source_file_path, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', {
                'url': source_base_url,
                'format': 'fmt',
            })
            project.save()

            source = project.sources['s1']
            self.assertEqual(source.url, '')
            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'x', 'y.txt')))

    def test_can_add_source_with_existing_remote(self):
        with TestDir() as test_dir:
            source_base_url = osp.join(test_dir, 'test_repo')
            source_file_path = osp.join(source_base_url, 'x', 'y.txt')
            os.makedirs(osp.dirname(source_file_path), exist_ok=True)
            with open(source_file_path, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.vcs.remotes.add('r1', { 'url': source_base_url })
            project.sources.add('s1', {
                'url': 'remote://r1/x/y.txt',
                'format': 'fmt'
            })
            project.save()

            source = project.sources['s1']
            remote = project.vcs.remotes[source.remote]
            self.assertEqual(source.url, 'y.txt')
            self.assertEqual(source.remote, 'r1')
            self.assertEqual(remote.url, source_base_url)
            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'y.txt')))

    def test_can_add_generated_source(self):
        with TestDir() as test_dir:
            source_name = 'source'
            origin = Source({
                'format': 'fmt',
                'options': { 'c': 5, 'd': 'hello' }
            })
            project = Project.generate(save_dir=test_dir)

            project.sources.add(source_name, origin)
            project.save()

            added = project.sources[source_name]
            self.assertEqual(added.format, origin.format)
            self.assertEqual(added.options, origin.options)

    def test_can_pull_dir_source(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo', 'x')
            source_path = osp.join(source_url, 'y.txt')
            os.makedirs(osp.dirname(source_path), exist_ok=True)
            with open(source_path, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', { 'url': source_url })
            shutil.rmtree(project.sources.source_dir('s1'))

            project.sources.pull('s1')

            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'y.txt')))

    def test_can_pull_file_source(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo', 'x', 'y.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', { 'url': source_url })
            shutil.rmtree(project.sources.source_dir('s1'))

            project.sources.pull('s1')

            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'y.txt')))

    def test_can_pull_source_with_existing_remote_rel_dir(self):
        with TestDir() as test_dir:
            source_base_url = osp.join(test_dir, 'test_repo')
            source_file_path = osp.join(source_base_url, 'x', 'y.txt')
            os.makedirs(osp.dirname(source_file_path), exist_ok=True)
            with open(source_file_path, 'w') as f:
                f.write('hello')
            source_file_path2 = osp.join(source_base_url, 'x', 'z.txt')
            with open(source_file_path2, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.vcs.remotes.add('r1', { 'url': source_base_url })
            project.sources.add('s1', {
                'url': 'remote://r1/x/',
                'format': 'fmt'
            })
            shutil.rmtree(project.sources.source_dir('s1'))

            project.sources.pull('s1')

            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'y.txt')))
            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'z.txt')))

    def test_can_pull_source_with_existing_remote_rel_file(self):
        with TestDir() as test_dir:
            source_base_url = osp.join(test_dir, 'test_repo')
            source_file_path = osp.join(source_base_url, 'x', 'y.txt')
            os.makedirs(osp.dirname(source_file_path), exist_ok=True)
            with open(source_file_path, 'w') as f:
                f.write('hello')
            # another file in the remote directory, should not be copied
            source_file_path2 = osp.join(source_base_url, 'x', 'z.txt')
            with open(source_file_path2, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.vcs.remotes.add('r1', { 'url': source_base_url })
            project.sources.add('s1', {
                'url': 'remote://r1/x/y.txt',
                'format': 'fmt'
            })
            shutil.rmtree(project.sources.source_dir('s1'))

            project.sources.pull('s1')

            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'y.txt')))
            self.assertFalse(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'z.txt')))

    def test_can_pull_source_with_existing_remote_root_file(self):
        with TestDir() as test_dir:
            source_base_url = osp.join(test_dir, 'test_repo')
            source_file_path = osp.join(source_base_url, 'y.txt')
            os.makedirs(osp.dirname(source_file_path), exist_ok=True)
            with open(source_file_path, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.vcs.remotes.add('r1', { 'url': source_file_path })
            project.sources.add('s1', {
                'url': 'remote://r1',
                'format': 'fmt'
            })
            shutil.rmtree(project.sources.source_dir('s1'))

            project.sources.pull('s1')

            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'y.txt')))

    def test_can_pull_source_with_existing_remote_root_dir(self):
        with TestDir() as test_dir:
            source_base_url = osp.join(test_dir, 'test_repo')
            source_file_path = osp.join(source_base_url, 'y.txt')
            os.makedirs(osp.dirname(source_file_path), exist_ok=True)
            with open(source_file_path, 'w') as f:
                f.write('hello')
            source_file_path2 = osp.join(source_base_url, 'z.txt')
            with open(source_file_path2, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.vcs.remotes.add('r1', { 'url': source_base_url })
            project.sources.add('s1', {
                'url': 'remote://r1',
                'format': 'fmt'
        })
            shutil.rmtree(project.sources.source_dir('s1'))

            project.sources.pull('s1')

            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'y.txt')))
            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), 'z.txt')))

    def test_can_remove_source_and_keep_data(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo', 'x', 'y.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', { 'url': source_url })

            project.sources.remove('s1', keep_data=True)

            self.assertFalse('s1' in project.sources)
            self.assertTrue(osp.isfile(osp.join(
                project.sources.source_dir('s1'), osp.basename(source_url))))

    def test_can_remove_source_and_wipe_data(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo', 'x', 'y.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', { 'url': source_url })

            project.sources.remove('s1', keep_data=False)

            self.assertFalse('s1' in project.sources)
            self.assertFalse(osp.isfile(osp.join(
                project.sources.source_dir('s1'), osp.basename(source_url))))

    def test_can_checkout_source_rev_cached(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo', 'x', 'y.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', { 'url': source_url })
            local_source_path = osp.join(
                project.sources.source_dir('s1'), osp.basename(source_url))
            project.save()
            project.vcs.commit(None, message="First commit")

            with open(local_source_path, 'w') as f:
                f.write('world')
            project.vcs.commit(None, message="Second commit")

            project.vcs.checkout('HEAD~1', ['s1'])

            self.assertTrue(osp.isfile(local_source_path))
            with open(local_source_path) as f:
                self.assertEqual('hello', f.readline().strip())

    @skip('Source data status checks are not implemented yet')
    def test_can_checkout_source_rev_noncached(self):
        # Can't detect automatically if there is no cached source version
        # in DVC cache, or if checkout produced a mismatching version of data.
        # For example:
        # a source was transformed without application
        # - its stages changed, but files were not
        # - it was committed, no changes in source data,
        #     so no updates in the DVC cache
        # checkout produces an outdated version of the source.
        # Resolution - source rebuilding.
        raise NotImplementedError()

    def test_can_update_source(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo', 'x', 'y.txt')
            os.makedirs(osp.dirname(source_url), exist_ok=True)
            with open(source_url, 'w') as f:
                f.write('hello')

            project = Project.generate(save_dir=test_dir)
            project.sources.add('s1', { 'url': source_url })
            project.save()
            project.vcs.commit(None, message="First commit")

            with open(source_url, 'w') as f:
                f.write('world')

            project.sources.pull('s1')

            local_source_path = osp.join(
                project.sources.source_dir('s1'), osp.basename(source_url))
            self.assertTrue(osp.isfile(local_source_path))
            with open(local_source_path) as f:
                self.assertEqual('world', f.readline().strip())


        dataset = project.make_dataset()

        compare_datasets(self, CustomExtractor(), dataset)

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

        compare_datasets(self, DstExtractor(), dataset)


class DatasetItemTest(TestCase):
    def test_ctor_requires_id(self):
        with self.assertRaises(Exception):
            # pylint: disable=no-value-for-parameter
            DatasetItem()
            # pylint: enable=no-value-for-parameter

    @staticmethod
    def test_ctors_with_image():
        for args in [
            { 'id': 0, 'image': None },
            { 'id': 0, 'image': 'path.jpg' },
            { 'id': 0, 'image': np.array([1, 2, 3]) },
            { 'id': 0, 'image': lambda f: np.array([1, 2, 3]) },
            { 'id': 0, 'image': Image(data=np.array([1, 2, 3])) },
        ]:
            DatasetItem(**args)