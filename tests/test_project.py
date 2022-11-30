import os
import os.path as osp
import shutil
import textwrap
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Bbox, Label
from datumaro.components.config_model import Model, Source
from datumaro.components.dataset import DEFAULT_FORMAT, Dataset
from datumaro.components.errors import (
    DatasetMergeError,
    EmptyCommitError,
    EmptyPipelineError,
    ForeignChangesError,
    MismatchingObjectError,
    MissingObjectError,
    MissingSourceHashError,
    OldProjectError,
    PathOutsideSourceError,
    ReadonlyProjectError,
    SourceExistsError,
    SourceUrlInsideProjectError,
    UnexpectedUrlError,
    UnknownTargetError,
)
from datumaro.components.extractor import DatasetItem, DatasetBase
from datumaro.components.launcher import Launcher
from datumaro.components.media import Image
from datumaro.components.project import DiffStatus, Project
from datumaro.components.transformer import ItemTransform
from datumaro.util.scope import scope_add, scoped
from datumaro.util.test_utils import TestDir, compare_datasets, compare_dirs

from .requirements import Requirements, mark_requirement


class ProjectTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_init_and_load(self):
        test_dir = scope_add(TestDir())

        scope_add(Project.init(test_dir)).close()
        scope_add(Project(test_dir))

        self.assertTrue(".datumaro" in os.listdir(test_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_find_project_in_project_dir(self):
        test_dir = scope_add(TestDir())

        scope_add(Project.init(test_dir))

        self.assertEqual(osp.join(test_dir, ".datumaro"), Project.find_project_dir(test_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_find_project_when_no_project(self):
        test_dir = scope_add(TestDir())

        self.assertEqual(None, Project.find_project_dir(test_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_add_local_model(self):
        class TestLauncher(Launcher):
            pass

        source_name = "source"
        config = Model({"launcher": "test", "options": {"a": 5, "b": "hello"}})

        test_dir = scope_add(TestDir())
        project = scope_add(Project.init(test_dir))
        project.env.launchers.register("test", TestLauncher)

        project.add_model(source_name, launcher=config.launcher, options=config.options)

        added = project.models[source_name]
        self.assertEqual(added.launcher, config.launcher)
        self.assertEqual(added.options, config.options)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_run_inference(self):
        class TestLauncher(Launcher):
            def launch(self, inputs):
                for inp in inputs:
                    yield [Label(inp[0, 0, 0])]

        expected = Dataset.from_iterable(
            [
                DatasetItem(0, media=Image(data=np.zeros([2, 2, 3])), annotations=[Label(0)]),
                DatasetItem(1, media=Image(data=np.ones([2, 2, 3])), annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )

        launcher_name = "custom_launcher"
        model_name = "model"

        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(0, media=Image(data=np.zeros([2, 2, 3]) * 0)),
                DatasetItem(1, media=Image(data=np.ones([2, 2, 3]) * 1)),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url, save_media=True)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.env.launchers.register(launcher_name, TestLauncher)
        project.add_model(model_name, launcher=launcher_name)
        project.import_source("source", source_url, format=DEFAULT_FORMAT)

        dataset = project.working_tree.make_dataset()
        model = project.make_model(model_name)

        inference = dataset.run_model(model)

        compare_datasets(self, expected, inference)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_import_local_dir_source(self):
        test_dir = scope_add(TestDir())
        source_base_url = osp.join(test_dir, "test_repo")
        source_file_path = osp.join(source_base_url, "x", "y.txt")
        os.makedirs(osp.dirname(source_file_path), exist_ok=True)
        with open(source_file_path, "w") as f:
            f.write("hello")

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_base_url, format="fmt")

        source = project.working_tree.sources["s1"]
        self.assertEqual("fmt", source.format)
        compare_dirs(self, source_base_url, project.source_data_dir("s1"))
        with open(osp.join(test_dir, "proj", ".gitignore")) as f:
            self.assertTrue("/s1" in [line.strip() for line in f])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_import_local_file_source(self):
        # In this variant, we copy and read just the file specified

        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "f.txt")
        os.makedirs(osp.dirname(source_url), exist_ok=True)
        with open(source_url, "w") as f:
            f.write("hello")

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format="fmt")

        source = project.working_tree.sources["s1"]
        self.assertEqual("fmt", source.format)
        self.assertEqual("f.txt", source.path)

        self.assertEqual({"f.txt"}, set(os.listdir(project.source_data_dir("s1"))))
        with open(osp.join(test_dir, "proj", ".gitignore")) as f:
            self.assertTrue("/s1" in [line.strip() for line in f])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_import_local_source_with_relpath(self):
        # This form must copy all the data in URL, but read only
        # specified files. Required to support subtasks and subsets.

        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    subset="a",
                    media=Image(data=np.zeros([2, 2, 3])),
                    annotations=[Bbox(1, 2, 3, 4, label=0)],
                ),
                DatasetItem(
                    1,
                    subset="b",
                    media=Image(data=np.zeros((10, 20, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url, save_media=True)

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="b",
                    media=Image(data=np.zeros((10, 20, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )

        project = scope_add(Project.init(osp.join(test_dir, "proj")))

        project.import_source(
            "s1", url=source_url, format=DEFAULT_FORMAT, rpath=osp.join("annotations", "b.json")
        )

        source = project.working_tree.sources["s1"]
        self.assertEqual(DEFAULT_FORMAT, source.format)

        compare_dirs(self, source_url, project.source_data_dir("s1"))
        read_dataset = project.working_tree.make_dataset("s1")
        compare_datasets(self, expected_dataset, read_dataset, require_media=True)

        with open(osp.join(test_dir, "proj", ".gitignore")) as f:
            self.assertTrue("/s1" in [line.strip() for line in f])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_import_local_source_with_relpath_outside(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        os.makedirs(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))

        with self.assertRaises(PathOutsideSourceError):
            project.import_source("s1", url=source_url, format=DEFAULT_FORMAT, rpath="..")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_import_local_source_with_url_inside_project(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "qq")
        with open(source_url, "w") as f:
            f.write("hello")

        project = scope_add(Project.init(test_dir))

        with self.assertRaises(SourceUrlInsideProjectError):
            project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_report_incompatible_sources(self):
        test_dir = scope_add(TestDir())
        source1_url = osp.join(test_dir, "dataset1")
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
            ],
            categories=["a", "b"],
        )
        dataset1.save(source1_url)

        source2_url = osp.join(test_dir, "dataset2")
        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
            ],
            categories=["c", "d"],
        )
        dataset2.save(source2_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source1_url, format=DEFAULT_FORMAT)
        project.import_source("s2", url=source2_url, format=DEFAULT_FORMAT)

        with self.assertRaises(DatasetMergeError) as cm:
            project.working_tree.make_dataset()

        self.assertEqual({"s1.root", "s2.root"}, cm.exception.sources)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_import_sources_with_same_names(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

        with self.assertRaises(SourceExistsError):
            project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_import_generated_source(self):
        test_dir = scope_add(TestDir())
        source_name = "source"
        origin = Source(
            {
                # no url
                "format": "fmt",
                "options": {"c": 5, "d": "hello"},
            }
        )
        project = scope_add(Project.init(test_dir))

        project.import_source(source_name, url="", format=origin.format, options=origin.options)

        added = project.working_tree.sources[source_name]
        self.assertEqual(added.format, origin.format)
        self.assertEqual(added.options, origin.options)
        with open(osp.join(test_dir, ".gitignore")) as f:
            self.assertTrue("/" + source_name in [line.strip() for line in f])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_import_source_with_wrong_name(self):
        test_dir = scope_add(TestDir())
        project = scope_add(Project.init(test_dir))

        for name in {"dataset", "project", "build", ".any"}:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Source name"):
                project.import_source(name, url="", format="fmt")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_add_project_local_source(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")

        project = scope_add(Project.init(proj_dir))

        source_base_url = osp.join(proj_dir, "x")
        source_file_path = osp.join(source_base_url, "y.txt")
        os.makedirs(osp.dirname(source_file_path))
        with open(source_file_path, "w") as f:
            f.write("hello")

        name, source = project.add_source(source_base_url, format="fmt")

        self.assertEqual("x", name)
        self.assertEqual(project.working_tree.sources[name], source)
        self.assertEqual("fmt", source.format)

        with open(osp.join(test_dir, "proj", ".gitignore")) as f:
            self.assertTrue("/x" in [line.strip() for line in f])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_add_source_deep_in_the_project(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "proj")

        project = scope_add(Project.init(proj_dir))

        source_base_url = osp.join(proj_dir, "x", "y")
        source_file_path = osp.join(source_base_url, "y.txt")
        os.makedirs(osp.dirname(source_file_path))
        with open(source_file_path, "w") as f:
            f.write("hello")

        with self.assertRaises(UnexpectedUrlError):
            project.add_source(source_base_url, format="fmt")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_add_source_outside_project(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "x")
        os.makedirs(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))

        with self.assertRaises(UnexpectedUrlError):
            project.add_source(source_url, format="fmt")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_remove_source_and_keep_data(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_source.txt")
        os.makedirs(osp.dirname(source_url), exist_ok=True)
        with open(source_url, "w") as f:
            f.write("hello")

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

        project.remove_source("s1", keep_data=True)

        self.assertFalse("s1" in project.working_tree.sources)
        compare_dirs(self, source_url, project.source_data_dir("s1"))
        with open(osp.join(test_dir, "proj", ".gitignore")) as f:
            self.assertFalse("/s1" in [line.strip() for line in f])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_remove_source_and_wipe_data(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_source.txt")
        os.makedirs(osp.dirname(source_url), exist_ok=True)
        with open(source_url, "w") as f:
            f.write("hello")

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

        project.remove_source("s1", keep_data=False)

        self.assertFalse("s1" in project.working_tree.sources)
        self.assertFalse(osp.exists(project.source_data_dir("s1")))
        with open(osp.join(test_dir, "proj", ".gitignore")) as f:
            self.assertFalse("/s1" in [line.strip() for line in f])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_redownload_source_rev_noncached(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0, media=Image(data=np.ones((2, 3, 3))), annotations=[Bbox(1, 2, 3, 4, label=0)]
                ),
                DatasetItem(
                    1,
                    subset="s",
                    media=Image(data=np.zeros((10, 20, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url, save_media=True)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.commit("A commit")

        # remove local source data
        project.remove_cache_obj(project.working_tree.build_targets["s1"].head.hash)
        shutil.rmtree(project.source_data_dir("s1"))

        read_dataset = project.working_tree.make_dataset("s1")

        compare_datasets(self, source_dataset, read_dataset)
        compare_dirs(
            self, source_url, project.cache_path(project.working_tree.build_targets["s1"].root.hash)
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_redownload_source_and_check_data_hash(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    media=Image(data=np.zeros((2, 3, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=0)],
                ),
                DatasetItem(
                    1,
                    subset="s",
                    media=Image(data=np.zeros((10, 20, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url, save_media=True)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.commit("A commit")

        # remove local source data
        project.remove_cache_obj(project.working_tree.build_targets["s1"].head.hash)
        shutil.rmtree(project.source_data_dir("s1"))

        # modify the source repo
        with open(osp.join(source_url, "extra_file.txt"), "w") as f:
            f.write("text\n")

        with self.assertRaises(MismatchingObjectError):
            project.working_tree.make_dataset("s1")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_use_source_from_cache_with_working_copy(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    media=Image(data=np.zeros((2, 3, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=0)],
                ),
                DatasetItem(
                    1,
                    subset="s",
                    media=Image(data=np.zeros((10, 20, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url, save_media=True)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.commit("A commit")

        shutil.rmtree(project.source_data_dir("s1"))

        read_dataset = project.working_tree.make_dataset("s1")

        compare_datasets(self, source_dataset, read_dataset)
        self.assertFalse(osp.isdir(project.source_data_dir("s1")))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_raises_an_error_if_local_data_unknown(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    media=Image(data=np.zeros((2, 3, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=0)],
                ),
                DatasetItem(
                    1,
                    subset="s",
                    media=Image(data=np.zeros((10, 20, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url, save_media=True)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.commit("A commit")

        # remove the cached object so that it couldn't be matched
        project.remove_cache_obj(project.working_tree.build_targets["s1"].root.hash)

        # modify local source data
        with open(osp.join(project.source_data_dir("s1"), "extra.txt"), "w") as f:
            f.write("text\n")

        with self.assertRaises(ForeignChangesError):
            project.working_tree.make_dataset("s1")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_working_copy_of_source(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    media=Image(data=np.zeros((2, 3, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=0)],
                ),
                DatasetItem(
                    1,
                    subset="s",
                    media=Image(data=np.zeros((1, 2, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url, save_media=True)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

        read_dataset = project.working_tree.make_dataset("s1")

        compare_datasets(self, source_dataset, read_dataset)
        compare_dirs(self, source_url, project.source_data_dir("s1"))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_current_revision_of_source(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    media=Image(data=np.zeros((2, 3, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=0)],
                ),
                DatasetItem(
                    1,
                    subset="s",
                    media=Image(data=np.zeros((1, 2, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url, save_media=True)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.commit("A commit")

        shutil.rmtree(project.source_data_dir("s1"))

        read_dataset = project.head.make_dataset("s1")

        compare_datasets(self, source_dataset, read_dataset)
        self.assertFalse(osp.isdir(project.source_data_dir("s1")))
        compare_dirs(self, source_url, project.head.source_data_dir("s1"))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_make_dataset_from_project(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

        read_dataset = project.working_tree.make_dataset()

        compare_datasets(self, source_dataset, read_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_make_dataset_from_source(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

        built_dataset = project.working_tree.make_dataset("s1")

        compare_datasets(self, dataset, built_dataset)
        self.assertEqual(DEFAULT_FORMAT, built_dataset.format)
        self.assertEqual(project.source_data_dir("s1"), built_dataset.data_path)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_make_dataset_from_empty_project(self):
        test_dir = scope_add(TestDir())

        project = scope_add(Project.init(osp.join(test_dir, "proj")))

        with self.assertRaises(EmptyPipelineError):
            project.working_tree.make_dataset()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_make_dataset_from_unknown_target(self):
        test_dir = scope_add(TestDir())

        project = scope_add(Project.init(osp.join(test_dir, "proj")))

        with self.assertRaises(UnknownTargetError):
            project.working_tree.make_dataset("s1")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_add_filter_stage(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

        new_tree = project.working_tree.clone()
        stage = new_tree.build_targets.add_filter_stage("s1", '/item/annotation[label="b"]')

        self.assertTrue(stage in new_tree.build_targets)
        self.assertTrue(stage not in project.working_tree.build_targets)

        resulting_dataset = project.working_tree.make_dataset(new_tree.make_pipeline("s1"))
        compare_datasets(
            self,
            Dataset.from_iterable(
                [
                    DatasetItem(2, annotations=[Label(1)]),
                ],
                categories=["a", "b"],
            ),
            resulting_dataset,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_add_convert_stage(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

        stage = project.working_tree.build_targets.add_convert_stage("s1", DEFAULT_FORMAT)

        self.assertTrue(stage in project.working_tree.build_targets)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_add_transform_stage(self):
        class TestTransform(ItemTransform):
            def __init__(self, extractor, p1=None, p2=None):
                super().__init__(extractor)
                self.p1 = p1
                self.p2 = p2

            def transform_item(self, item):
                return self.wrap_item(item, attributes={"p1": self.p1, "p2": self.p2})

        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.working_tree.env.transforms.register("tr", TestTransform)

        # TODO: simplify adding stages and making datasets from them
        new_tree = project.working_tree.clone()
        stage = new_tree.build_targets.add_transform_stage(
            "s1", "tr", params={"p1": 5, "p2": ["1", 2, 3.5]}
        )

        self.assertTrue(stage in new_tree.build_targets)
        self.assertTrue(stage not in project.working_tree.build_targets)

        resulting_dataset = project.working_tree.make_dataset(new_tree.make_pipeline("s1"))
        compare_datasets(
            self,
            Dataset.from_iterable(
                [
                    DatasetItem(
                        1, annotations=[Label(0)], attributes={"p1": 5, "p2": ["1", 2, 3.5]}
                    ),
                    DatasetItem(
                        2, annotations=[Label(1)], attributes={"p1": 5, "p2": ["1", 2, 3.5]}
                    ),
                ],
                categories=["a", "b"],
            ),
            resulting_dataset,
        )

        project.working_tree.config.update(new_tree.config)
        self.assertTrue(stage in project.working_tree.build_targets)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_make_dataset_from_stage(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)

        built_dataset = project.working_tree.make_dataset("s1.root")

        compare_datasets(self, dataset, built_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_commit(self):
        test_dir = scope_add(TestDir())
        project = scope_add(Project.init(test_dir))

        commit_hash = project.commit("First commit", allow_empty=True)

        self.assertTrue(project.is_ref(commit_hash))
        self.assertEqual(len(project.history()), 2)
        self.assertEqual(project.history()[0], (commit_hash, "First commit"))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_commit_empty(self):
        test_dir = scope_add(TestDir())
        project = scope_add(Project.init(test_dir))

        with self.assertRaises(EmptyCommitError):
            project.commit("First commit")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_commit_patch(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_source.txt")
        os.makedirs(osp.dirname(source_url), exist_ok=True)
        with open(source_url, "w") as f:
            f.write("hello")

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", source_url, format=DEFAULT_FORMAT)
        project.commit("First commit")

        source_path = osp.join(project.source_data_dir("s1"), osp.basename(source_url))
        with open(source_path, "w") as f:
            f.write("world")

        commit_hash = project.commit("Second commit", allow_foreign=True)

        self.assertTrue(project.is_ref(commit_hash))
        self.assertNotEqual(
            project.get_rev("HEAD~1").build_targets["s1"].head.hash,
            project.working_tree.build_targets["s1"].head.hash,
        )
        self.assertTrue(project.is_obj_cached(project.working_tree.build_targets["s1"].head.hash))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_commit_foreign_changes(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_source.txt")
        os.makedirs(osp.dirname(source_url), exist_ok=True)
        with open(source_url, "w") as f:
            f.write("hello")

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", source_url, format=DEFAULT_FORMAT)
        project.commit("First commit")

        source_path = osp.join(project.source_data_dir("s1"), osp.basename(source_url))
        with open(source_path, "w") as f:
            f.write("world")

        with self.assertRaises(ForeignChangesError):
            project.commit("Second commit")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_checkout_revision(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_source.txt")
        os.makedirs(osp.dirname(source_url), exist_ok=True)
        with open(source_url, "w") as f:
            f.write("hello")

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", source_url, format=DEFAULT_FORMAT)
        project.commit("First commit")

        source_path = osp.join(project.source_data_dir("s1"), osp.basename(source_url))
        with open(source_path, "w") as f:
            f.write("world")
        project.commit("Second commit", allow_foreign=True)

        project.checkout("HEAD~1")

        compare_dirs(self, source_url, project.source_data_dir("s1"))
        with open(osp.join(test_dir, "proj", ".gitignore")) as f:
            self.assertTrue("/s1" in [line.strip() for line in f])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_checkout_sources(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.import_source("s2", url=source_url, format=DEFAULT_FORMAT)
        project.commit("Commit 1")
        project.remove_source("s1", keep_data=False)  # remove s1 from tree
        shutil.rmtree(project.source_data_dir("s2"))  # modify s2 "manually"

        project.checkout(sources=["s1", "s2"])

        compare_dirs(self, source_url, project.source_data_dir("s1"))
        compare_dirs(self, source_url, project.source_data_dir("s2"))
        with open(osp.join(test_dir, "proj", ".gitignore")) as f:
            lines = [line.strip() for line in f]
            self.assertTrue("/s1" in lines)
            self.assertTrue("/s2" in lines)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_checkout_with_force(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.import_source("s2", url=source_url, format=DEFAULT_FORMAT)
        project.commit("Commit 1")
        project.remove_source("s1", keep_data=False)  # remove s1 from tree
        shutil.rmtree(project.source_data_dir("s2"))  # modify s2 "manually"

        project.checkout(force=True)

        compare_dirs(self, source_url, project.source_data_dir("s1"))
        compare_dirs(self, source_url, project.source_data_dir("s2"))
        with open(osp.join(test_dir, "proj", ".gitignore")) as f:
            lines = [line.strip() for line in f]
            self.assertTrue("/s1" in lines)
            self.assertTrue("/s2" in lines)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_checkout_sources_from_revision(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.commit("Commit 1")
        project.remove_source("s1", keep_data=False)
        project.commit("Commit 2")

        project.checkout(rev="HEAD~1", sources=["s1"])

        compare_dirs(self, source_url, project.source_data_dir("s1"))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_check_status(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.import_source("s2", url=source_url, format=DEFAULT_FORMAT)
        project.import_source("s3", url=source_url, format=DEFAULT_FORMAT)
        project.import_source("s4", url=source_url, format=DEFAULT_FORMAT)
        project.import_source("s5", url=source_url, format=DEFAULT_FORMAT)
        project.commit("Commit 1")

        project.remove_source("s2")
        project.import_source("s6", url=source_url, format=DEFAULT_FORMAT)

        shutil.rmtree(project.source_data_dir("s3"))

        project.working_tree.build_targets.add_transform_stage("s4", "reindex")
        project.working_tree.make_dataset("s4").save()
        project.refresh_source_hash("s4")

        s5_dir = osp.join(project.source_data_dir("s5"))
        with open(osp.join(s5_dir, "annotations", "t.txt"), "w") as f:
            f.write("hello")

        status = project.status()
        self.assertEqual(
            {
                "s2": DiffStatus.removed,
                "s3": DiffStatus.missing,
                "s4": DiffStatus.modified,
                "s5": DiffStatus.foreign_modified,
                "s6": DiffStatus.added,
            },
            status,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_compare_revisions(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        project.import_source("s2", url=source_url, format=DEFAULT_FORMAT)
        rev1 = project.commit("Commit 1")

        project.remove_source("s2")
        project.import_source("s3", url=source_url, format=DEFAULT_FORMAT)
        rev2 = project.commit("Commit 2")

        diff = project.diff(rev1, rev2)
        self.assertEqual(diff, {"s2": DiffStatus.removed, "s3": DiffStatus.added})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_restore_revision(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "test_repo")
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        )
        dataset.save(source_url)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source("s1", url=source_url, format=DEFAULT_FORMAT)
        rev1 = project.commit("Commit 1")

        project.remove_cache_obj(rev1)

        self.assertFalse(project.is_rev_cached(rev1))

        head_dataset = project.head.make_dataset()

        self.assertTrue(project.is_rev_cached(rev1))
        compare_datasets(self, dataset, head_dataset)

    @mark_requirement(Requirements.DATUM_BUG_404)
    @scoped
    def test_can_add_plugin(self):
        test_dir = scope_add(TestDir())
        scope_add(Project.init(test_dir)).close()

        plugin_dir = osp.join(test_dir, ".datumaro", "plugins")
        os.makedirs(plugin_dir)
        with open(osp.join(plugin_dir, "__init__.py"), "w") as f:
            f.write(
                textwrap.dedent(
                    """
                from datumaro.components.extractor import (SubsetBase,
                    DatasetItem)

                class MyExtractor(SubsetBase):
                    def __init__(self, *args, **kwargs):
                        super().__init__()

                    def __iter__(self):
                        yield from [
                            DatasetItem('1'),
                            DatasetItem('2'),
                        ]
            """
                )
            )

        project = scope_add(Project(test_dir))
        project.import_source("src", url="", format="my")

        expected = Dataset.from_iterable([DatasetItem("1"), DatasetItem("2")])
        compare_datasets(self, expected, project.working_tree.make_dataset())

    @mark_requirement(Requirements.DATUM_BUG_402)
    @scoped
    def test_can_transform_by_name(self):
        class CustomExtractor(DatasetBase):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def __iter__(self):
                return iter(
                    [
                        DatasetItem("a"),
                        DatasetItem("b"),
                    ]
                )

        test_dir = scope_add(TestDir())
        extractor_name = "ext1"
        project = scope_add(Project.init(test_dir))
        project.env.extractors.register(extractor_name, CustomExtractor)
        project.import_source("src1", url="", format=extractor_name)
        dataset = project.working_tree.make_dataset()

        dataset = dataset.transform("reindex")

        expected = Dataset.from_iterable(
            [
                DatasetItem(1),
                DatasetItem(2),
            ]
        )
        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_modify_readonly(self):
        test_dir = scope_add(TestDir())
        dataset_url = osp.join(test_dir, "dataset")
        Dataset.from_iterable(
            [
                DatasetItem("a"),
                DatasetItem("b"),
            ]
        ).save(dataset_url)

        proj_dir = osp.join(test_dir, "proj")
        with Project.init(proj_dir) as project:
            project.import_source("source1", url=dataset_url, format=DEFAULT_FORMAT)
            project.commit("first commit")
            project.remove_source("source1")
            commit2 = project.commit("second commit")
            project.checkout("HEAD~1")
            project.remove_cache_obj(commit2)
            project.remove_cache_obj(project.working_tree.sources["source1"].hash)

        project = scope_add(Project(proj_dir, readonly=True))

        self.assertTrue(project.readonly)

        with self.subTest("add source"), self.assertRaises(ReadonlyProjectError):
            project.import_source("src1", url="", format=DEFAULT_FORMAT)

        with self.subTest("remove source"), self.assertRaises(ReadonlyProjectError):
            project.remove_source("src1")

        with self.subTest("add model"), self.assertRaises(ReadonlyProjectError):
            project.add_model("m1", launcher="x")

        with self.subTest("remove model"), self.assertRaises(ReadonlyProjectError):
            project.remove_model("m1")

        with self.subTest("checkout"), self.assertRaises(ReadonlyProjectError):
            project.checkout("HEAD")

        with self.subTest("commit"), self.assertRaises(ReadonlyProjectError):
            project.commit("third commit", allow_empty=True)

        # Can't re-download the source in a readonly project
        with self.subTest("make_dataset"), self.assertRaises(MissingObjectError):
            project.get_rev("HEAD").make_dataset()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_import_without_hashing(self):
        test_dir = scope_add(TestDir())
        dataset_url = osp.join(test_dir, "dataset")
        dataset = Dataset.from_iterable(
            [
                DatasetItem("a"),
                DatasetItem("b"),
            ]
        )
        dataset.save(dataset_url)

        proj_dir = osp.join(test_dir, "proj")
        project = scope_add(Project.init(proj_dir))
        project.import_source("source1", url=dataset_url, format=DEFAULT_FORMAT, no_hash=True)

        self.assertEqual("", project.working_tree.sources["source1"].hash)
        compare_dirs(self, dataset_url, project.source_data_dir("source1"))
        compare_datasets(self, dataset, project.working_tree.make_dataset())

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_check_status_of_unhashed(self):
        test_dir = scope_add(TestDir())
        dataset_url = osp.join(test_dir, "dataset")
        Dataset.from_iterable(
            [
                DatasetItem("a"),
                DatasetItem("b"),
            ]
        ).save(dataset_url)

        proj_dir = osp.join(test_dir, "proj")
        project = scope_add(Project.init(proj_dir))
        project.import_source("source1", url=dataset_url, format=DEFAULT_FORMAT, no_hash=True)
        project.import_source("source2", url=dataset_url, format=DEFAULT_FORMAT, no_hash=True)
        project.working_tree.build_targets.add_transform_stage("source2", "reindex")

        status = project.status()
        self.assertEqual(status["source1"], DiffStatus.added)
        self.assertEqual(status["source2"], DiffStatus.added)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_commit_unhashed(self):
        test_dir = scope_add(TestDir())
        dataset_url = osp.join(test_dir, "dataset")
        Dataset.from_iterable(
            [
                DatasetItem("a"),
                DatasetItem("b"),
            ]
        ).save(dataset_url)

        proj_dir = osp.join(test_dir, "proj")
        project = scope_add(Project.init(proj_dir))
        project.import_source("source1", url=dataset_url, format=DEFAULT_FORMAT, no_hash=True)
        project.commit("a commit")

        self.assertNotEqual("", project.working_tree.sources["source1"].hash)
        self.assertNotEqual("", project.working_tree.build_targets["source1"].head.hash)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_redownload_unhashed(self):
        test_dir = scope_add(TestDir())
        dataset_url = osp.join(test_dir, "dataset")
        Dataset.from_iterable(
            [
                DatasetItem("a"),
                DatasetItem("b"),
            ]
        ).save(dataset_url)

        proj_dir = osp.join(test_dir, "proj")
        project = scope_add(Project.init(proj_dir))
        project.import_source("source1", url=dataset_url, format=DEFAULT_FORMAT, no_hash=True)
        project.working_tree.build_targets.add_transform_stage("source1", "reindex")
        project.commit("a commit")

        with self.assertRaises(MissingSourceHashError):
            project.working_tree.make_dataset("source1.root")

    @mark_requirement(Requirements.DATUM_BUG_602)
    @scoped
    def test_can_save_local_source_with_relpath(self):
        test_dir = scope_add(TestDir())
        source_url = osp.join(test_dir, "source")
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    subset="a",
                    media=Image(data=np.ones((2, 3, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=0)],
                ),
                DatasetItem(
                    1,
                    subset="b",
                    media=Image(data=np.zeros((10, 20, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.save(source_url, save_media=True)

        project = scope_add(Project.init(osp.join(test_dir, "proj")))
        project.import_source(
            "s1", url=source_url, format=DEFAULT_FORMAT, rpath=osp.join("annotations", "b.json")
        )

        read_dataset = project.working_tree.make_dataset("s1")
        self.assertEqual(read_dataset.data_path, project.source_data_dir("s1"))

        read_dataset.save()


class BackwardCompatibilityTests_v0_1(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_migrate_old_project(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(0, subset="train", annotations=[Label(0)]),
                DatasetItem(1, subset="test", annotations=[Label(1)]),
                DatasetItem(2, subset="train", annotations=[Label(0)]),
            ],
            categories=["a", "b"],
        )

        test_dir = scope_add(TestDir())
        old_proj_dir = osp.join(test_dir, "old_proj")
        new_proj_dir = osp.join(test_dir, "new_proj")
        shutil.copytree(
            osp.join(osp.dirname(__file__), "assets", "compat", "v0.1", "project"), old_proj_dir
        )

        with self.assertLogs(None) as logs:
            Project.migrate_from_v1_to_v2(old_proj_dir, new_proj_dir, skip_import_errors=True)

            self.assertIn("Failed to migrate the source 'source3'", "\n".join(logs.output))

        project = scope_add(Project(new_proj_dir))
        loaded_dataset = project.working_tree.make_dataset()
        compare_datasets(self, expected_dataset, loaded_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_cant_load_old_project(self):
        test_dir = scope_add(TestDir())
        proj_dir = osp.join(test_dir, "old_proj")
        shutil.copytree(
            osp.join(osp.dirname(__file__), "assets", "compat", "v0.1", "project"), proj_dir
        )

        with self.assertRaises(OldProjectError):
            scope_add(Project(proj_dir))
