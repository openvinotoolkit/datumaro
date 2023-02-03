import os
import os.path as osp
import shutil
from unittest.case import TestCase

from datumaro.cli.util.project import WrongRevpathError, parse_full_revpath, split_local_revpath
from datumaro.components.dataset import DEFAULT_FORMAT, Dataset, IDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import (
    MultipleFormatsMatchError,
    ProjectNotFoundError,
    UnknownTargetError,
)
from datumaro.components.project import Project
from datumaro.util.scope import scope_add, scoped
from datumaro.util.test_utils import TestDir

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path


class TestRevpath(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_parse(self):
        test_dir = scope_add(TestDir())

        dataset_url = osp.join(test_dir, "source")
        Dataset.from_iterable([DatasetItem(1)]).save(dataset_url)

        proj_dir = osp.join(test_dir, "proj")
        proj = scope_add(Project.init(proj_dir))
        proj.import_source("source-1", dataset_url, format=DEFAULT_FORMAT)
        ref = proj.commit("second commit", allow_empty=True)

        with self.subTest("project"):
            dataset, project = parse_full_revpath(proj_dir)
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertTrue(isinstance(project, Project))

        with self.subTest("project ref"):
            dataset, project = parse_full_revpath(f"{proj_dir}@{ref}")
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertTrue(isinstance(project, Project))

        with self.subTest("project ref source"):
            dataset, project = parse_full_revpath(f"{proj_dir}@{ref}:source-1")
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertTrue(isinstance(project, Project))

        with self.subTest("project ref source stage"):
            dataset, project = parse_full_revpath(f"{proj_dir}@{ref}:source-1.root")
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertTrue(isinstance(project, Project))

        with self.subTest("ref"):
            dataset, project = parse_full_revpath(ref, proj)
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("ref source"):
            dataset, project = parse_full_revpath(f"{ref}:source-1", proj)
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("ref source stage"):
            dataset, project = parse_full_revpath(f"{ref}:source-1.root", proj)
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("source"):
            dataset, project = parse_full_revpath("source-1", proj)
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("source stage"):
            dataset, project = parse_full_revpath("source-1.root", proj)
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("dataset (in context)"):
            dataset, project = parse_full_revpath(dataset_url, proj)
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("dataset format (in context)"):
            dataset, project = parse_full_revpath(f"{dataset_url}:datumaro", proj)
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("dataset (no context)"):
            dataset, project = parse_full_revpath(dataset_url)
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("dataset format (no context)"):
            dataset, project = parse_full_revpath(f"{dataset_url}:datumaro")
            if project:
                scope_add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_ambiguous_format(self):
        test_dir = scope_add(TestDir())

        dataset_url = osp.join(test_dir, "source")

        # create an ambiguous dataset by merging annotations from
        # datasets in different formats
        annotation_dir = osp.join(dataset_url, "training/street")
        assets_dir = get_test_asset_path()
        os.makedirs(annotation_dir)
        for asset in [
            "ade20k2017_dataset/dataset/training/street/1_atr.txt",
            "ade20k2020_dataset/dataset/training/street/1.json",
        ]:
            shutil.copy(osp.join(assets_dir, asset), annotation_dir)

        with self.subTest("no context"):
            with self.assertRaises(WrongRevpathError) as cm:
                parse_full_revpath(dataset_url)
            self.assertEqual(
                {ProjectNotFoundError, MultipleFormatsMatchError},
                set(type(e) for e in cm.exception.problems),
            )

        proj_dir = osp.join(test_dir, "proj")
        proj = scope_add(Project.init(proj_dir))

        with self.subTest("in context"):
            with self.assertRaises(WrongRevpathError) as cm:
                parse_full_revpath(dataset_url, proj)
            self.assertEqual(
                {UnknownTargetError, MultipleFormatsMatchError},
                set(type(e) for e in cm.exception.problems),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_split_local_revpath(self):
        with self.subTest("full"):
            self.assertEqual(("rev", "tgt"), split_local_revpath("rev:tgt"))

        with self.subTest("rev only"):
            self.assertEqual(("rev", ""), split_local_revpath("rev:"))

        with self.subTest("build target only"):
            self.assertEqual(("", "tgt"), split_local_revpath("tgt"))

        with self.subTest("build target only (empty rev)"):
            self.assertEqual(("", "tgt"), split_local_revpath(":tgt"))
