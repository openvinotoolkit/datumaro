from unittest.case import TestCase
import os.path as osp

from datumaro.cli.util.project import (
    WrongRevpathError, parse_full_revpath, split_local_revpath,
)
from datumaro.components.dataset import DEFAULT_FORMAT, Dataset, IDataset
from datumaro.components.errors import (
    MultipleFormatsMatchError, ProjectNotFoundError, UnknownTargetError,
)
from datumaro.components.extractor import DatasetItem
from datumaro.components.project import Project
from datumaro.util.scope import scoped
from datumaro.util.test_utils import TestDir
import datumaro.util.scope as scope

from ..requirements import Requirements, mark_requirement


class TestRevpath(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_parse(self):
        test_dir = scope.add(TestDir(frame_id=5))

        dataset_url = osp.join(test_dir, 'source')
        Dataset.from_iterable([DatasetItem(1)]).save(dataset_url)

        proj_dir = osp.join(test_dir, 'proj')
        proj = scope.add(Project.init(proj_dir))
        proj.import_source('source-1', dataset_url, format=DEFAULT_FORMAT)
        ref = proj.commit("second commit", allow_empty=True)

        with self.subTest("project"):
            dataset, project = parse_full_revpath(proj_dir)
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertTrue(isinstance(project, Project))

        with self.subTest("project ref"):
            dataset, project = parse_full_revpath(f"{proj_dir}@{ref}")
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertTrue(isinstance(project, Project))

        with self.subTest("project ref source"):
            dataset, project = parse_full_revpath(f"{proj_dir}@{ref}:source-1")
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertTrue(isinstance(project, Project))

        with self.subTest("project ref source stage"):
            dataset, project = parse_full_revpath(
                f"{proj_dir}@{ref}:source-1.root")
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertTrue(isinstance(project, Project))

        with self.subTest("ref"):
            dataset, project = parse_full_revpath(ref, proj)
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("ref source"):
            dataset, project = parse_full_revpath(f"{ref}:source-1", proj)
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("ref source stage"):
            dataset, project = parse_full_revpath(f"{ref}:source-1.root", proj)
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("source"):
            dataset, project = parse_full_revpath("source-1", proj)
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("source stage"):
            dataset, project = parse_full_revpath("source-1.root", proj)
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("dataset (in context)"):
            with self.assertRaises(WrongRevpathError) as cm:
                parse_full_revpath(dataset_url, proj)
            self.assertEqual(
                {UnknownTargetError, MultipleFormatsMatchError},
                set(type(e) for e in cm.exception.problems)
            )

        with self.subTest("dataset format (in context)"):
            dataset, project = parse_full_revpath(
                f"{dataset_url}:datumaro", proj)
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

        with self.subTest("dataset (no context)"):
            with self.assertRaises(WrongRevpathError) as cm:
                parse_full_revpath(dataset_url)
            self.assertEqual(
                {ProjectNotFoundError, MultipleFormatsMatchError},
                set(type(e) for e in cm.exception.problems)
            )

        with self.subTest("dataset format (no context)"):
            dataset, project = parse_full_revpath(f"{dataset_url}:datumaro")
            if project:
                scope.add(project)
            self.assertTrue(isinstance(dataset, IDataset))
            self.assertEqual(None, project)

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
