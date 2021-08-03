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
from datumaro.util.test_utils import TestDir

from ..requirements import Requirements, mark_requirement


class TestRevpath(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_parse(self):
        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, 'source')
            dataset = Dataset.from_iterable([DatasetItem(1)])
            dataset.save(dataset_url)

            proj_dir = osp.join(test_dir, 'proj')
            proj = Project.init(proj_dir)
            proj.import_source('source-1', dataset_url, format=DEFAULT_FORMAT)
            ref = proj.commit("second commit", allow_empty=True)


            with self.subTest("project"):
                self.assertTrue(isinstance(parse_full_revpath(proj_dir, None),
                    IDataset))

            with self.subTest("project ref"):
                self.assertTrue(isinstance(
                    parse_full_revpath(proj_dir + "@" + ref, None),
                    IDataset))

            with self.subTest("project ref source"):
                self.assertTrue(isinstance(
                    parse_full_revpath(proj_dir + "@" + ref + ":source-1", None),
                    IDataset))

            with self.subTest("ref"):
                self.assertTrue(isinstance(
                    parse_full_revpath(ref, proj),
                    IDataset))

            with self.subTest("ref source"):
                self.assertTrue(isinstance(
                    parse_full_revpath(ref + ":source-1", proj),
                    IDataset))

            with self.subTest("source"):
                self.assertTrue(isinstance(
                    parse_full_revpath("source-1", proj),
                    IDataset))

            with self.subTest("dataset (in context)"):
                with self.assertRaises(WrongRevpathError) as cm:
                    parse_full_revpath(dataset_url, proj)
                self.assertEqual(
                    {UnknownTargetError, MultipleFormatsMatchError},
                    set(type(e) for e in cm.exception.problems)
                )

            with self.subTest("dataset format (in context)"):
                self.assertTrue(isinstance(
                    parse_full_revpath(dataset_url + ":datumaro", proj),
                    IDataset))

            with self.subTest("dataset (no context)"):
                with self.assertRaises(WrongRevpathError) as cm:
                    parse_full_revpath(dataset_url, None)
                self.assertEqual(
                    {ProjectNotFoundError, MultipleFormatsMatchError},
                    set(type(e) for e in cm.exception.problems)
                )

            with self.subTest("dataset format (no context)"):
                self.assertTrue(isinstance(
                    parse_full_revpath(dataset_url + ":datumaro", None),
                    IDataset))

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
