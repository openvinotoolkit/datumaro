from unittest.case import TestCase
import contextlib
import io
import json
import os
import os.path as osp
import shutil

from datumaro.plugins.ade20k2017_format import Ade20k2017Importer
from datumaro.plugins.ade20k2020_format import Ade20k2020Importer
from datumaro.plugins.image_dir_format import ImageDirImporter
from datumaro.util.os_util import suppress_output
from datumaro.util.test_utils import TestDir
from datumaro.util.test_utils import run_datum as run

from tests.requirements import Requirements, mark_requirement

ADE20K2017_DIR = osp.join(
    osp.dirname(__file__), '../assets/ade20k2017_dataset/dataset')
ADE20K2020_DIR = osp.join(
    osp.dirname(__file__), '../assets/ade20k2020_dataset/dataset')

class DetectFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_unambiguous(self):
        output_file = io.StringIO()

        with contextlib.redirect_stdout(output_file):
            run(self, 'detect-format', ADE20K2017_DIR)

        output = output_file.getvalue()

        self.assertIn(Ade20k2017Importer.NAME, output)
        self.assertNotIn(Ade20k2020Importer.NAME, output)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ambiguous(self):
        with TestDir() as test_dir:
            annotation_dir = osp.join(test_dir, 'training/street')
            os.makedirs(annotation_dir)

            for asset in [
                osp.join(ADE20K2017_DIR, 'training/street/1_atr.txt'),
                osp.join(ADE20K2020_DIR, 'training/street/1.json'),
            ]:
                shutil.copy(asset, annotation_dir)

            output_file = io.StringIO()

            with contextlib.redirect_stdout(output_file):
                run(self, 'detect-format', test_dir)

            output = output_file.getvalue()

            self.assertIn(Ade20k2017Importer.NAME, output)
            self.assertIn(Ade20k2020Importer.NAME, output)

    # Ideally, there should be a test for the case where no formats match,
    # but currently that's impossible, because some low-confidence detectors
    # always match.

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_rejections(self):
        output_file = io.StringIO()

        with contextlib.redirect_stdout(output_file):
            run(self, 'detect-format', '--show-rejections', ADE20K2017_DIR)

        output = output_file.getvalue()

        self.assertIn(Ade20k2017Importer.NAME, output)

        self.assertIn(Ade20k2020Importer.NAME, output)
        self.assertIn('*/**/*.json', output)

        self.assertIn(ImageDirImporter.NAME, output)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_json_report(self):
        with suppress_output(), TestDir() as test_dir:
            report_path = osp.join(test_dir, 'report.json')

            run(self, 'detect-format', '--show-rejections',
                '--json-report', report_path, ADE20K2017_DIR)

            with open(report_path, 'rb') as report_file:
                report = json.load(report_file)

        self.assertIsInstance(report, dict)
        self.assertIn('detected_formats', report)
        self.assertEqual(['ade20k2017'], report['detected_formats'])

        self.assertIn('rejected_formats', report)

        self.assertIn('ade20k2020', report['rejected_formats'])
        ade20k2020_rejection = report['rejected_formats']['ade20k2020']

        self.assertIn('reason', ade20k2020_rejection)
        self.assertEqual(ade20k2020_rejection['reason'], 'unmet_requirements')
        self.assertIn('message', ade20k2020_rejection)
        self.assertIsInstance(ade20k2020_rejection['message'], list)

        self.assertTrue(any(
            '*/**/*.json' in line for line in ade20k2020_rejection['message']))

        self.assertIn('image_dir', report['rejected_formats'])
        image_dir_rejection = report['rejected_formats']['image_dir']

        self.assertIn('reason', image_dir_rejection)
        self.assertEqual(image_dir_rejection['reason'],
            'insufficient_confidence')
        self.assertIn('message', image_dir_rejection)
        self.assertIsInstance(image_dir_rejection['message'], list)
