from unittest import TestCase
import os.path as osp

from datumaro.components.format_detection import (
    FormatDetectionConfidence, FormatRequirementsUnmet, apply_format_detector,
)
from datumaro.util.test_utils import TestDir

from tests.requirements import Requirements, mark_requirement


class FormatDetectionTest(TestCase):
    def setUp(self) -> None:
        test_dir_context = TestDir()
        self._dataset_root = test_dir_context.__enter__()
        self.addCleanup(test_dir_context.__exit__, None, None, None)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_empty_detector(self):
        result = apply_format_detector(self._dataset_root, lambda c: None)
        self.assertEqual(result, FormatDetectionConfidence.MEDIUM)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_custom_confidence(self):
        result = apply_format_detector(self._dataset_root,
            lambda c: FormatDetectionConfidence.LOW)
        self.assertEqual(result, FormatDetectionConfidence.LOW)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_root_path(self):
        provided_root = None
        def detect(context):
            nonlocal provided_root
            provided_root = context.root_path

        apply_format_detector(self._dataset_root, detect)
        self.assertEqual(provided_root, self._dataset_root)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_fail(self):
        def detect(context):
            context.fail('abcde')

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives, ('abcde',))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_require_file_success(self):
        with open(osp.join(self._dataset_root, 'foobar.txt'), 'w'):
            pass

        selected_file = None
        def detect(context):
            nonlocal selected_file
            selected_file = context.require_file('**/[fg]oo*.t?t')

        result = apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result, FormatDetectionConfidence.MEDIUM)
        self.assertEqual(selected_file, 'foobar.txt')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_require_file_failure(self):
        with open(osp.join(self._dataset_root, 'foobar.txt'), 'w'):
            pass

        def detect(context):
            context.require_file('*/*')

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(len(result.exception.failed_alternatives), 1)
        self.assertIn('*/*', result.exception.failed_alternatives[0])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_require_file_exclude_fname_one(self):
        with open(osp.join(self._dataset_root, 'foobar.txt'), 'w'):
            pass

        def detect(context):
            context.require_file('foobar.*', exclude_fnames='*.txt')

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(len(result.exception.failed_alternatives), 1)
        self.assertIn('foobar.*', result.exception.failed_alternatives[0])
        self.assertIn('*.txt', result.exception.failed_alternatives[0])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_require_file_exclude_fname_many(self):
        for ext in ('txt', 'lst'):
            with open(osp.join(self._dataset_root, f'foobar.{ext}'), 'w'):
                pass

        def detect(context):
            context.require_file('foobar.*', exclude_fnames=('*.txt', '*.lst'))

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(len(result.exception.failed_alternatives), 1)
        self.assertIn('foobar.*', result.exception.failed_alternatives[0])
        self.assertIn('*.txt', result.exception.failed_alternatives[0])
        self.assertIn('*.lst', result.exception.failed_alternatives[0])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_probe_text_file_success(self):
        with open(osp.join(self._dataset_root, 'foobar.txt'), 'w') as f:
            print('123', file=f)

        def detect(context):
            with context.probe_text_file('foobar.txt', 'abcde') as f:
                if next(f) != '123\n':
                    raise Exception

        result = apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result, FormatDetectionConfidence.MEDIUM)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_probe_text_file_failure_bad_file(self):
        def detect(context):
            with context.probe_text_file('foobar.txt', 'abcde'):
                pass

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives,
            ('foobar.txt: abcde',))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_probe_text_file_failure_exception(self):
        with open(osp.join(self._dataset_root, 'foobar.txt'), 'w'):
            pass

        def detect(context):
            with context.probe_text_file('foobar.txt', 'abcde'):
                raise Exception

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives,
            ('foobar.txt: abcde',))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_probe_text_file_nested_req(self):
        with open(osp.join(self._dataset_root, 'foobar.txt'), 'w'):
            pass

        def detect(context):
            with context.probe_text_file('foobar.txt', 'abcde'):
                context.fail('abcde')

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives,
            ('abcde',))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_require_any_success(self):
        alternatives_executed = set()

        def detect(context):
            nonlocal alternatives_executed
            with context.require_any():
                with context.alternative():
                    alternatives_executed.add(1)
                    context.fail('bad alternative 1')
                with context.alternative():
                    alternatives_executed.add(2)
                    # good alternative 2
                with context.alternative():
                    alternatives_executed.add(3)
                    context.fail('bad alternative 3')

        result = apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result, FormatDetectionConfidence.MEDIUM)
        self.assertEqual(alternatives_executed, {1, 2, 3})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_require_any_failure(self):
        def detect(context):
            with context.require_any():
                with context.alternative():
                    context.fail('bad alternative 1')
                with context.alternative():
                    context.fail('bad alternative 2')

        with self.assertRaises(FormatRequirementsUnmet) as result:
            apply_format_detector(self._dataset_root, detect)

        self.assertEqual(result.exception.failed_alternatives,
            ('bad alternative 1', 'bad alternative 2'))
