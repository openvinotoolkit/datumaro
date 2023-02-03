import contextlib
import io
import os.path as osp
from unittest import TestCase
from unittest.case import skipIf

from datumaro.components.extractor_tfds import AVAILABLE_TFDS_DATASETS, TFDS_EXTRACTOR_AVAILABLE
from datumaro.util import parse_json
from datumaro.util.test_utils import TestDir, mock_tfds_data
from datumaro.util.test_utils import run_datum as run

from ...requirements import Requirements, mark_requirement


@skipIf(not TFDS_EXTRACTOR_AVAILABLE, reason="TFDS is not installed")
class DescribeDownloadsTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_text(self):
        with mock_tfds_data():
            output_file = io.StringIO()

            with contextlib.redirect_stdout(output_file):
                run(self, "describe-downloads")

            output = output_file.getvalue()

            # Since the output is not in a structured format, it's difficult to test
            # that it looks exactly as we want it to. As a simplification, we'll
            # just check that it contains all the data that we expect.

            for name, dataset in AVAILABLE_TFDS_DATASETS.items():
                assert f"tfds:{name}" in output

                dataset_metadata = dataset.query_remote_metadata()

                for attribute in (
                    "default_output_format",
                    "download_size",
                    "home_url",
                    "human_name",
                    "num_classes",
                    "version",
                ):
                    assert str(getattr(dataset_metadata, attribute)) in output

                expected_description = dataset_metadata.description
                # We indent the description, so it's not going to occur in the output stream
                # verbatim. Just make sure the first line is there instead.
                expected_description = expected_description.split("\n", maxsplit=1)[0]

                assert expected_description in output

                for subset_name, subset_metadata in dataset_metadata.subsets.items():
                    assert subset_name in output
                    assert str(subset_metadata.num_items) in output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_json(self):
        with mock_tfds_data():
            output_file = io.TextIOWrapper(io.BytesIO())

            with contextlib.redirect_stdout(output_file):
                run(self, "describe-downloads", "--report-format=json")

            output = parse_json(output_file.buffer.getvalue())

            assert output.keys() == {f"tfds:{name}" for name in AVAILABLE_TFDS_DATASETS}

            for name, dataset in AVAILABLE_TFDS_DATASETS.items():
                dataset_metadata = dataset.query_remote_metadata()
                dataset_description = output[f"tfds:{name}"]
                for attribute in (
                    "default_output_format",
                    "description",
                    "download_size",
                    "home_url",
                    "human_name",
                    "num_classes",
                    "version",
                ):
                    assert dataset_description.pop(attribute) == getattr(
                        dataset_metadata, attribute
                    )

                subset_descriptions = dataset_description.pop("subsets")
                assert subset_descriptions.keys() == dataset_metadata.subsets.keys()

                for subset_name, subset_metadata in dataset_metadata.subsets.items():
                    assert subset_descriptions[subset_name] == {
                        "num_items": subset_metadata.num_items
                    }

                # Make sure we checked all attributes
                assert not dataset_description

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_report_file(self):
        for format in ["text", "json"]:
            with self.subTest(format=format):
                stdout_file = io.TextIOWrapper(io.BytesIO())

                with contextlib.redirect_stdout(stdout_file):
                    run(self, "describe-downloads", "--report-format=json")

                stdout_output = stdout_file.buffer.getvalue()

                with TestDir() as test_dir:
                    redirect_path = osp.join(test_dir, "report.txt")
                    run(
                        self,
                        "describe-downloads",
                        "--report-format=json",
                        f"--report-file={redirect_path}",
                    )

                    with open(redirect_path, "rb") as redirect_file:
                        redirected_output = redirect_file.read()

                assert redirected_output == stdout_output
