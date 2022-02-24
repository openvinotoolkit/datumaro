# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import contextlib
import sys
from typing import Dict, Type

from datumaro.components.extractor_tfds import (
    AVAILABLE_TFDS_DATASETS,
    TFDS_EXTRACTOR_AVAILABLE,
    TfdsDatasetRemoteMetadata,
)
from datumaro.util import dump_json

from ..util import MultilineFormatter


def build_parser(
    parser_ctor: Type[argparse.ArgumentParser] = argparse.ArgumentParser,
):
    parser = parser_ctor(
        help="Print information about downloadable datasets",
        description="""
        TBD
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "--report-format",
        choices=("text", "json"),
        default="text",
        help="Format in which to report the information",
    )
    parser.add_argument(
        "--report-file", help="File to which to write the report (default: standard output)"
    )
    parser.set_defaults(command=describe_downloads_command)

    return parser


def get_sensitive_args():
    return {
        describe_downloads_command: [],
    }


def describe_downloads_command(args):
    dataset_metas: Dict[str, TfdsDatasetRemoteMetadata] = {}

    if TFDS_EXTRACTOR_AVAILABLE:
        for dataset_name, dataset in AVAILABLE_TFDS_DATASETS.items():
            dataset_metas[f"tfds:{dataset_name}"] = dataset.query_remote_metadata()

    if args.report_format == "text":
        with (
            open(args.report_file, "w") if args.report_file else contextlib.nullcontext(sys.stdout)
        ) as report_file:
            if dataset_metas:
                print("Available datasets:", file=report_file)

                for name, meta in sorted(dataset_metas.items()):
                    print(file=report_file)
                    print(f"{name} ({meta.human_name}):", file=report_file)
                    print(
                        f"  default output format: {meta.default_output_format}",
                        file=report_file,
                    )

                    print("  description:", file=report_file)
                    for line in meta.description.rstrip("\n").split("\n"):
                        print(f"    {line}", file=report_file)

                    print(f"  download size: {meta.download_size} bytes", file=report_file)
                    print(f"  home URL: {meta.home_url or 'N/A'}", file=report_file)
                    print(f"  number of classes: {meta.num_classes}", file=report_file)
                    print("  subsets:", file=report_file)
                    for subset_name, subset_meta in sorted(meta.subsets.items()):
                        print(f"    {subset_name}: {subset_meta.num_items} items", file=report_file)
                    print(f"  version: {meta.version}", file=report_file)
            else:
                print("No datasets available.", file=report_file)
                print(file=report_file)
                print(
                    "You can enable TFDS datasets by installing "
                    "TensorFlow and TensorFlow Datasets:",
                    file=report_file,
                )
                print("    pip install datumaro[tf,tfds]", file=report_file)

    elif args.report_format == "json":

        def meta_to_raw(meta: TfdsDatasetRemoteMetadata):
            raw = {}

            # We omit the media type from the output, because there is currently no mechanism
            # for mapping media types to strings. The media type could be useful information
            # for users, though, so we might want to implement such a mechanism eventually.

            for attribute in (
                "default_output_format",
                "description",
                "download_size",
                "home_url",
                "human_name",
                "num_classes",
                "version",
            ):
                raw[attribute] = getattr(meta, attribute)

            raw["subsets"] = {
                name: {"num_items": subset.num_items} for name, subset in meta.subsets.items()
            }

            return raw

        with (
            open(args.report_file, "wb")
            if args.report_file
            else contextlib.nullcontext(sys.stdout.buffer)
        ) as report_file:
            report_file.write(
                dump_json(
                    {name: meta_to_raw(meta) for name, meta in dataset_metas.items()},
                    indent=True,
                    append_newline=True,
                )
            )
    else:
        assert False, "unreachable code"
