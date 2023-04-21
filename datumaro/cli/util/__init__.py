# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import textwrap
from typing import Iterable, List


def add_subparser(subparsers, name, builder):
    return builder(lambda **kwargs: subparsers.add_parser(name, **kwargs))


class MultilineFormatter(argparse.HelpFormatter):
    """
    Keeps line breaks introduced with '|n' separator
    and spaces introduced with '|s',
    also removes the 'code-block' directives.
    """

    def __init__(self, keep_natural=False, **kwargs):
        super().__init__(**kwargs)
        self._keep_natural = keep_natural

    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(" ", text).strip()
        text = text.replace("|s", " ").replace(".. code-block::", "|n")

        paragraphs = text.split("|n ")
        if self._keep_natural:
            paragraphs = sum((p.split("\n ") for p in paragraphs), [])

        multiline_text = ""
        for paragraph in paragraphs:
            formatted_paragraph = (
                textwrap.fill(paragraph, width, initial_indent=indent, subsequent_indent=indent)
                + "\n"
            )
            multiline_text += formatted_paragraph
        return multiline_text

    def _format_action(self, action):
        """
        This requires to remove an ugly curly braced list generated by add_subparsers().
        Please see the following example.

        <w/o this patch>
            Commands:
            {get,describe}
                get           Download a publicly available dataset
                describe      Print information about downloadable datasets

        <w/ this patch>
            Commands:
                get           Download a publicly available dataset
                describe      Print information about downloadable datasets
        """
        parts = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


def required_count(nmin=0, nmax=0):
    assert 0 <= nmin and 0 <= nmax and nmin or nmax

    class RequiredCount(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            k = len(values)
            if not ((nmin and (nmin <= k) or not nmin) and (nmax and (k <= nmax) or not nmax)):
                msg = "Argument '%s' requires" % self.dest
                if nmin and nmax:
                    msg += " from %s to %s arguments" % (nmin, nmax)
                elif nmin:
                    msg += " at least %s arguments" % nmin
                else:
                    msg += " no more %s arguments" % nmax
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)

    return RequiredCount


def at_least(n):
    return required_count(n, 0)


def join_cli_args(args: argparse.Namespace, *names: Iterable[str]) -> List:
    "Merges arg values in a list"

    joined = []

    for name in names:
        value = getattr(args, name)
        if not isinstance(value, list):
            value = [value]
        joined += value

    return joined


def show_video_import_warning():
    log.warning(
        "Using 'video_frames' in a project may lead "
        "to different results across multiple runs, if the "
        "system setup changes (library version, OS, etc.). "
        "If you need stable results, consider splitting the video "
        "manually using instructions at: "
        "https://openvinotoolkit.github.io/datumaro/latest/docs/data-formats/media_formats"
    )


def make_subcommands_help(commands, help_line_start=0):
    desc = ""
    for command_name, command_type, command_help in commands:
        msg = ("  %-" + str(max(0, help_line_start - 2 - 1)) + "s%s\n") % (
            command_name,
            command_help,
        )
        if command_type is None:
            msg = msg.lstrip()  # un-indent for not subparser
        desc += msg

    return desc
