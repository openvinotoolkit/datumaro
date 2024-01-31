# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import re
from contextlib import contextmanager
from pathlib import Path

import requests
import streamlit as st

FILTER_SHARE = re.compile(r"^.*\[share_\w+\].*$", re.MULTILINE)


@contextmanager
def readme(project, usage=None, source=None):
    content = requests.get(f"https://raw.githubusercontent.com/okld/{project}/main/README.md").text
    st.markdown(FILTER_SHARE.sub("", content))

    demo = st.container()

    if usage or source:
        st.title("")

    if usage:
        with st.expander("USAGE"):
            st.help(usage)

    if source:
        with st.expander("SOURCE"):
            st.code(Path(source).read_text())

    with demo:
        yield


github_pypi_desc = """
:factory: Dataset management &nbsp; [![GitHub][github_badge]][github_link] [![PyPI][pypi_badge]][pypi_link]
=====================

Import a dataset and manipulate it!

[github_badge]: https://badgen.net/badge/icon/GitHub?icon=github&color=black&label
[github_link]: https://github.com/openvinotoolkit/datumaro

[pypi_badge]: https://badgen.net/pypi/v/streamlit-elements?icon=pypi&color=black&label
[pypi_link]: https://pypi.org/project/datumaro/
"""
