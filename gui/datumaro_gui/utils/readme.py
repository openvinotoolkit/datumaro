import re
import requests
import streamlit as st

from contextlib import contextmanager
from pathlib import Path

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
