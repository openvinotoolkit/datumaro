# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Callable

import streamlit as st


def page_group(param):
    key = f"{__name__}_page_group_{param}"

    if key not in st.session_state:
        st.session_state[key] = PageGroup(param)

    return st.session_state[key]


class PageGroup:
    def __init__(self, param):
        self._param: str = param
        self._default: str = None
        self._selected: str = None

    @property
    def selected(self) -> str:
        params = st.query_params.to_dict()
        return params.get(self._param, self._default)

    def item(self, label: str, callback: Callable, default=False) -> None:
        key = f"{__name__}_{self._param}_{label}"
        page = self._normalize_label(label)

        if default:
            self._default = page

        selected = page == self.selected

        if selected:
            self._selected = callback

        st.session_state[key] = selected
        st.checkbox(label, key=key, disabled=selected, on_change=self._on_change, args=(page,))

    def show(self) -> None:
        if self._selected is not None:
            self._selected()
        else:
            st.title("🤷 404 Not Found")

    def _on_change(self, page: str) -> None:
        params = st.query_params.to_dict()
        params[self._param] = [page]
        for k, v in params.items():
            st.query_params[k] = v

    def _normalize_label(self, label: str) -> str:
        return "".join(char.lower() for char in label if char.isascii()).strip().replace(" ", "-")
