# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import streamlit as st


def main():
    st.markdown((Path(__file__).parents[2] / "ABOUTUS.md").read_text())


if __name__ == "__main__":
    main()
