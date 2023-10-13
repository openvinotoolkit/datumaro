import streamlit as st

from pathlib import Path


def main():
    st.markdown((Path(__file__).parents[2]/"ABOUTUS.md").read_text())


if __name__ == "__main__":
    main()
