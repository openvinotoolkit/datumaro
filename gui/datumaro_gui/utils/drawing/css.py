# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import re

# Define a custom CSS style
custom_css = """
<style>
    .css-q8sbsg p {
        font-size: 16px;
    }
    .container-outline p {
        border: 2px solid #000; /* Adjust the border properties as needed */
        padding: 10px; /* Adjust the padding as needed */
    }
</style>
"""

# Define CSS styles for the boxes
box_style = """
    .highlight {
    border-radius: 0.4rem;
    color: white;
    padding: 0.5rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    }
    .bold {
    padding-left: 1rem;
    font-weight: 700;
    }
    .red {
    background-color: lightcoral;
    }
    .blue {
    background-color: lightblue;
    }
    .yellow {
    background-color: rgba(255,218,138,.8);
    }
    .lightgrayish {
    background-color: #B0C4DE;
    }
    .lightgray {
    background-color: #E5E5E5;
    }
    .lightmintgreen {
    background-color: #A9DFBF;
    }
    .lightpurple {
    background-color: #C9A0DC;
    }
    .box {
    width: auto;
    max-width: 1000px;
    margin: "5px 5px 5px 5px";
    }
    .smallbox {
    width: auto;
    max-width: 1000px;
    margin: "1px 1px 1px 1px";
    }
    .stat_highlight {
    border-radius: 0.4rem;
    color: white;
    padding: 5px;
    width: auto;
    margin: "5px 5px 5px 5px";
    }
"""

# Define CSS styles for the buttons
btn_style = """
    button {
    margin-top: 12px !important;
    padding-top: 1px;
    padding-bottom: 1px;
    }
"""


def parse_css_string(css_string):
    css_dict = {}
    # Use regular expressions to extract CSS class names and styles
    matches = re.findall(r"\.(.*?)\s*{([^}]*)}", css_string)
    for match in matches:
        class_name, class_styles = match
        # Convert each class's styles to a dictionary and store
        style_dict = {}
        for style_pair in class_styles.split(";"):
            style_pair = style_pair.strip()
            if style_pair:
                key, value = style_pair.split(":")
                style_dict[key.strip()] = value.strip()
        css_dict[class_name.strip()] = style_dict
    return css_dict


def apply_css_styles(css_string, *class_names):
    css_dict = parse_css_string(css_string)
    class_styles = [css_dict.get(class_name, {}) for class_name in class_names]
    return merge_styles(*class_styles)


def merge_styles(*style_dicts):
    merged_styles = {}
    for style_dict in style_dicts:
        for key, value in style_dict.items():
            merged_styles.setdefault(key, []).append(value)
    return {key: " ".join(values) for key, values in merged_styles.items()}
