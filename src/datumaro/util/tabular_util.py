# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import re

emoji_pattern = re.compile(
    "|".join(
        [
            "[\U0001F600-\U0001F64F]",  # emoticons
            "[\U0001F300-\U0001F5FF]",  # symbols & pictographs
            "[\U0001F680-\U0001F6FF]",  # transport & map symbols
            "[\U0001F1E0-\U0001F1FF]",  # flags (iOS)
            "[\u2600-\u26FF]",  # Miscellaneous Symbols
            "[\u2700-\u27BF]",  # Dingbats
            "[\U0001F900-\U0001F9FF]",  # Supplemental Symbols and Pictographs
        ]
    ),
    flags=re.UNICODE,
)
