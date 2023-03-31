# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pyarrow as pa


def pa_batches_decoder(batches, column=None):
    if not isinstance(batches, list):
        batches = [batches]
    table = pa.Table.from_batches(batches)
    if column:
        data = table.column(column).to_pylist()
    else:
        data = table.to_pylist()
    return data
