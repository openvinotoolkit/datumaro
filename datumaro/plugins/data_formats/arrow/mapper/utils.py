# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base64

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


def b64encode(obj, prefix=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = b64encode(v, prefix)
    elif isinstance(obj, (list, tuple)):
        _obj = []
        for v in obj:
            _obj.append(b64encode(v, prefix))
        if isinstance(obj, list):
            _obj = list(_obj)
        obj = _obj
    elif isinstance(obj, bytes):
        obj = base64.b64encode(obj).decode()
        if prefix:
            obj = prefix + obj
    return obj


def b64decode(obj, prefix=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = b64decode(v, prefix)
    elif isinstance(obj, (list, tuple)):
        _obj = []
        for v in obj:
            _obj.append(b64decode(v, prefix))
        if isinstance(obj, list):
            _obj = list(_obj)
        obj = _obj
    elif isinstance(obj, str):
        if prefix and obj.startswith(prefix):
            obj = obj.replace(prefix, "", 1)
            obj = base64.b64decode(obj)
        else:
            try:
                _obj = base64.b64decode(obj)
                if base64.b64encode(_obj).decode() == obj:
                    obj = _obj
            except Exception:
                pass
    return obj
