
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from contextlib import contextmanager
from io import StringIO
import logging


@contextmanager
def logging_disabled(max_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable
    logging.disable(max_level)
    try:
        yield
    finally:
        logging.disable(previous_level)

@contextmanager
def catch_logs(logger=None):
    logger = logging.getLogger(logger)

    old_propagate = logger.propagate
    prev_handlers = logger.handlers

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger.handlers = [handler]
    logger.propagate = False

    try:
        yield stream
    finally:
        logger.handlers = prev_handlers
        logger.propagate = old_propagate