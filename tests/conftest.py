# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

def pytest_configure(config):
    # register additional markers
    config.addinivalue_line("markers", "unit: mark a test as unit test")
    config.addinivalue_line("markers", "e2e: mark a test as end-to-end test")

    config.addinivalue_line("markers", "priority_low: mark a test as low priority")
    config.addinivalue_line("markers", "priority_medium: mark a test as medium priority")
    config.addinivalue_line("markers", "priority_high: mark a test as high priority")

    config.addinivalue_line("markers", "components(ids): link a test with a component")
    config.addinivalue_line("markers", "reqids(ids): link a test with a requirement")

