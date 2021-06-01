# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

def pytest_configure(config):
    # register additional markers
    config.addinivalue_line("markers", "unit: mark a test as unit test")
    config.addinivalue_line("markers", "component: mark a test a component test")
    config.addinivalue_line("markers", "cli: mark a test a CLI test")

    config.addinivalue_line("markers", "priority_low: mark a test as low priority")
    config.addinivalue_line("markers", "priority_medium: mark a test as medium priority")
    config.addinivalue_line("markers", "priority_high: mark a test as high priority")

    config.addinivalue_line("markers", "components(ids): link a test with a component")
    config.addinivalue_line("markers", "reqids(ids): link a test with a requirement")
    config.addinivalue_line("markers", "bugs(ids): link a test with a bug")
