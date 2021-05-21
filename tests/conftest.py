# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT


def pytest_configure(config):
    # register an additional markers
    config.addinivalue_line("markers", "unit: mark a specific test with the unit test type")
    config.addinivalue_line("markers", "component: mark a specific test with the component test type")
    config.addinivalue_line("markers", "priority_low: mark a test with the priority low")
    config.addinivalue_line("markers", "priority_medium: mark a test with the priority medium")
    config.addinivalue_line("markers", "priority_high: mark a test with the priority high")
    config.addinivalue_line("markers", "components(component_name): mark a specific test with the component name")
    config.addinivalue_line("markers", "reqids(requirement): mark a specific test with the github issue requirement")
    config.addinivalue_line("markers", "bugs(bug_name): mark a specific test with the bug indication (github issue)")
    config.addinivalue_line("markers", "skip(reason): skip a specific test because of reason")

