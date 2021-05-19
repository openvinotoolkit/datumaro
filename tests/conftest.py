def pytest_configure(config):
    # register an additional markers
    config.addinivalue_line("markers", "component": mark specific test type")
    config.addinivalue_line("markers", "priority_low: mark tests with priority low")
    config.addinivalue_line("markers", "priority_medium: mark specific test type")
    config.addinivalue_line("markers", "priority_high: mark specific test type")
    config.addinivalue_line("markers", "components(component_name): mark specific component name")
    config.addinivalue_line("markers", "reqids(requirement_name): mark specific component name")
    config.addinivalue_line("markers", "bugs(bug_name): mark specific component name")

