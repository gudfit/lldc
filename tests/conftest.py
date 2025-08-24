def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow (e.g., downloads models)"
    )
