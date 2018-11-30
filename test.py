import os
import unittest

# todo: switch to pytest or tox?

if __name__ == "__main__":
    loader = unittest.TestLoader()
    tests_dir = os.path.join(os.path.dirname(__file__), "tests")
    tests = loader.discover(tests_dir, pattern="*.py")
    runner = unittest.TextTestRunner()
    result = runner.run(tests)
    if result.failures or result.errors:
        raise SystemExit(f"{len(result.failures) + len(result.errors)} tests failed")

