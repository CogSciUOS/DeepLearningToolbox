"""Testsuite for the `dltb.base` package.
"""

# standard imports
from typing import Iterable
from pathlib import Path
import unittest

# toolbox imports
from dltb import base

class TestBasePackage(unittest.TestCase):
    """Tests for the `dltb.base` package.
    """

    @property
    def package_directory(self) -> Path:
        """The directory of the `dltb.base` package.
        """
        return Path(base.__file__).parent

    @property
    def test_directory(self) -> Path:
        """The test directory for the `dltb.base` package.
        """
        return self.package_directory / 'tests'

    def base_modules(self) -> Iterable[str]:
        """The modules defined in the `dltb.base` package.
        """
        for path in self.package_directory.glob('*.py'):
            yield str(path.relative_to(self.package_directory).
                      with_suffix(''))
        yield 'image'

    def base_tests(self) -> Iterable[str]:
        """The tests defined for the  `dltb.base` package.
        """
        for path in self.test_directory.glob('test_*.py'):
            yield str(path.relative_to(self.test_directory).
                      with_suffix(''))[5:]

    def test_untested_modules(self):
        """Check for untested modules.
        """
        modules = set(self.base_modules())
        tests = set(self.base_tests())
        untested_ok = set(('__init__', ))
        # FIXME[hack]: there should be tests for all modules
        untested_ok |= set([
            'sound', 'install', 'hardware', 'fail',
            'types', 'store', 'observer',
            'state', 'resource',
            'background', 'metadata', 'gui',
            'reuse', 'info'
        ])
        self.assertEqual(modules - tests, untested_ok)

    def test_unused_tests(self):
        """Check for unused tests.
        """
        modules = set(self.base_modules())
        tests = set(self.base_tests())
        unused_ok = set(('00package', ))
        self.assertEqual(tests - modules, unused_ok)
