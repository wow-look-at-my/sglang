"""Make ``import test_selector`` work: the precise-test dir is not a package."""

import sys
from pathlib import Path

_PRECISE_TEST_DIR = str(Path(__file__).resolve().parent.parent)
if _PRECISE_TEST_DIR not in sys.path:
    sys.path.insert(0, _PRECISE_TEST_DIR)
