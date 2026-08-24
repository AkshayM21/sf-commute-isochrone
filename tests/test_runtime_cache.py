"""Compatibility checks for the extracted pure cache seam."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from core import runtime_cache  # noqa: E402
from core import server_raptor  # noqa: E402


def test_server_raptor_reexports_exact_runtime_cache_objects():
    for name in (
            "BoundedLRU", "BoundedCellCache", "_owned_payload_nbytes",
            "_array_tuple_weight", "_walkpath_tree_weight"):
        assert getattr(server_raptor, name) is getattr(runtime_cache, name)
