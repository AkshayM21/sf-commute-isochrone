"""Focused contract tests for the server's committed-MC seed policy."""

import inspect
import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def test_mc_seed_is_a_shared_explicit_deterministic_value():
    from core import server_raptor as sr

    assert isinstance(sr.MC_BASE_SEED, int)
    assert sr.mc_seed() == sr.MC_BASE_SEED
    assert sr.mc_seed() == sr.mc_seed()
    source = inspect.getsource(sr)
    assert "hashlib" not in source
    assert "sha256" not in source
