"""Shared helpers for reading the R5 golden oracles (tests/raptor_golden/) against a live
RaptorEngine. numpy-only and JVM-free — do NOT import r5py (or pandas) here.

The one rule this module exists to enforce: oracle arrays are aligned to the ENGINE's cell
order BY CELL ID, never by position. Oracle row order happening to equal engine cell order is
a coincidence of the current bake — the grid already changed size once (3068 -> 2999 on the
neighborhoods-dataset swap), and a positional read after a regen would silently score the
WRONG cell (or IndexError on a count change).
"""
import numpy as np


def purewalk_aligned(engine, z):
    """Oracle pure-walk seconds aligned to ``engine.cell_ids`` by cell id (-1 where the oracle
    has no row for a cell). ``z`` is a loaded oracle npz with ``cell_ids`` + ``purewalk``."""
    pw = np.full(len(engine.cell_ids), -1, dtype=np.int64)
    opos = {c: i for i, c in enumerate(z["cell_ids"].astype(str))}
    for i, c in enumerate(engine.cell_ids):
        j = opos.get(c)
        if j is not None:
            pw[i] = int(z["purewalk"][j])
    return pw
