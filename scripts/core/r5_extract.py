"""Shared parsers for R5 ``TravelTimeMatrix`` DataFrames (egress + pure-walk extraction).

ONE implementation used by BOTH sides of the RAPTOR validation contract:
  * scripts/raptor_oracle.py — writes the golden oracles (egress_g/egress_w/purewalk) the
    JVM-free validator feeds the engine, and
  * scripts/server.py        — the legacy R5 branch of ``_raptor_egress_purewalk`` (the same
    per-workplace inputs, computed live when USE_WALK_GRAPH=0).
If these two parsers drift (dedup rule, rounding, sentinel), the goldens stop testing what
the server actually serves — so the min-dedup, minutes->seconds rounding and the -1
"unreachable" sentinel live here exactly once.

JVM-free and import-light: this module takes ALREADY-COMPUTED DataFrames (it never touches
r5py) and imports pandas lazily inside the functions, so importing it costs nothing on the
lean boot path. numpy is a hard dependency of every core module already.
"""
import numpy as np


def tt_col(df):
    """The travel-time column of an R5 TravelTimeMatrix frame: ``travel_time`` when a single
    percentile was requested, else the first ``travel_time_pXX`` column. (This pick was
    previously copy-pasted at five call sites across server.py + raptor_oracle.py.)"""
    if "travel_time" in df.columns:
        return "travel_time"
    return [c for c in df.columns if c.startswith("travel_time")][0]


def egress_from_ttm(ttm, to_gid, dtype=np.int64):
    """One-origin W->stops WALK matrix -> ``(egress_g, egress_w)``.

    ``to_gid`` maps a ``to_id`` string to a RAPTOR stop gid, or None to skip the row (the
    oracle bridges R5 stop indices -> gids; the server's stop ids are "S<gid>" directly).
    Several R5 stops can bridge to one gid, so the MIN seconds per gid wins. Returns the
    gids sorted ascending (int32) and their walk seconds (``dtype`` — int64 for the live
    engine, int32 in the baked oracles), minutes rounded to whole seconds."""
    import pandas as pd
    col = tt_col(ttm)
    egr = {}
    for to, v in zip(ttm["to_id"].astype(str), ttm[col]):
        if pd.isna(v):
            continue
        g = to_gid(to)
        if g is None:
            continue
        sec = int(round(float(v) * 60))
        if g not in egr or sec < egr[g]:
            egr[g] = sec
    egress_g = np.array(sorted(egr), dtype=np.int32)
    egress_w = np.array([egr[g] for g in egress_g], dtype=dtype)
    return egress_g, egress_w


def purewalk_from_ttm(ttm, cell_pos, n_cells, dtype=np.int64):
    """One-origin W->cells WALK matrix -> per-cell walk seconds in engine cell order
    (``cell_pos``: cell_id -> index; unknown ids skipped), -1 = unreachable within the cap.
    Minutes rounded to whole seconds; ``dtype`` as in :func:`egress_from_ttm`."""
    import pandas as pd
    col = tt_col(ttm)
    pw = np.full(int(n_cells), -1, dtype=dtype)
    for to, v in zip(ttm["to_id"].astype(str), ttm[col]):
        if pd.isna(v):
            continue
        i = cell_pos.get(to)
        if i is not None:
            pw[i] = int(round(float(v) * 60))
    return pw
