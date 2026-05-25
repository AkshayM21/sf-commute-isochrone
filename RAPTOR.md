# RAPTOR engine — reverse range-RAPTOR grid travel-times (Phase 1)

A self-built reverse **range-RAPTOR** router that replaces the expensive per-cell R5 pass for
the map's travel-time coloring. It computes door-to-door times from every grid cell to a
workplace **in one reverse search + a precomputed walk-access table**, near-exact vs R5 and
~200–700× faster, with no heavy per-visitor compute. R5 stays in-process only for the on-demand
hover breakdown (Phase 2 moves that to RAPTOR too).

Flag-gated: `USE_RAPTOR=1` (default **OFF** — the live app is byte-identical until flipped).

## How it works
1. **Build** (`core/raptor_build.py`, JVM-free, ~2 s, disk-cached by feed fingerprint): parse
   the 3 GTFS feeds for the model date into flat CSR arrays (patterns, FIFO-split for
   overtaking, synthesized 250 m footpaths across feeds, after-midnight trips excluded by the
   morning band).
2. **Reverse range-RAPTOR** (`core/raptor.py` + `core/raptor_numba.py`): one arrive-by search
   rooted at the workplace, swept over a grid of arrival deadlines, gives the *latest departure*
   from every transit stop that still reaches work on time. The hot sweep is an nogil numba
   kernel (byte-equivalent to the pure-python reference, **120× faster**: 1.66 s → 14 ms).
3. **Assemble** per cell = `min( pure_walk, min over access stops [ access_walk + stop_time_to_work ] )`,
   using the **baked cell→stop walk-access table** (`scripts/raptor_oracle.py`, the one-time R5
   walk-matrix precompute; the grid is fixed). Two semantics off the same profile:
   - **depart-after** p5/p50 over [08:35, 09:05] — bit-comparable to R5's window model (validated);
   - **arrive-by-09:00** — the product semantic ("where can you live and reach work by 9").

Per request the server computes the workplace's egress (W→stops) + pure-walk (W→cells) via **one
light R5 walk matrix**, then runs the engine. The only R5 use on the map path; no per-cell pass.

### Hard-won modeling facts (preserved from the spike)
- **60 s board slack** (reach a stop ≥60 s before departure) — without it the router runs ~2.7 min
  uniformly too fast.
- **Deadline step 180 s** matches R5's minute-quantized model; finer adds a fast bias (DS=60 →
  MAE 1.04, bias −0.92 vs DS=180 → MAE 0.75, bias +0.04).
- **Access cap 25 min** cleared the access-starved periphery (max err 15→7, mismatches 20→5 vs a
  20-min cap). 250 m footpaths beat 400 m (400 m adds transfers R5 doesn't take → fast bias).
- **Wider trip band** (to 10:20 = window + routing cap) fixed most periphery reachability misses.

## Accuracy vs R5 (5 diverse workplaces, full 2999-cell grid, depart-after p50)

| workplace  |   n  | MAE | p95 | max | bias  | true-mism | within-2min |
|------------|-----:|----:|----:|----:|------:|----------:|------------:|
| downtown   | 2996 | 0.75| 3.0 |  6  | −0.32 |     0     |    95%      |
| sunset     | 2885 | 0.76| 2.0 |  7  | +0.26 |     0     |    98%      |
| bayview    | 2753 | 0.97| 2.0 |  7  | +0.61 |     5     |    96%      |
| westportal | 2997 | 0.57| 2.0 |  6  | +0.11 |     0     |    99%      |
| caltrain   | 2982 | 0.71| 3.0 |  6  | −0.41 |     0     |    94%      |
| **AGGREGATE** | **14613** | **0.75** | **2.0** | **7** | **+0.04** | **5** | **96%** |

Ground truth = R5's exact forward per-cell pass (R5 has no native arrive-by, so the headline is
its depart-after window p50; the engine reproduces it by inverting the reverse profile, which
transitively validates the arrive-by read-off). Near-75-min-cap reachability flips (R5 72–74 vs
unreachable) are excluded as R5-internal minutiae — there are 52 of those.

**Residual (plateau):** max err 7 and the only 5 true mismatches are all in transit-sparse
**bayview SE** at R5 66–70 min (right at the 75-min routing cap) — borderline reachability where
a few-minute modeling nuance flips the cell. Chasing them is diminishing returns (the user
relaxed the target to "within ~1 min of R5"); the headline beats today's fast map decisively
(MAE 1.49, max 7, *wrong direction*).

## Speed & footprint
- Reverse sweep **14 ms**; profile inversion 11 ms; assembly **117 ms** (numba) — full grid.
- **depart-after 137 ms, arrive-by 26–44 ms** per request, single core (numba, nogil).
- `/compute` end-to-end (incl. the one R5 walk matrix) **~140 ms** for the full grid.
- nogil kernels → concurrent requests scale across cores (no per-request all-core hogging).
- Resident ~15 MB (structs + access CSR); COW-shareable across fork workers → flat memory under
  load. Suits a small free box with many concurrent users.

vs the old path: R5 exact per-cell pass was ~30 s (11 cores) / 90–150 s (4 cores), one heavy job
per visitor. RAPTOR removes it; `map == refine` (both the engine), so the old fast-vs-exact
~7-min contradiction is gone (map vs R5 hover now agree within ~1–2 min, imperceptible).

## Run it
```bash
# 1) one-time: bake the access table + R5 oracles (the only JVM step; rerun on a GTFS repull)
R5_MAX_MEMORY=4G EXACT_THREADS=6 .venv/bin/python scripts/raptor_oracle.py
# 2) validate the engine vs R5 (JVM-free)
.venv/bin/python scripts/raptor_validate.py
# 3) run the server with RAPTOR on
USE_RAPTOR=1 RAPTOR_SEMANTIC=departafter .venv/bin/python scripts/server.py
```
Knobs (env): `USE_RAPTOR`, `RAPTOR_SEMANTIC` (departafter|arriveby), `RAPTOR_ACCESS_CAP` (25),
`RAPTOR_DEADLINE_STEP` (180), `RAPTOR_BOARD_SLACK` (60), `RAPTOR_FOOTPATH_M` (250).
Tests: `pytest tests/test_raptor.py` (JVM-free).

## Phase 2 (next): journey reconstruction → drop the JVM
Design in `prototypes/spike_raptor/PHASE2_DESIGN.md`. Add back-pointers to the reverse search to
reconstruct the leg breakdown + color-by-line dominant route from RAPTOR itself (built on the
arrive-by single-deadline tree so hover == map by construction), replacing the R5 `/itinerary`
and `/attribution`, after which r5py is dev/test-only and the runtime is JVM-free.
