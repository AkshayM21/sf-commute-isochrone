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

## Two semantics (RAPTOR_SEMANTIC), and what each ships
- **`departafter`** — map = depart-after p5/p50 (the R5-validated MAE 0.75 above); breakdown +
  color-by-line stay on R5 recorded paths. **This is the validated, R5-consistent Phase-1 ship.**
- **`arriveby`** (default, per review) — map = the *actual commute* of the latest-feasible journey
  arriving by 09:00, AND the breakdown + color-by-line come from the same RAPTOR traced tree
  (Phase 2). Internally consistent (hover == map exact) and deterministic, but see the route
  caveat below.

## Phase 2: journey reconstruction from RAPTOR — status
Design: `prototypes/spike_raptor/PHASE2_DESIGN.md`. Implemented: `raptor.reverse_raptor_traced`
(back-pointers), `raptor_journey.JourneyTree` (leg breakdown + dominant line from the GTFS times
we load), `raptor_build` v2 (feed-aware names: Muni 8 vs BART Red-N). Wired into `/itinerary` +
`/attribution` behind `arriveby`.

**Works:** hover == map EXACTLY (0/2997 violations — the breakdown legs sum to the cell's map
minutes by construction), feed-aware names, color-by-line in ~5 ms and **deterministic across
reboots** (R5's flipped ~1057 cells per boot). No R5 recorded-path compute, no `_HEAVY_LOCK`/fan.

**Open wall (measured):** RAPTOR's reconstructed routes match R5's dominant line only **~46%**
(59% corridor-collapsed; **18% even at the same departure minute**). Root cause: (1) the reverse
**latest-departure** objective produces valid-but-different journeys than R5's **earliest-arrival**;
(2) the arrive-by anchor is optimistic (commute ~3–6 min under the depart-after p50); (3) the
Market-St Metro tunnel (K/L/M/N/J/T) and the BART downtown trunk (Yellow/Red/Blue/Green) are
**interchangeable** — R5's own dominant-line pick there is per-boot arbitrary, so exact match to it
is ill-defined. So this is not a clean R5-match; closing to ≥90% needs a **forward earliest-arrival
reconstruction** (run forward RAPTOR from each cell's chosen departure to recover R5's actual
fastest journey) — a clear next step. Validate: `scripts/raptor_validate_paths.py` (vs R5 dominant),
`scripts/raptor_check_anchor.py` (same-departure check).

**JVM not yet dropped.** Even in `arriveby`, the runtime still calls R5 once per workplace for the
egress/pure-walk walk matrix (`_raptor_egress_purewalk`). Fully removing r5py needs an R5-free
pedestrian router for W→stops/cells (e.g. snap W to the nearest baked grid cell, or a pandana/OSM
walk graph) — the last ~140 ms R5 dependency. Until then R5 is loaded but does **no** heavy
per-cell pass and **no** recorded-path breakdowns (those are RAPTOR in `arriveby`).

## Phase A: service-noise Monte-Carlo — realistic + fragility + alt-lines (`RAPTOR_MC=1`, default ON)
The arrive-by map is the **perfect-timing best case** (you leave home to the second and catch every
vehicle with ~0 wait). Departure timing is the user's control, so we keep perfect as the BASE and
add a **realistic** number + a **fragility** score from a Monte-Carlo over *service* noise — the
part you can't control. Each draw perturbs the schedule and **re-runs the same validated reverse
sweep**, which **re-optimizes per scenario** (miss the express → the min over patterns lands on the
next local), so the spread captures missed-transfer **re-routing**, not a naive same-line headway.

- **Perturbation** (`raptor.apply_delays` / `raptor_numba._perturb`): per trip, delay
  `δ(pos)=δ₀+slope·scheduled_elapsed(pos)` applied to dep & arr (dwell preserved); `δ₀,slope ~
  Gamma` with means by mode/operator (bus noisiest, rail steadiest; keyed on `pat_mode`+`pat_feed`).
  A per-pattern **FIFO cumulative-max clamp** (no overtaking on the arr OR dep column) keeps the
  per-position binary search valid (tested).
- **Hot path** (`raptor_numba.montecarlo`, `parallel=True` nogil): draws run in `prange`, each
  thread holding ONE perturbed schedule + ONE latest profile, **streaming** each draw's per-cell
  median-departure commute into `commute_all[n_cells, R]` (never R×n_stops). ~**0.1 s** for 24
  draws (full grid). Per-cell outputs: `realistic = p50` (clamped ≥ perfect), `frag = p90−p50`
  (the "bad-day +Y" headline), `std`, `stuck` (fraction of draws hitting the cap = last-train/
  peak risk).
- **Alt-lines** (`RaptorEngine._mc_alt_lines`): the dominant line across ~4 *traced* perturbed
  arrive-by trees (reuses `JourneyTree`), so a delayed express lets the next local show up. Pricier
  (pure-python trace, ~0.7 s for 4 draws), env-tunable (`RAPTOR_MC_ALT_DRAWS`), excludes the cell's
  normal line.
- **Serving:** lazy `/variance` endpoint (`server._raptor_mc`, cached per workplace, deterministic
  per-workplace seed), fetched by the frontend AFTER `/compute` paints the perfect map (progressive
  refinement, like `/compute_exact`). NEVER on the hover path. The hover stays the single
  perfect-timing journey (hover == perfect-map exact); realistic/fragility/alt are header
  annotations. The `Realistic`/`Best-case` `#metric` toggle now switches MC-p50 ↔ perfect.

**Validation** (`scripts/raptor_validate_mc.py`, `tests/test_raptor.py::test_mc_*`): realistic vs
R5's *schedule-perfect* p50 — MAE **2.19**, bias **+1.14** (mildly positive: delays only add time;
larger in fragile peripheral workplaces like bayview +2.4), **0** `perfect ≤ realistic` violations,
FIFO-clamp sortedness. There is no true R5 ground truth for a *delayed* commute, so this bounds
realistic's drift from schedule-perfect rather than measuring error.

> **Caveat (documented limit):** the MC re-routes **clairvoyantly** (it knows the perturbed schedule
> at home), so `realistic` is a **lower bound** on real "committed-plan" variance — it answers "is
> there a good alternative under bad conditions" (route *resilience*), not "I'm on the platform and
> just missed it." A committed-plan MC (fix departure + first leg, re-optimize the tail) is the more
> expensive future upgrade.

Knobs (env): `RAPTOR_MC` (1), `RAPTOR_MC_DRAWS` (24), `RAPTOR_MC_ALT_DRAWS` (4), `RAPTOR_MC_SHAPE`
(2.0), `RAPTOR_MC_MU_{BUS,METRO,CABLE,BART,CALTRAIN}` (70/45/40/25/40 s initial delay mean).
