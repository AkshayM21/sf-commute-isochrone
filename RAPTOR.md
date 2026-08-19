# RAPTOR engine — reverse range-RAPTOR grid travel-times (Phase 1)

A self-built reverse **range-RAPTOR** router that replaces the expensive per-cell R5 pass for
the map's travel-time coloring. It computes door-to-door times from every grid cell to a
workplace **in one reverse search + a precomputed walk-access table**, near-exact vs R5 and
~200–700× faster, with no heavy per-visitor compute. R5 is retained for offline validation bakes,
not normal map or hover handling.

Flag-gated: `USE_RAPTOR=1` is the **DEFAULT**. The server's default `RAPTOR_SEMANTIC=departafter`
serves the **planned scheduled** metric: one first-boarding-anchored journey, not a p5/p50
percentile. `RAPTOR_SEMANTIC=arriveby` remains the legacy perfect-timing alternative.

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
   walk-matrix precompute; the grid is fixed). Three engine modes share these inputs:
   - **planned scheduled depart-after** — the served map: a single scheduled, first-boarding-
     anchored value whose itinerary and map value are derived from the same planned tree.
   - **legacy depart-after p5/p50** — the R5-comparable validation/comparison percentile model.
   - **legacy arrive-by-09:00** — a perfect-timing alternative.

## Default product metric

The server routes `RAPTOR_SEMANTIC=departafter` through the planned scheduled model. The headline
is therefore neither a best-case nor a median percentile: it is the selected scheduled branch.
`/itinerary`, color-by-line, and the map use the same planned tree, so the displayed route totals
match the painted value. The legacy percentile model remains available through the engine API for
R5 comparison, but it is not the served map metric. All product requests are JVM-free when the
walk graph is enabled.

`RAPTOR_SEMANTIC=arriveby` is retained for the perfect-timing arrive-by alternative. It is a
different metric and should not be described as the default or compared directly to the planned
headline as though it were a percentile.

Per request the server computes workplace egress (W→stops) + pure-walk (W→cells) with the
hill-aware walk graph, then runs the engine. R5 is not on the served map path.

### Hard-won modeling facts (preserved from the spike)
- **60 s board slack** (reach a stop ≥60 s before departure) — without it the router runs ~2.7 min
  uniformly too fast.
- **Deadline step 180 s** matches R5's minute-quantized model; finer adds a fast bias (DS=60 →
  MAE 1.04, bias −0.92 vs DS=180 → MAE 0.75, bias +0.04).
- **Access cap 25 min** cleared the access-starved periphery (max err 15→7, mismatches 20→5 vs a
  20-min cap). 250 m footpaths beat 400 m (400 m adds transfers R5 doesn't take → fast bias).
- **Wider trip band** (to 10:20 = window + routing cap) fixed most periphery reachability misses.
- **SNAPSHOT footpath relax** (2026-06-10, output-changing): every footpath relaxation pass (the
  egress-seed pass + each round's pass) computes candidates from the SOURCE stops' values FROZEN
  at the start of the pass, never from live `best[]`. That makes the relax order-independent, so
  the three lockstep siblings (`raptor.reverse_raptor` / `reverse_raptor_traced` /
  `raptor_numba._profile`) are now **BYTE-EQUAL** despite their different scan orders (the old
  live-read let multi-hop walk cascades fire per scan order — measured on the ferry workplace:
  **7.8% of reachable stops** diverged, the live-read better by up to ~1183 s and the snapshot
  better by up to ~263 s; cell-level ~50/2999 served cells shifted −2..+6 min, 0 reachability
  flips). numba==python is now exact end-to-end (`test_lockstep_byte_equal`,
  `test_numba_matches_python` ==0). The python `stop_arrival_profile` fallback mirrors the
  kernel's explicit binary search (byte-parity on any input).
- **MONOTONE profile rows** (2026-06-10, output-changing): a stop's latest-departure row must be
  non-decreasing in the deadline (any journey arriving by T also arrives by T' > T), but the
  marked-stop pruning + one-hop footpath policy left ~58 non-monotone rows (worst violation
  958 s, ferry workplace) that the profile inversion + the MC tail readout binary-searched as if
  sorted. The profile producers now finish with a **row-wise running max** over the deadline
  axis (in-kernel in `raptor_numba._profile`, `np.maximum.accumulate` in the python
  `reverse_profile` fallback) — provably valid (it restores only feasible journeys the pruning
  dropped) and it makes every downstream binary search well-defined. Output effect at minute
  resolution: **none measured** — 0 cells change on depart-after across the 5 oracles + ferry,
  0 on the arrive-by traced tree (served map + golden), 0 on the MC overlay; the 5-oracle
  aggregate (MAE 0.7502) is unchanged. The value is defensive (the restored entries are real
  but sub-minute on current data). Guard: `test_profile_rows_monotone`; lockstep compares the
  cummaxed reference columns.
- **RIDE-DEPTH TRACE NODES — max-transfers phantom-ride fix** (2026-06-15, trace-only, output-changing
  on the served arrive-by map). The per-stop back-pointer arrays (`par_*`) carry only each stop's
  LATEST parent, but `par_nxfer` is the RAPTOR round index, **not** the path's ride depth: a round-k
  footpath-after-board pass can relax a footpath OUT OF a stop freshly BOARDED that round, so a
  stop's *final* parent sits one ride deeper than the value a later board CONSUMED from it (boards
  read `prev[]`, captured BEFORE the same round's footpath pass overwrote it). Walking the per-stop
  chain then appended a **phantom ride** → a trace could draw MORE transit legs than `max_rounds`
  (e.g. a "0 transfers" cell drawing a 2-leg PM→CA cable transfer; 5–84 cells at `max_rounds=1`,
  11 at 2, depending on workplace) AND, even at the default `max_rounds=8`, the wrong (longer) trace
  inflated the served arrive-by map value on ~1 cell/oracle (the map = the traced journey's actual
  time via `commute_and_dominant`/`_clock`). `reverse_raptor_traced` now also builds an **immutable
  ride-depth NODE TABLE**: each relaxation captures the EXACT continuation its value consumed (a
  board captures the alight stop's `prev`-node — what the scan read; a footpath captures the source's
  SNAPSHOT node — what the frozen relax read), and each node carries its true ride depth (egress=0,
  footpath inherits, board = consumed+1). Since a board fires only in rounds 1..max_rounds and adds
  exactly one to a depth ≤ round−1, every node's depth ≤ max_rounds, so `raptor_journey._trace_uncached`
  (now walking `best_node[]`→`nd_next`) can never exceed the cap. **`best`/the validated profile +
  lockstep trio are UNTOUCHED** (the node table is built ALONGSIDE the per-stop arrays). Effect: the
  drawn legs + `xfers` honor the cap; the served arrive-by map drops the phantom-inflated cells to
  their true journey time (golden re-stamped: ~194 cells −5..−6 min, 0 reachability flips). Guard:
  `test_traced_journey_respects_max_rounds` (≤ max_rounds transit legs at max_rounds∈{1,2}, hover==map
  preserved). The MC alt-line traces go through the same path → fixed by the same change.
- **MILD WALK-RELUCTANCE PRIOR — "transit slightly preferred over walking"** (2026-06-15, decision-only,
  output-changing on the served arrive-by map; `RAPTOR_WALK_RELUCTANCE` default **1.15**, `RAPTOR_WALK_PRIOR_EPS`
  default **60 s**, both env-overridable; `=1.0` is an EXACT no-op). The objective is pure
  earliest-arrival, so among options that reach work at ~the same time it had **no preference for less
  walking** and the access argmin would send a fast walker to a geometrically FARTHER station at the
  same displayed minute (the user's exact complaint). The prior weights walk seconds by `beta` in the
  access-stop **DECISION only**; the reported door-to-door minutes stay **TRUE clock time** (the chosen
  journey's actual time, never an inflated number). Two safeguards keep it a tie-break, never a
  multi-minute degradation: (1) an **EPS BAND** — the prior re-selects only among access stops whose
  TRUE door-to-door time is within ±`eps` (60 s) of the time-optimal (max-latest-departure) winner, so
  the map value moves by at most the rounding minute; a genuinely-faster farther stop (> eps better)
  is never traded away. The band compares the TRACED journey time (`jtime[stop] + access_walk` from the
  node table), NOT the home departure — the arrive-by latest-departure can ARRIVE before the deadline,
  so two stops with near-equal home departures can have very different door-to-door times (a 22 s
  home-dep tie hid a 4-min slower journey before this fix). (2) a **TRANSFER GUARD** — candidates are
  capped to displayed ride depth ≤ the time-optimal stop's depth (the user's "not a million transfers"),
  using the corrected node-table ride depth (significant rides only, matching `_clock`'s tiny-hop fold),
  so the prior can never add a transfer. The walk-vs-transit decision stays on the ANCHOR's true home
  (the prior only re-selects among transit access stops, never flips a cell walk↔transit). Threaded
  through `assemble_departafter[_arith]` + `assemble_arriveby`/`_assemble_arriveby_window` (the fast
  kernels, used by the depart-after R5-validation + the `arriveby()` method) AND
  `JourneyTree._select_arrays` (the served map + hover + color-by-line + committed-MC first legs).
  Kept OUT of `reverse_raptor*`/`_profile` (lockstep + cummax + MC `_draw_profile` byte-equality
  intact). **R5 parity holds**: depart-after p50 MAE 0.7502→0.7508 (+0.0006), bias +0.064→+0.064, max 7,
  true-mism 5 — within noise (R5 has no walk-reluctance field, so this is purely the tie-break;
  reporting true time means it only re-rounds the rare sub-minute near-tie). Served arrive-by effect
  over 5 workplaces: ~1300 cells save walking (≈9.5 k walk-min), **590 SHED a transfer, 0 gain one**,
  reported time ≤+1 min on 64 cells (eps rounding). Golden re-stamped (404/2999 cells: Fix-1 −5/−6
  corrections + Fix-2 −1 closer-stop + +1 eps rounding; 0 reachability flips). Guards:
  `test_walk_prior_reduces_walk_without_adding_transfers`, `test_walk_prior_is_noop_at_one`,
  `test_walk_prior_mae_within_budget`.
- **MIN-JOURNEY SELECTION + NO-OVERSHOOT ALIGHT — "the shown route is always the fastest"**
  (2026-06-16, reconstruction-only, output-changing on the served arrive-by map; ALL in
  `core/raptor_journey.py` + a 1-line constructor plumb in `raptor_engine.journey_tree`). The
  arrive-by reverse tree optimizes LATEST HOME DEPARTURE, not the shortest trip — a latest-departure
  journey can ARRIVE well before the deadline (it just couldn't have left later), so two stops with
  near-equal latest departures can have very different door-to-door durations. Two visible bugs
  followed: (1) `JourneyTree._select_arrays` anchored the access-stop pick on max-latest-home
  (`base_key = best[stop] - aw`), so on ~1774 cells it showed a SLOWER route than an available faster
  line (which surfaced only as an "alt"; the primary could even be absent from its own "also serves").
  (2) `_trace_from` followed the latest-departure node chain's `nd_alight`, which could ride PAST the
  stop closest to W and walk back (a longer ride AND longer egress walk — strictly dominated; proven:
  the 22 bus 198 Dolores→650 Townsend rode to −122.39991 at faster walk instead of −122.40286, 13 min
  ride + 11 min walk vs 11 + 8). **CHANGE 1:** the `_select_arrays` anchor is now the segmented ARGMIN
  of `cell_jt = jtime[stop] + access_walk` (the SAME true-time quantity `alt_lines_window` ranks by, so
  primary == fastest alt), with the walk prior re-applied as a tie-break (eps band one-sided off the
  min + transfer guard, min `cell_jt + (beta-1)·aw` then least walk then first index). `latest` stays
  `best[chosen]-aw` so reported clock time is exact; walk-vs-transit now compares DURATIONS (pure walk
  wins iff its sec ≤ the min-journey transit sec, walk wins ties). **`beta==1.0` is no longer the old
  byte-equal max-latest-home anchor — it is "min-journey, first index"** (the objective changed; the
  golden + `test_select_matches_reference_loop` were re-stamped). **CHANGE 2:** for the FINAL ride (the
  board whose continuation is the egress seed), `_trace_from` + `_build_node_stats` keep the line/trip
  + board (preserving CHANGE 1's selection + the reported departure + the committed-MC first leg) but
  re-pick the alight to MINIMIZE `arr[p] + egress_walk(stop@p)` over forward egress-reachable positions
  (shared `_min_overshoot_alight`); intermediate rides keep their alight (the transfer stop is fixed —
  changing it would alter the transfer sequence). Per-stop egress seconds are threaded into the
  constructor (`egress_g, egress_w` from `journey_tree`, gid→sec, `EGRESS_INF` sentinel for
  unreachable; legacy callers without them fall back to the node's `nd_egress`, behavior unchanged).
  `jtime` is computed off the SAME no-overshoot arrival so the anchor, the alt window, and the traced
  journey agree — **hover==map preserved** (`commute_and_dominant` traces via the same path, so the map
  value automatically drops to the optimized time). **Kept OUT of `reverse_raptor*`/`_profile`/the
  numba reverse kernel + the depart-after `assemble_*` path** — R5 parity is UNMOVED (depart-after p50
  MAE 0.75 aggregate, bias +0.06, max 7, mism 5 — byte-identical to baseline; this is arrive-by
  reconstruction only). Golden re-stamped (284/2999 cells: **283 faster −1..−6 min** + 1 eps-rounding
  +1; 0 reachability flips). Faster-walk→longer-commute non-monotonicity on the served map nearly
  halved over 5 workplaces (922→496 cells, 6.35%→3.41%; max magnitude 10→8 min); the residual is the
  structural single-journey-per-access-stop limit + the arrive-by latest-departure re-pick genuinely
  opening a different corridor at a different walk speed (a real routing effect, not the bug). Full-grid
  trace ~50 ms (no perf regression). Guards: the re-stamped `test_select_matches_reference_loop`
  (min-journey reference loop) + the unchanged `test_hover_equals_map_invariant`,
  `test_traced_journey_respects_max_rounds`, `test_walk_prior_*`.

## Accuracy vs R5 (5 diverse workplaces, full 2999-cell grid, depart-after p50)
**Snapshot-relax baseline (re-stamped 2026-06-10):**

| workplace  |   n  | MAE | p95 | max | bias  | true-mism | within-2min |
|------------|-----:|----:|----:|----:|------:|----------:|------------:|
| downtown   | 2996 | 0.74| 3.0 |  5  | −0.30 |     0     |    95%      |
| sunset     | 2885 | 0.76| 2.0 |  7  | +0.28 |     0     |    98%      |
| bayview    | 2753 | 0.98| 2.0 |  7  | +0.62 |     5     |    96%      |
| westportal | 2997 | 0.59| 2.0 |  6  | +0.18 |     0     |    99%      |
| caltrain   | 2982 | 0.70| 3.0 |  6  | −0.40 |     0     |    94%      |
| **AGGREGATE** | **14613** | **0.75** | **2.0** | **7** | **+0.06** | **5** | **97%** |

Ground truth = R5's exact forward per-cell pass (R5 has no native arrive-by, so the headline is
its depart-after window p50; the engine reproduces it by inverting the reverse profile, which
transitively validates the arrive-by read-off). Near-75-min-cap reachability flips (R5 72–74 vs
unreachable) are excluded as R5-internal minutiae — there are 52 of those.

**Residual (plateau):** max err 7 and the only 5 true mismatches are all in transit-sparse
**bayview SE** at R5 66–70 min (right at the 75-min routing cap) — borderline reachability where
a few-minute modeling nuance flips the cell. Chasing them is diminishing returns (the user
relaxed the target to "within ~1 min of R5"); the headline beats today's fast map decisively
(MAE 1.49, max 7, *wrong direction*).

## Speed & footprint (re-measured 2026-06-10, post arith-assembly + snapshot relax + monotone cummax)
- Reverse sweep **14 ms**; profile inversion ~11 ms; assembly now arithmetic-indexed (~7×).
- **depart-after ~41 ms, arrive-by ~35 ms** per request, full grid, single core (numba, nogil).
- `/compute` end-to-end **~105–130 ms** for a FRESH workplace (incl. the JVM-free walk-router
  Dijkstra); repeat for the same workplace+params is served from cache (~0 ms).
- MC committed kernel (24 draws, full grid, `prange`) **~120 ms**.
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
`RAPTOR_DEADLINE_STEP` (180), `RAPTOR_BOARD_SLACK` (60), `RAPTOR_FOOTPATH_M` (250),
`RAPTOR_WALK_RELUCTANCE` (1.15) + `RAPTOR_WALK_PRIOR_EPS` (60) — the mild walk prior (see Hard-won facts).
Tests: `pytest tests/test_raptor.py` (JVM-free).

## Server semantics (`RAPTOR_SEMANTIC`)

- **`departafter`** (default) — serves the planned scheduled map and its matching planned tree.
  The separately callable legacy p5/p50 model is for validation/comparison, not this map.
- **`arriveby`** — opt-in perfect-timing arrive-by routing, with a matching traced tree.

## Phase 2: journey reconstruction from RAPTOR — status
Design: `prototypes/spike_raptor/PHASE2_DESIGN.md`. Implemented: `raptor.reverse_raptor_traced`
(back-pointers), `raptor_journey.JourneyTree` (leg breakdown + dominant line from the GTFS times
we load), `raptor_build` v2 (feed-aware names: Muni 8 vs BART Red-N). Wired into `/itinerary` +
`/attribution` behind `arriveby`.

**Works:** hover == map EXACTLY (0/2997 violations — the breakdown legs sum to the cell's map
minutes by construction), feed-aware names, color-by-line in ~5 ms and **deterministic across
reboots** (R5's flipped ~1057 cells per boot). No R5 recorded-path compute, no `_HEAVY_LOCK`/fan.

**Hover route GEOMETRY (2026-06-10):** `/itinerary` now also returns `geom` — an ordered leg
list aligned **1:1 with the breakdown legs** (same trace, same `_clock` folding), each
`{mode, name, feed, tmode, min, wait?, pts:[[lat,lon],…], approx?}`. Ride legs = the pattern's
stop-coordinate sequence board→alight; walk legs = REAL street paths from `core/walk.PathTree`
(Dijkstra `return_predecessors=True` predecessor chains, direction-aware: egress/pure-walk walk
the W-rooted TRANSPOSED tree, access a cell-rooted forward tree; straight 2-pt `approx:true`
fallback without the graph). The W-rooted tree is cached per workplace (`_WALKPATH_TREE_CACHE`),
assembled responses per cell inside the tree-cache entry (`entry["geom"]`, ~45 ms cold / ~0 ms
warm). Frontend draws it on hover (white-cased polylines, line-label pills, total badge); click
PINS it (popup lifecycle = pin lifecycle; hover-draw suppressed while pinned). Legacy R5 path
serves `geom:null`. Tests: `test_api.py::test_itinerary_geom_*`.

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

**JVM drop:** done in **Phase B** (below) via `USE_WALK_GRAPH=1` — the per-workplace
egress/pure-walk walk matrix (`_raptor_egress_purewalk`) is replaced by a JVM-free hill-aware
pedestrian router, and the boot skips the R5 network entirely. With `USE_RAPTOR=1 USE_WALK_GRAPH=1`
+ arrive-by, **r5py is never imported** (verified: 0 libjvm handles in-process, RSS ~333 MB vs
~1.6 GB+ with the JVM).

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
- **`realistic` = COMMITTED-PLAN** (`raptor_numba.montecarlo_committed`, `parallel=True` nogil): you
  commit your departure + first leg from the *published* plan (`JourneyTree.committed_first_legs`
  extracts, per cell, the committed home departure + first board pattern/position/alight from the
  unperturbed arrive-by tree — no foreknowledge of delays). Per draw it boards the **next available
  trip on that committed line** (a late earlier train you can also catch), rides to the committed
  alight, then re-optimizes the **tail** from the *actual* (late) arrival via the perturbed reverse
  profile. So a late first leg that blows a transfer **eats a real headway**. It tracks R5's
  depart-window p50 closely (agg **41.3**, bias ≈ **0** over 5 workplaces — committed = best
  departure + small service delay, which is what R5's window median already measures), with a modest
  fragility tail (`frag90` 3–8 min; SF transit is frequent). `perfect ≤ committed` holds (clamped +
  asserted). **GOTCHA:** the boarding key is `commit_home + walk0` with NO extra board_slack —
  `commit_home` already places you at the stop exactly at your committed trip's departure (the
  perfect-timing arrive-by assumption), so adding slack would skip your own trip to the next one and
  eat a full headway on every cell (the board-slack bug, see Issues 2026-05-25).
- **Hot path:** draws run in `prange`, each thread holding ONE perturbed schedule + ONE latest
  profile (`_draw_profile`), **streaming** each draw's per-cell commute into `commute_all[n_cells, R]`
  (never R×n_stops). ~**0.1–0.2 s** for 24 draws (full grid). Per-cell outputs: `realistic = p50`
  (clamped ≥ perfect), `frag = p90−p50` (the "bad-day +Y" headline), `std`, `stuck` (fraction of
  draws hitting the cap = last-train/peak risk).
- **Alt-lines = a door-to-door DOMINANCE WINDOW** (`RaptorEngine._alt_window` +
  `JourneyTree.alt_lines_window`, 2026-06-16, replaces the old K-draw vote): per cell, every distinct
  transit line whose **best per-access-stop door-to-door time is within `ALT_WINDOW_MIN` (default 5,
  env `RAPTOR_ALT_WINDOW_MIN`) of the cell's best** is an alternative ("also serves: 38, 5R"), sorted
  closest-first. Computed off the UNPERTURBED arrive-by tree: each access stop offers exactly one
  journey (its latest-departure node chain → `jtime[stop] + access_walk` + a per-stop dominant line
  from one bottom-up node-table pass, `_build_stop_dominant`), so the window is **deterministic,
  K-free, and walk-speed-STABLE** — a within-window bus stays listed when you walk faster (its gap to
  best changes only by the walk-speed delta on the access leg). This fixed the user's complaint that
  short-walk buses VANISHED on speed-up: the old "dominant in ≥1 of K=4 random perturbed draws minus
  primary" was a noisy lottery that dropped near-best buses on ~29% of cells (862/2984) when walking
  sped up because they stopped winning ANY of 4 dice rolls (`.plans/alt_walkspeed_diag.md`). A
  perturbation-draw candidate pool was *measured to ADD churn*, so alts come purely from the
  deterministic tree; the realistic/fragility MC keeps its own `RAPTOR_MC_DRAWS`. `RAPTOR_MC_ALT_DRAWS`
  now only GATES alts on/off (>0). Retention on speed-up: **downtown 80%, aggregate 63%** of
  within-window slow alts (guard `test_mc_alt_window_walk_speed_stable`); the residual is the
  structural single-journey-per-access-stop limit + the arrive-by latest-departure re-pick on the
  SE/periphery where faster walking genuinely opens a better corridor (a real, correct drop, not the
  display bug). The server still drops the cell's PRIMARY line + caps at 4 closest.
- **Drawn alt routes** (the "draw the alt-line route on hover" feature): `montecarlo()` returns an
  `alt_bundle` = `{"alt_stop": list[{line: access_stop}|None], "draws": []}` (a tiny per-cell map, no
  perturbed schedules). The server stashes the bundle + the per-cell chip set in the `/variance` MC
  cache entry (same coarse key = workplace+rides+**speed**), and `/itinerary` returns
  `alts: [{line, min, legs:[…geom legs…]}]` — for each chip line the journey traced from that line's
  windowed access stop on the SAME (cached) primary tree via `JourneyTree.itinerary_via_stop`, then
  assembled through the SAME `_JourneyGeomProvider` the primary route uses (real street walk legs;
  provider shared with the primary geom so the per-cell access Dijkstra isn't redone). No perturbed
  re-trace, no separate JourneyTrees: the tree is the one `/compute` already built. Lazy + cached: the
  assembled `alts` cache per cell (`alt_geom`, warm hover ≈0 ms); `/itinerary` **never** triggers the
  MC build, so `alts: []` until `/variance` lands (then the frontend re-hovers). `server._itinerary_alts`
  + `_mc_peek`. The alt geom legs sum to the alt total and honor `max_rounds` (the node-chain trace,
  same as the primary).
- **Per-route TYPICAL for the compare card** (2026-06-16): each alt also gets its OWN committed-plan
  typical (`real`) + fragility (`frag`), scored by the SAME committed MC as the primary so the pinned
  compare list can show every strip on the metric the user selected (the old bug: primary on Typical
  vs alts on Best-case made an alt look faster). `RaptorEngine.route_typicals(tree, ci, stops, …)`
  builds, per route, the committed first leg from the journey traced via that route's access stop
  (`JourneyTree.committed_legs_via_stops` → the SAME `_fill_committed_leg` rule as
  `committed_first_legs`), then runs **one** `montecarlo_commute_committed` over the combined
  primary+alts batch FOR ONE CELL — the R per-draw reverse profiles (the dominant, cell-independent
  cost) are computed once and shared across all routes, so a 3-4-route pin resolves in ~130-190 ms.
  `perfect ≤ committed` is floored PER route (each at its own best-case). Served on `/itinerary?pin=1`
  ONLY (never a plain hover), cached per pinned cell in the MC entry's `typ` dict under `_TYP_LOCK`
  (`server._itinerary_alt_typicals`); the seed mirrors `/variance`'s per-workplace seed so the PRIMARY
  route's typical is byte-identical to that cell's served `realistic` (the primary strip == the
  headline). Frontend (`compareHTML`/`optRead`/`drawSelected`): every strip's number + badge follow
  the selector, the legs+"typical wait" reconciliation is per strip, and the list is sorted by the
  displayed metric. Guards: `test_api.py::test_itinerary_pin_per_route_typicals`,
  `test_route_quality.py::test_per_route_typical_honors_perfect_le_committed`.
- **Serving:** lazy `/variance` endpoint (`server._raptor_mc`, cached per workplace, deterministic
  per-workplace seed, reuses the cached arrive-by tree → no re-trace), fetched by the frontend AFTER
  `/compute` paints the perfect map (progressive refinement, like `/compute_exact`). NEVER on the
  hover path. The hover stays the single perfect-timing journey (hover == perfect-map exact);
  realistic/fragility/alt are header annotations. The `Realistic`/`Best-case` `#metric` toggle
  switches MC-p50 ↔ perfect.

- **Depart-after metric contract (Stage 3 of the depart-after map migration, 2026-06-17).** Under
  depart-after **best-case and typical are DIFFERENT percentiles of the departure window, so each is a
  DIFFERENT drawn journey** (can be a different route). The contract reflects that — it does NOT make
  "Typical" the MC committed number:
  - **Map color.** best-case = p5 (`cells[c][0]`), typical = p50 (`cells[c][1]`), served by `/compute`.
    The MC never overrides the typical color (the depart-after `/variance` carries NO `realistic`).
  - **`/itinerary` returns BOTH journeys** so the frontend switches on the metric toggle WITHOUT a
    re-fetch. `server._raptor_tree` caches BOTH a p50 `DepartAfterJourneyTree` (`tree`) and a p5 tree
    (`tree5`) per workplace (~50 ms each; the per-T* reverse-traced trees inside are lazy). Each
    journey's total is the cell's painted percentile EXACTLY (hover==map for BOTH): the p50 journey
    (`tree.itinerary(ci)`) total == `cells[c][1]`, the p5 journey (`tree5.itinerary(ci)`) total ==
    `cells[c][0]`. Response shape: root + `typical` = the p50 journey, `best` = the p5 journey (each
    `{total, xfers, legs, geom}`); legs reconcile to each journey's own total.
  - **`/variance` (depart-after) = `{frag, stuck, alt}` ONLY, NO `realistic`.** The typical headline is
    the bare p50 the map paints; the MC is used SOLELY for the p90 tail (→ `frag = max(0, p90 − p50)`,
    with p90 the committed-MC p90 and p50 the served depart-after p50 = the floor `_raptor_mc_build`
    passes to `montecarlo`, so frag ≥ 0 by construction) + the alt-line set. `arrive-by` is
    BYTE-UNCHANGED — it still serves `{realistic, variance}` (its perfect-timing base is too rosy, so
    the committed `realistic` IS its typical headline; depart-after's p50 already is the typical).
    - **frag derivation (the metric-contract fix, 2026-06-17):** the chip reconciles as
      `displayed_headline + frag == committed_p90`. Arrive-by's headline IS the committed p50, so its
      frag is the engine's `mc["frag"] = round(p90 − p50)` (kept byte-identical). Depart-after's
      headline is the BARE served p50 (the painted floor `cells[c][1]`), which is `< committed_p50` on
      ~77% of cells (committed p50 drifts above the floor), so `mc["frag"]` would understate the bad
      day by `committed_p50 − served_p50` (mean +2.2, max +15). So `montecarlo`/`route_typicals` now
      ALSO return `committed_p90` (= `round(p90)` of the floored draws; `route_typicals` only with
      `return_committed_p90=True`), and the depart-after callers derive `frag = max(0, committed_p90 −
      served_p50)` → `served_p50 + frag == committed_p90` EXACTLY. NOTE: you can NOT reuse the absolute
      for arrive-by (`realistic + frag != committed_p90` on ~half the cells: `realistic = ceil(p50)` vs
      `frag = round(p90 − p50)` use different rounding) — arrive-by keeps its own `frag` field, which is
      why the two derivations are kept separate, not unified.
  - **Per-route (`/itinerary?pin=1`):** each route (primary + each alt) carries its OWN p5 + p50
    journey + its OWN `frag`. The alt journeys are anchored on the alt's **per-stop percentile**
    (`DepartAfterJourneyTree.itinerary_via_stop(ci, s, percentile=5|50)` →
    `_stop_percentile_anchor`: the percentile of `tt_s(D) = arrivalW[s, D+aw] − D` over the window),
    which guarantees **alt p5 ≤ alt p50 PER alt** (the per-stop percentile is monotone in the
    percentile — the old latest-departure best-case mixed the cell's p5/p50 deadline trees and could
    read an alt's best-case SLOWER than its typical). The primary's p50 journey total == the served
    p50 (`cells[c][1]`). Per-route `frag` comes from `route_typicals` on the p50 tree (primary floored
    at `cells[ci][1]`, each alt at its per-stop p50); each route's frag ≥ 0.

  Reused from the earlier Stage-3 build: the committed-MC surface on `DepartAfterJourneyTree`
  (`committed_first_legs` / `committed_legs_via_stops` / `_select` / `alt_lines_window`), the
  `_raptor_mc_build` p50 floor (which is what makes frag ≥ 0), and the alt enumeration. Changed: the
  served `realistic` is no longer the depart-after typical (dropped from `/variance`); `/itinerary`
  returns both p5 + p50 journeys (was p50-only); each alt journey is anchored on its per-stop
  percentile (was the latest-departure best-case); per-route surfaces p5+p50 totals + frag (was a
  single committed "real"). Guards: `test_route_quality.py::test_departafter_mc_overlay_and_per_route_journeys`
  (engine) + `test_route_quality.py::test_departafter_frag_reconciles_with_headline` (the metric-contract
  regression: `served_p50 + frag == committed_p90` per cell AND per route, 0 violations across the golden
  workplaces, plus a sentinel that the new frag exceeds the old `committed_p90 − committed_p50` where
  committed_p50 > served_p50 — mean +3.2, max +16) + `test_api.py::test_itinerary_equals_map_departafter`
  (the subprocess driver now asserts the both-journey hover==map + the `{frag,stuck,alt}`-only `/variance`
  + per-route p5≤p50 + that the primary pinned strip's frag == the served `/variance` frag). The
  frontend wiring is a SEPARATE follow-up. The served default remains planned `departafter`; the
  arrive-by MC + `/itinerary` + `/variance` paths remain available as the opt-in alternative.

**Validation** (`scripts/raptor_validate_mc.py`, `tests/test_raptor.py::test_mc_*`): committed vs R5's
*schedule-perfect* p50 (NOT ground truth for a delayed commute — committed should sit ABOVE it).
Aggregate over 5 workplaces **41.3**, bias vs R5 **+0.25** (per-workplace −0.8…+2.7; snapshot-relax
re-stamp 2026-06-10); **0** `perfect ≤ committed` violations, the zero-perturbation guard
(committed == perfect ± grid rounding + seam slack, the board-slack regression test), FIFO-clamp
sortedness, and numba==python (the committed kernel's `commute_all` is **byte-equal** to the
pure-python reference since the snapshot footpath relax — the old ≤2 min tolerance is retired).

> **Caveat:** the committed model fixes only the FIRST leg; the **tail** still re-optimizes
> clairvoyantly (you have real-time info en route, e.g. you check the app and grab the next local). So
> `committed` is a **tight lower bound**, not the full truth — a *fully* committed sim (re-decide at
> every leg as the delay reveals itself) would be a per-leg forward simulation and is the remaining
> upgrade.

Knobs (env): `RAPTOR_MC` (1), `RAPTOR_MC_DRAWS` (24), `RAPTOR_MC_ALT_DRAWS` (12 — now just an alt
on/off gate, >0 = on; no longer drives perturbed traces), `RAPTOR_ALT_WINDOW_MIN` (5 — the alt
dominance window: a line within this many minutes of the cell's best is "also serves"), `RAPTOR_MC_SHAPE`
(2.0), `RAPTOR_MC_MU_{BUS,METRO,CABLE,BART,CALTRAIN}` (70/45/40/25/40 s initial delay mean — BART/
Caltrain matched by feed STEM, not position), `RAPTOR_MC_DEADLINE_STEP` (60 — the MC tail readout
runs on its own finer deadline grid `Tgrid_mc`, since the map's 180 s step rounded every draw up by
U(0,180) s ≈ +1.5 min before the percentile). The served stats are one consistent distribution: the
draws are floored at `perfect` BEFORE p50/p90/std/stuck (so `realistic + frag ≈ p90` per cell), and
the pre-floor p50 is returned as `realistic_raw` for the tests/validator.

## Phase B: hill-aware R5-free walk router + walk-speed toggle (`USE_WALK_GRAPH=1`) — DROPS THE JVM
Replaces R5's last runtime job (the per-workplace walk matrix) with a custom **slope-aware**
pedestrian router. SF is steep and R5 ignores grade, so this is **more accurate**, not just a swap.

- **Build** (`scripts/build_walk_graph.py`, offline, ~10 s): parse `data/osm_sf.pbf` with
  **`esy-osm-pbf`** (pure-Python; two-pass: ways→node-refs, then node coords); sample a USGS 3DEP
  DEM (`scripts/fetch_dem.sh` → `data/dem_sf.tif`, ~10 m, free) per node; weight each **directed**
  half-edge by **Tobler's hiking function** `exp(3.5·(|S+0.05|−0.05))` (gentle downhill fastest,
  uphill/steep-descent slower) × stairs penalty, as REFERENCE seconds @4.8 km/h; keep the giant
  connected component → `data/walk_graph.npz` (215 k nodes, 8 MB). A grade-agnostic `w_flat` is
  stored too, for R5-parity validation.
- **Runtime** (`core/walk.py`, JVM-free numpy+scipy): cKDTree **k-nearest snapping**
  (nearest-edge approx), scipy one-to-many **Dijkstra**. Walking is **directional** — egress
  (alight→work) + pure-walk (home→work) route on the **transposed** graph so uphill≠downhill (on a
  real 16% block: uphill 126 s vs downhill 89 s). `scripts/bake_walk_access.py` rebakes the
  cell→stop access table from the router (same CSR format the engine consumes).
- **Speed toggle** (slow 3.4 / **med 4.2** / fast 5.2 km/h): a `walk_scalar = 4.8/pace` threads
  through access, transfer footpaths, egress, and pure walking in every engine tree/commute path
  (and therefore Monte Carlo through its tree) + the server (`?speed=`, in the cache keys) + the
  frontend (`#speed` control, `&sp=` hash). The access table + egress stay reference seconds,
  cached once. The engine API's `walk_scalar=1.0` default intentionally means the **4.8 km/h bake
  reference**, not served Medium; the server passes Medium's calibrated `4.8/4.2` scalar explicitly.
- **JVM drop:** with `USE_RAPTOR=1 USE_WALK_GRAPH=1` + arrive-by, the server parses flags up top and
  skips the r5py/`com.conveyal` import + the `NET` build (`_NEED_R5=False`); cell coords come from
  the JVM-free grid (`ORIGIN_LL`). Verified: 0 libjvm handles in-process, **RSS ~333 MB** (was
  ~1.6 GB+), CPU-bound → fits a free tier.

**Validation:** `scripts/raptor_validate_walk.py` (W→stops vs R5: flat MAE **24.7 s**/+12 s bias =
mechanics OK; steep >8% grade **100% ours ≥ R5** = the intended hill divergence). Door-to-door via
the walk-baked access table (`bake_walk_access.py`): flat MAE **1.38**, hill **1.67** vs R5 (hill
reads longer on steep cells — more accurate, not an R5 match). Tests: `tests/test_walk.py` (graph
sanity, **directionality**, flat==R5, hill≥flat).

## Run it JVM-free (the full new stack)
```bash
# one-time bakes (the only JVM step is the R5 oracle/access for TESTS; the walk path is JVM-free):
scripts/fetch_dem.sh && .venv/bin/python scripts/build_walk_graph.py   # walk graph (+ DEM)
.venv/bin/python scripts/bake_walk_access.py                           # cell->stop access (JVM-free)
# run the server with the whole stack on, NO JVM:
USE_RAPTOR=1 RAPTOR_SEMANTIC=arriveby USE_WALK_GRAPH=1 .venv/bin/python scripts/server.py
```
Knobs (env): `USE_WALK_GRAPH`, `?speed=slow|med|fast`, `WALK_STAIRS_MULT` (2.5).
Validate: `raptor_validate_walk.py`, `bake_walk_access.py`; tests `pytest tests/test_walk.py`.
