# Agent-Based UI Testing Playbook

This playbook describes a repeatable way to pressure-test the SF Commute Explorer with coding
agents, API probes, and real-browser checks. It is written for this repository, but the workflow is
general: discover difficult states deterministically through the API, replay only the most valuable
states in a browser, and keep model correctness separate from display quality.

The goal is not to make an agent wander around until it sees something suspicious. The goal is to
produce a replayable run record with exact destinations, speeds, cell IDs, coordinates, service
date, measurements, and artifacts so a failure can be reproduced without the original agent.

## What this method tests

The method covers four distinct layers:

1. **Routing truth:** every visible alternative is a real, drawable journey whose totals and
   structural labels agree with the response.
2. **API-to-DOM truth:** the browser displays the same families, branches, services, times, and
   selection state that the API authored.
3. **Interaction behavior:** hover, click, keyboard, tap, scrolling, speed changes, and restored
   permalinks select the intended route and never leave stale UI behind.
4. **Presentation quality:** route-heavy cards remain legible, bounded, scrollable, and useful at
   desktop and mobile sizes.

Do not collapse these into one subjective pass/fail. A beautiful card can advertise a false route;
a truthful card can still be too crowded to use.

## Core principles

### Test the user-facing state machine, not the retired DOM

The 2026-07 route-inspector overhaul exposed a common agent-testing failure: an extensive E2E suite
can become a faithful test of an interface that no longer exists. The current contract is:

```text
desktop cell hover -> compact explanation + primary route
desktop cell click -> focused route inspector
touch first tap -> compact commute preview
touch same-cell second tap or Inspect routes -> focused route inspector
route-row hover/focus -> temporary exact-route preview
route-row click/activation -> locked exact route
mobile Routes/Map -> same selection, different information density
```

Tests should locate `.route-choice[data-key][data-family][data-branch]`, not reconstruct the old
nested family/branch control tree. There is exactly one native route button per authoritative
`choice_key` across the recommendation, practical, and remaining-choice sections; multiple real
Pareto choices may share one family/branch pair. The sections form a disjoint union rather than
repeating the recommendation inside an “all routes” list. `.boarding-group` is noninteractive
structure for the remaining choices only. A test that finds duplicate choice keys or nested
interactive elements is a failure even when clicking happens to work in one browser.

This state-machine framing also catches mobile regressions that a desktop-emulation test misses:
the first tap must not open a two-screen inspector, tapping another cell must replace the preview,
and the preview's close/background actions must clear both the focused tile and transient route.

### Discover with the API, judge with the browser

Browser exploration is the slowest and least deterministic part of the test. Use it only after an
API scan has found route-heavy states.

For each fixed destination and walk speed:

1. Call `/healthz` once and capture `engine`, `semantic`, `walk`, and `svc_date`.
2. Call `/compute?lat=...&lon=...&speed=...` once.
3. Call `/variance?dlat=...&dlon=...&speed=...` once when variance/alternative metadata is needed.
4. Select a small, deterministic set of reachable cell IDs.
5. Call `/itinerary?id=...&dlat=...&dlon=...&speed=...&pin=1` only for those cells.
6. Rank the returned cases by structural complexity.
7. Replay the non-dominated or highest-value cases in a real browser.

The existing implementation is in
`tests/e2e/route_family_hotspots.py` and `tests/e2e/test_route_families.py`. The normal run is
deliberately bounded; the broader scan is opt-in.

### Freeze public inputs

Automated runs must not depend on a private saved workplace, live autocomplete ordering, or a fresh
geocoder result. Keep a versioned catalog of neutral public destinations with:

- a stable slug;
- a human-readable public label;
- latitude and longitude;
- a short reason the destination is useful, such as “dense downtown service” or “edge-of-network
  transfer pressure.”

The current public catalog lives in `tests/e2e/route_family_hotspots.py`. It includes downtown,
Mission Bay, City Hall, Parnassus, Stonestown, Outer Sunset, and Pier 70 examples. Add a destination
because it covers a routing topology, not merely because its address is familiar.

Every artifact must also capture:

- the sampler seed (`ROUTE_FAMILY_HOTSPOT_SEED`);
- `/healthz.svc_date` and routing semantic;
- base URL and build identity (commit plus a dirty-worktree marker when applicable);
- destination coordinates and label;
- exact walk speed;
- exact cell ID and the response's origin coordinates;
- viewport, device mode, and browser version for UI replays.

The service date matters. A route that disappears after a feed update is not automatically a
regression, but a result without its service date cannot be compared honestly.

### Sample deterministically, not uniformly

A citywide random click sweep wastes most of its budget on simple cards. Select candidates from the
computed grid using a seeded mixture of:

- cells with the most alternatives;
- cells with high bad-day fragility;
- near, middle, and far travel-time bands;
- cells near reachability or transfer-count boundaries;
- seeded fill candidates, so the scan does not become a hand-picked golden set.

Use a stable hash of `(seed, destination slug, speed, stratum, cell ID)` for ties. Never rely on
Python's process-randomized `hash()` or API dictionary iteration order.

### Rank complexity as a Pareto problem

A single weighted “complexity score” is convenient for ordering, but it can hide uniquely difficult
states. Keep the full metric vector and retain its Pareto frontier before applying a scalar tie-break.

Useful dimensions include:

- number of route options;
- number of route families;
- total branch count;
- maximum branches or options in one family;
- maximum transfers in one route;
- unique visible and cataloged services;
- label character count and longest label;
- route-time spread;
- geometry point count;
- service/branch cross-product size;
- scroll debt and viewport coverage after rendering.

Case A dominates case B only if A is at least as complex in every selected dimension and strictly
more complex in one. Replay a bounded number of frontier cases, then use a documented scalar score
to fill remaining browser slots. This preserves rare cases such as “few options but one extremely
long label” that a route-count-only ranking would discard.

## Exact replay contract

An address identifies the destination, not the map cell being audited. A browser failure report must
include the returned cell ID and origin coordinates.

Replay destinations through the coordinate permalink rather than geocoding:

```text
/#wp=<dest-lat>,<dest-lon>,<public-label>&metric=r&cmode=time&colors=on&mt=any&sp=<speed>&th=auto
```

Open the permalink in a fresh document (or explicitly reload after changing its hash). The app
does not listen for `hashchange`; same-document navigation to another `#wp=` leaves the previous
destination active and can produce a very convincing but false route regression. Always assert the
runtime destination coordinates before comparing a replay.

Then center the Leaflet map on the saved origin coordinate and convert it with
`map.latLngToContainerPoint`. Drive a real mouse move/click or touchscreen tap at that point. After
the card opens, read the active `routePin` and fetch its exact public `/itinerary` response. If the
canvas chooses a neighboring cell because of projection or pixel boundaries, report both the
requested origin and actual `routePin`; do not silently compare the DOM to the original cell.

Leaflet renders cells on a canvas, so there is no cell DOM node to select. Coordinate replay is more
reliable than scanning arbitrary pixels, while real pointer/touch input still exercises the user
path.

## Routing and false-advertising gates

Run these checks on every sampled pinned response before opening a browser. A failure here is a
model/API issue, not a design opinion.

### Journey consistency

- The primary and every advertised alternative have complete family and branch metadata.
- Every advertised alternative contains at least one transit leg; a slower, line-labeled walk is not
  a route alternative.
- Best-case and typical slots have drawable legs and geometry.
- Geometry begins near the reported origin and ends near the requested destination, using a small
  documented tolerance.
- Leg minutes plus visible waits equal that route slot's displayed total exactly.
- No two options represent the same service sequence, initial direction, destination-facing
  terminal geometry, and metric slots.
- No private selection helpers such as `_family_seed` leak into the public response.

### Family and branch truth

- Exactly the primary option's family is labeled `primary boarding corridor`.
- Every visibly boarded first service appears in its family's service catalog.
- Family `lines` and service-catalog names agree.
- A `shared corridor` tag appears only when the catalog contains multiple usable services.
- A `multiple finishes` tag appears only when more than one branch is actually represented.
- Family and branch keys are opaque structural IDs; tests must not treat concrete line names as the
  specification.
- Every public option has a durable, unique `choice_key`. Family/branch keys group choices; they are
  not a substitute for exact selectable identity.
- Each service's `shown` flag agrees with whether that service has a visible route option.
- Family-to-branch and branch-to-service key references are mutual and point only to catalog rows.
- A branch advertises every visible post-boarding transit tail. It may also include a proven
  pre-cap tail; server-side symbolic tests own the proof for those non-visible catalog entries.
- A walk-finish branch contains one-seat routes; a transit-finish branch contains a transfer.

These checks form a **false-advertising gate**: no label may imply a service, interchangeability,
finish, or route choice that the public response cannot support.

### The hidden-service proof boundary

The public API can prove that its advertised catalog is internally consistent and that every visible
line is represented. It cannot prove that the server did not silently omit a valid line before
building the response.

That completeness claim belongs in server-side symbolic tests. Construct generic fixtures with:

- multiple services sharing a boarding corridor and direction;
- one service selected for display and one valid service removed only by display caps;
- an opposite-direction or loop service that must not join the family;
- services that share the first corridor but diverge toward incompatible destinations;
- dominated and non-dominated tails;
- names that are arbitrary tokens, never production line literals.

Assert that the hidden candidate universe produces the expected public family catalog and branch
relationships. Browser tests should consume only the public response. Do not import server internals
into the E2E process to make an otherwise unprovable assertion appear end-to-end.

## API-to-DOM truth assertions

After a real pointer or touch interaction opens the pinned card, build normalized API and DOM
snapshots and compare them by authoritative keys.

At minimum compare:

- the recommended row's key to the API primary route's authoritative `choice_key` plus its
  family/branch grouping;
- exactly one route row per public `choice_key` across the whole inspector, while allowing several
  choices in one authoritative branch;
- complete API choice/family/branch key sets from the union of recommended, practical, and expanded
  remaining-choice rows, while also asserting the three section-level choice-key sets are pairwise
  disjoint;
- branch-qualified boarding services in the first action sentence;
- option/family/branch counts and the selected row's time, walk, transfer, and bad-day values;
- active exact route key plus family/branch keys after hover, focus, click, keyboard activation, and
  tap;
- one full-strength selected route and only subdued, unlabeled alternative geometry;
- absence of legacy route-strip and nested family/branch-control DOM.

Prefer `data-family`, `data-branch`, and stable element IDs over visible text as selectors. Text is an
asserted output, not a locator. Keep normalization in one helper so server label changes require one
intentional contract update rather than scattered selector edits.

Do not compare a pinned DOM card to a non-pinned response. Pinning can legitimately add planned
branch alternatives. The desktop hover card should be checked against the non-pinned itinerary; the
pinned compare card should be checked against its own `pin=1` response.

### Measure interaction work, not just final state

A mobile inspector can finish in the correct state and still feel broken because it showed a blank
sheet for several seconds or redrew hundreds of Leaflet layers twice per tap. Instrument the work:

- count `/itinerary` requests by URL and assert a preview-to-inspector transition reuses the cached
  lightweight response before making at most one `pin=1` enrichment request;
- wait for alternative discovery before starting expensive pinned enrichment, so a variance arrival
  cannot race a duplicate full lookup;
- wrap the exact draw entry point and assert one draw for a new route key, zero for tapping the
  already-selected key, and zero for switching between Routes and Map views;
- record route-layer count before and after view switches to prove data stayed loaded rather than
  merely leaving the same label selected;
- project the drawn layer bounds into container pixels after a mobile Map-view switch and require
  them to fit inside the map region above the compact sheet; layer count alone cannot detect a
  correctly loaded route that the current camera placed entirely offscreen;
- collect failed requests, HTTP errors, `console.error`, and page exceptions throughout the flow.

This is a broadly useful pattern: assert both **time-to-useful-content** and **work performed**.
A test that waits only for eventual DOM text will miss the regressions users describe as “slow,”
“blank,” or “the route did not load.”

### Separate visual opacity from feature restyling

When a visual control affects thousands of map cells but not their data classification, apply it at
the shared renderer/pane level and test that the input handler does not call the per-feature redraw
path. For the heatmap opacity control, one CSS variable changes the Leaflet canvas compositor in
O(1); the selected cell has its own pane and contrast halo. Agent tests should verify both the DOM
control contract and the absence of a `layer.setStyle`/recompute call on slider input. This keeps a
seemingly cosmetic control from becoming a hidden mobile performance regression.

## Objective display measurements

An agent's “looks crowded” judgment is useful triage, not a stable regression assertion. Record the
judgment, but pair it with measurements.

Measure at least:

- horizontal overflow: `scrollWidth - clientWidth` for the card and scroll body;
- vertical scroll debt: `scrollHeight - clientHeight`;
- clipped labels: visible elements whose scroll dimensions exceed client dimensions under hidden or
  clipped overflow;
- card containment inside the viewport;
- overlap area with the settings panel, legend, marker controls, and close button;
- whether the close control remains fully visible and operable;
- whether scrolling to the bottom makes the last expert route choice fully visible;
- minimum interactive target dimension;
- number of families, branches, options, and label characters visible at once;
- percentage of viewport obscured by the card;
- route-line visibility on the remaining map;
- unexpected document-level horizontal scrolling.

Record soft thresholds separately from hard gates. Examples:

- **Hard:** horizontal overflow greater than 1 px, clipped essential labels, an inaccessible close
  button, unreachable final option, or a card outside the viewport.
- **Soft/design review:** more than one viewport of vertical scroll, the card covering most of the
  useful map, repetitive labels, indistinguishable service chips, or a route union too dense to
  follow.

For soft findings, capture a screenshot and state the task the UI makes difficult, such as “cannot
tell which services are interchangeable at boarding” rather than only “too busy.”

## Viewports and page states

At minimum replay every saved high-value hotspot in:

| Mode | Viewport | Input path | Why |
| --- | ---: | --- | --- |
| Desktop | 1280 x 800 | hover, click, keyboard | Normal map and compare-card state |
| Tight desktop | 900 x 700 | hover, click, keyboard | Early collision and scroll pressure |
| Mobile | 390 x 844, touch | first tap preview, Inspect, Route/Map | Responsive touch behavior without hover |

Also test these states when they are relevant:

- top of the card and fully scrolled to its bottom;
- the document or mobile sheet already scrolled before the route card opens;
- settings panel expanded and collapsed;
- legend visible with long contents;
- one exact route previewed, locked, then preview focus cleared;
- metric toggle after pinning;
- speed changed while the same origin/destination card is open;
- restored permalink and returning-user localStorage, each in a clean browser context;
- loading, retryable 503/429, empty, and failed itinerary states.

The scrolled-page state is important: viewport-relative positioning can look correct at `scrollY=0`
and detach from the map or cover controls after the page/sheet moves.

## Slow, medium, and fast pressure tests

Compare `slow`, `med`, and `fast` for the **same destination and exact cell ID**. If a cell is not
reachable at all speeds, keep that as an explicit reachability-boundary result instead of replacing
it with a different origin.

Capture for each speed:

- map p5/p50 values and reachability;
- primary and alternative totals in both metric slots;
- family, branch, and service-key sets;
- first boarding direction and transit sequence;
- walk-leg minutes and geometry endpoints;
- number/order of displayed options;
- card metrics and screenshot state.

Classify changes rather than requiring byte-identical responses:

1. **Expected metric change:** walking legs shorten and total time improves or stays equal.
2. **Expected structural change:** a faster pace makes a different stop/corridor genuinely optimal or
   opens a new non-dominated route.
3. **Suspicious churn:** a corridor disappears and reappears, an unrelated family key changes while
   geometry is structurally the same, or labels change without a route change.
4. **Contract failure:** faster walking makes the served commute worse where the active routing
   semantic promises true-zero monotonicity, the same journey's walk leg gets longer, a reachable
   cell becomes unreachable contrary to the active contract, or a displayed option becomes faster
   than the primary.
5. **Display failure:** the API change is valid but the card retains old services, focus, times, or
   route geometry after the speed response lands.

Check the routing semantic captured from `/healthz` before enforcing strict monotonicity. The repo's
default depart-after path is intended to satisfy true-zero walk-speed monotonicity. An opt-in semantic
with documented reconstruction/candidate limitations may need a diagnostic threshold instead of the
same hard assertion; record the exception rather than quietly skipping the comparison.

Useful cross-speed invariants are structural and generic: family keys should remain stable when the
boarding corridor/direction is unchanged, branch keys should remain stable when the destination-facing
tail is unchanged, and a faster pace must not turn an unavailable service into an advertised one
without a real supporting itinerary.

## Cold-cache and warm-cache checks

Run both because cache bugs often look like speed-toggle or destination contamination.

### Warm cache

- Request the same destination, speed, and cell twice.
- Confirm payload semantics and ordering are identical.
- Confirm the second request is not materially slower.
- Switch away to another cell and back; check focus, card contents, and geometry are restored from the
  correct entry.
- Repeat a pinned request before and after `/variance`, since alternative metadata may move between
  pre-variance and Monte Carlo caches.

### Cold cache

- Start from a clean, known server process and a fresh browser context.
- Capture boot time and the first compute/variance/itinerary latencies separately.
- Repeat the saved replay record after reboot and compare opaque keys, ordering, and payload truth.
- Treat service-date or source-file changes as a new baseline, not cache nondeterminism.

Do not approximate a cold server by clearing browser storage. That tests only client state.

## Console, network, and asynchronous-state checks

Every browser auditor should collect:

- `console.error` messages;
- uncaught page errors;
- failed requests;
- response status for app endpoints;
- request URL, including destination, speed, cell, and pin parameters;
- retry counts and `Retry-After` handling for 429/503 responses;
- requests that complete after a newer destination or speed selection.

A visually correct final card can still hide a stale-response race. Tag destination and speed changes
with a monotonically increasing test-side action number, then verify that older responses never win
the DOM or redraw the route. Explicitly exercise quick `slow -> fast -> med` changes and two rapid
destination changes on a warm cache.

## Screenshot and artifact hygiene

Screenshots are evidence, not the test oracle.

- Store the complete machine-readable scan as JSON; `tests/e2e/screens/route_family_hotspots.json` is
  the current default and is git-ignored.
- Use `/tmp` or a run-specific ignored directory for exploratory screenshots.
- Name files with run ID, service date, destination slug, speed, cell ID, viewport, and state, for
  example `20260714_2026-05-20_townsend_fast_1842_mobile_scrolled.png`.
- Capture one full UI state and a tighter card crop for each failure. Do not generate screenshots for
  every simple passing cell.
- Save the normalized API and DOM snapshots beside a failure screenshot.
- Keep artifacts from different agents in separate run directories to prevent overwrites.
- Never use or expose the user's saved workplace. Clear localStorage and hashes before loading public
  fixtures.
- Do not commit transient screenshots or new “goldens” merely because a feed or font changed. A human
  should approve stable visual baselines.

The failure record should reference artifact paths relative to the repo or an explicitly named `/tmp`
run directory. An unexplained screenshot attachment is not a reproducible bug report.

## Avoiding stale tests and selectors

UI migrations often leave tests that pass against elements no longer used by the product.

- Anchor selectors under the currently visible card (`#pincard .cmp`) and assert the container is
  open before inspecting descendants.
- Prefer stable IDs, roles, and the complete `data-key`/`data-family`/`data-branch` tuple.
- Assert that legacy containers or route strips are absent when their removal is part of the current
  contract.
- Use visible user signals to wait for computation; do not sleep as the primary synchronization
  mechanism.
- Keep a small settling wait only for a documented one-shot layout/animation pass, then measure.
- Fail when a locator matches zero **or more than the expected count**. Broad `.strip`, `.card`, or
  text selectors can silently attach to stale markup.
- Centralize API normalization, DOM snapshots, and current selectors in helpers.
- When a label changes intentionally, update expected normalized output and explain the product
  decision; do not weaken the assertion to substring matching.
- Periodically run a selector audit: list every E2E selector, map it to current template markup in
  `scripts/templates/index.html`, and remove tests for dead surfaces.

The current stable route-inspection selectors are `#touchpeek.open`, `#peekinspect`,
`#pincard.open`, `.pin-summary`, `.route-choice`, `.route-choice.recommended`,
`.practical-alternatives`, `#allroutes`, `.boarding-group`, `#route-directions`, `#pinview`, and `#pinadjust`.
Keep selectors scoped under the open inspector and assert route keys are disjoint across the
recommended, practical, and optional remaining-choice sections.

## Lessons from the 2026-07 agent pressure test

1. **A structural API regression can masquerade as a missing chip.** The missing transfer finish
   was caused by branch-enumeration order: a one-seat walk sibling appeared only after the one-tail
   pass had already run. Minimize this class of failure with arbitrary service tokens and a bounded
   closure test before debugging CSS.
2. **One golden address is a confirmation case, not a discovery strategy.** Preserve the exact
   saved cell for regression, but scan deterministic time/fragility strata across several public
   destinations to discover other dense states.
3. **Visible traces and service catalogs prove different things.** The chosen representative trace
   proves what is drawn; the server-authored branch service catalog proves what can be boarded on
   that corridor. The first action sentence may use the catalog, while later transfers must stay
   exact to the selected geometry.
4. **Perceptual agents need objective gates.** Ask a reviewer whether the UI feels crowded only
   after measuring overflow, clipping, target size, viewport obstruction, route-layer count, and
   scroll reachability. This makes the judgment actionable rather than aesthetic guesswork.
5. **Tile focus and opacity should be composited, not recomputed.** Persistent map strength belongs
   on the shared heatmap canvas. Explicit tap/pin selection uses two tiny dedicated layers (contrast
   halo plus selected fill); hover changes neither city opacity nor tile focus. Restyling thousands
   of polygons per pointer movement creates avoidable jitter and test flakiness.
6. **Mobile is a state machine, not smaller CSS.** Real touch checks must cover preview replacement,
   preview dismissal, inspection, Route/Map switching, settings adjustment, safe areas, and focus
   cleanup. Desktop hover assertions do not imply any of these work.
7. **Walk-speed checks need both monotonicity and structural classification.** `slow >= med >= fast`
   is the first gate, but an agent should also report whether the family/branch changed because a
   real access opportunity opened or because identity churned without geometric support.
8. **Parallelize analysis, not the local router.** Agents can independently review code, artifacts,
   screenshots, and viewport assignments. They must share one bounded endpoint queue and never run
   competing JIT-enabled Python processes against the same numba cache.
9. **Wait on the product contract, not an obsolete implementation label.** RAPTOR intentionally
   omits the legacy `fast ~Nms` string. The durable completion signal is a retained destination
   label plus populated neighborhood output. If every dependent browser test times out after 200
   responses, audit the shared wait helper before diagnosing a product cascade.
10. **Quantized keys are indexes, not final equality proofs.** A hard case exposed the identical
    service/tail choice twice because headings 18 degrees apart landed in adjacent coarse bins.
    Use continuous structural comparison for final dedupe (same qualified ride sequence, same
    terminal branch, non-opposite direction), keep true reverse rides distinct, and retain a saved
    regression on both sides of a quantizer edge.
11. **Test the current touch state machine end to end.** A first tap now opens `#touchpeek`; Inspect
    promotes cached content into `#pincard` and enriches in the background. A test that waits for a
    legacy Leaflet popup can report a false failure while the real mobile interaction works. Assert
    preview content, promotion, loaded route rows, no hover dependency, and stale-request immunity.

## Lessons from the 2026-08 route-truth and inspector pass

1. **A rounded minute can contain two different kinds of time.** The planned router may intentionally
   hold a few seconds or minutes before first boarding so a route remains feasible. Carry
   `physical_sec` and `schedule_allowance_sec` independently through wait folding, excess shaving,
   dropped sub-minute legs, formatting, geometry, and JSON. Reconcile positive slack onto the first
   access leg as allowance; never let a generic formatter append it to egress and call it walking.
2. **The graph's bake speed is not a product preset.** The walk graph is stored at 4.8 km/h, while
   the product presets are 3.4 / 4.2 / 5.2 km/h. A scalar of `1.0` means “reference graph pace,” not
   Medium. Every HTTP default, direct product entry point, test fixture, cache key, and CLI must
   either derive `4.8 / selected_pace` or say explicitly that it is exercising the reference API.
3. **Grouping identity and selectable identity are different contracts.** Family keys answer “same
   boarding corridor,” branch keys answer “same destination-facing finish,” and `choice_key`
   answers “this exact non-dominated option.” Using `(family, branch)` as a DOM key silently erases a
   second real tradeoff inside the branch. API and DOM audits must compare all three levels.
4. **Progressive disclosure needs an exhaustiveness proof.** Featured rows should be one
   recommendation plus at most three genuinely distinct tradeoffs. The single Additional disclosure
   must contain the exact set difference by `choice_key`, and its boarding-context headings must not
   mix stops or directions merely because they share a corridor family.
5. **Preserve truth metadata through reconciliation, not just in the first serializer.** A local fix
   to the primary trace can leave via-stop, branch-closure, or raw-formatting paths wrong. Inventory
   every formatter call site, centralize the signed target residual, and add at least one primary,
   one alternative, and one no-access-leg fixture.
6. **One API sampler should feed many reviewers.** Parallel agents are useful for static code review,
   desktop grading, mobile grading, and speed-diff analysis. They should consume one immutable
   hotspot artifact and one shared server rather than each warming the router, rediscovering the
   same cases, or consuming the variance budget independently.
7. **Record request identity and readiness, not only a screenshot.** A useful artifact includes
   destination, cell, speed, `pin`, cache state, response generation, and the moment the lightweight
   card and enriched card each became usable. This distinguishes genuine layout defects from stale
   responses, cold JIT, or a screenshot captured between progressive renders.
8. **Bound visual iteration.** Run objective API/DOM/overflow/target-size gates first, then ask a
   perceptual reviewer to grade only the Pareto-frontier cards. One implementation review and one
   confirmation pass catches density problems without turning subjective screenshot tweaking into
   an open-ended loop.

## Local execution and concurrency gotchas

The live browser suite drives an already-running server. It does not own or restart that process.
The standard entry points are:

```bash
PORT=8123 .venv/bin/python scripts/server.py
E2E_BASE_URL=http://127.0.0.1:8123 tests/e2e/run.sh test_route_families.py
ROUTE_FAMILY_HOTSPOT_SCAN=1 E2E_BASE_URL=http://127.0.0.1:8123 \
  tests/e2e/run.sh test_route_families.py -k broad
node --test tests/test_viz.mjs
.venv/bin/python -m pytest tests/ --ignore=tests/e2e -q
```

Important constraints:

- **Never run two Python processes that JIT the routing engine into the same numba cache.** In
  particular, do not boot the server beside pytest or run Python test shards in parallel. This can
  corrupt `.nbc` cache files. Stop the server before Python unit/integration tests, or deliberately
  isolate `NUMBA_CACHE_DIR` per process and verify no other writable caches are shared.
- Do not start a second server on the same port to get a “cold” run. Identify and stop the exact
  process first; avoid broad `pkill` commands.
- Keep heavy `/compute`, `/variance`, and `/itinerary?pin=1` calls sequential. Parallel agents can
  analyze different artifacts, but should not stampede one local server.
- The broad sampler enforces the live `/variance` budget: at most 18 destination-speed configs and
  90 pinned itineraries in its current defaults. Preserve explicit bounds.
- A sandbox may block localhost/network access or browser launch even though the server is healthy.
  Distinguish permission failures from product failures and request the narrowest appropriate
  approval.
- Browser setup is optional and may require the repo-local Playwright/Chromium dependencies listed
  in `tests/e2e/README.md`. Missing browser binaries are environment failures, not skipped product
  checks.
- Avoid geocoding during the route-card audit. It introduces external network, cache, provider, and
  rate-limit variability unrelated to the card.
- Keep the public server process running for browser-only agents, then stop it before the full Python
  suite. Run the JS suite at any time because it does not JIT the Python engine.

## Multi-agent operating model

Parallelism should separate roles, not duplicate traffic.

### Recommended roles

1. **Coordinator:** freezes the replay record, health response, seed, service date, budgets, and artifact
   directory; assigns disjoint ownership; merges findings.
2. **API sampler:** performs the sequential deterministic scan, validates false-advertising gates,
   computes the Pareto frontier, and writes one immutable JSON artifact.
3. **Desktop interaction auditor:** replays assigned frontier cases at desktop and tight-desktop
   sizes; captures API-to-DOM, keyboard, focus, scroll, console, and network evidence.
4. **Mobile interaction auditor:** replays a disjoint case set with real touch input and responsive
   scrolling; checks bottom-sheet and control reachability.
5. **Cross-speed auditor:** compares the same OD/cell across slow, medium, and fast using the API
   artifact first, then replays only suspicious transitions.
6. **Adversarial/design reviewer:** grades information hierarchy, map obstruction, density, and
   comprehensibility after objective gates have passed. This role proposes design work; it does not
   redefine routing truth.
7. **Fix reviewer:** after implementation, reruns the exact failed replay and searches for nearby
   regressions. It should not be the same agent that authored the fix when an independent review is
   available.

With limited agent slots, combine desktop/mobile or cross-speed/design roles, but keep the sampler a
single writer. Agents should never edit the same test or application file concurrently.

### Handoff/report format

Each finding should contain:

```text
ID: RF-UI-007
Severity: blocker | high | medium | low | design
Confidence: high | medium | exploratory
Reproduced: 3/3
Build: <commit> + concise dirty-worktree status
Server: <base URL>, engine, semantic, svc_date, cold/warm
Input: destination label + lat/lon, speed, cell ID + origin lat/lon
Browser: name/version, viewport, touch/mobile flags, scroll state
API: exact endpoint and normalized expected snapshot
Steps: minimal numbered replay steps
Expected: concrete routing or display contract
Actual: concrete difference and objective measurements
Artifacts: JSON, DOM snapshot, network log, screenshot paths
Classification: routing | API serialization | DOM truth | interaction | layout | design | harness | environment
Hypothesis: optional, clearly marked as inference
```

The coordinator should deduplicate findings by `(contract, destination, speed, cell, viewport)` and
preserve distinct manifestations if one root cause affects several user paths.

## Failure triage

Use this order to avoid fixing the harness around a real bug:

1. Replay the exact public API request outside the browser.
2. Validate the response against routing and false-advertising gates.
3. If the API fails, minimize the response to a generic server-side symbolic fixture.
4. If the API passes, compare a normalized DOM snapshot by family/branch keys.
5. If DOM truth passes, inspect interaction state, route layers, overflow, and scroll measurements.
6. Reproduce in a fresh context and a warm context. If only one fails, inspect caching and stale
   responses.
7. Reproduce at the adjacent walk speeds for the same cell. Classify a genuine route opening
   separately from unstable identity or stale UI.
8. Check console/network logs before calling a visual symptom a CSS bug.
9. Verify the selector still points to the active UI. A zero-match or hidden legacy match is a
   harness failure, not a product pass.
10. Only after the objective layers pass should a perceptual finding enter the design backlog.

Do not relax an assertion merely because a failure appears feed-dependent. First determine whether
the assertion encodes a structural contract or a concrete line/date accident. Replace the latter
with generic fixtures and keep the former.

## Optimizing runtime, cost, and reproducibility

- **One scan, many readers:** write one immutable API artifact and let browser/design agents consume
  it. Do not make every agent rediscover hotspots.
- **Bound the default:** one public destination, one speed, and roughly four candidates is enough for
  a smoke gate. Make the citywide and three-speed matrices explicit opt-ins.
- **Reuse compute results:** compute and variance once per destination-speed config, then pin only
  selected cells.
- **Use a Pareto frontier:** spend browser time on structurally distinct hard cases instead of five
  nearly identical high-score cards.
- **Replay exact coordinates:** bypass geocoding, autocomplete, and pixel-grid hunting.
- **Use staged escalation:** JS/static contract tests first, bounded API scan second, saved browser
  replay third, broad scan fourth, manual design review last.
- **Separate discovery from confirmation:** a broad run finds candidates; a small saved replay
  confirms them on every change.
- **Capture failures immediately:** save response and DOM snapshots at assertion time so later cache
  changes do not erase evidence.
- **Keep perceptual review sparse:** ask the agent to grade only the frontier cases with a rubric.
- **Make thresholds visible:** hard-code named, justified budgets rather than letting each agent
  decide what “crowded” means.
- **Avoid process parallelism around JIT:** parallelize analysis and viewport review, not Python boots
  or heavy endpoint calls.
- **Record timings but compare distributions:** cold and warm timings are diagnostics; use generous
  regression bands unless the environment is controlled.
- **Name cache state honestly:** “first in the progressive flow” is not process-cold. Itinerary
  intentionally reuses compute, pin enrichment intentionally reuses variance, and later speed
  cases can reuse speed-independent walking state. Reserve “cold” for a restarted process with a
  single isolated case, and record those dependencies in the artifact.
- **Validate the target before trusting a fast number:** assert engine, routing semantic, walk
  backend, service date, nonempty selected-cell responses, and stable process/configuration across
  the run. A fast empty `200`, wrong fallback stack, or restarted server is a harness failure.
- **Time product-ready markers, not request completion:** measure hover to a visible non-pending
  route, tap to a route-bearing preview, Inspect to a useful light card, and Inspect to committed
  pin enrichment. Navigation plus a populated neighborhood list does not cover route interaction.
- **Force adverse response order:** deferred tests should resolve light, variance, and pin requests
  in every meaningful order. The richer committed response must win; one in-flight owner must
  prevent duplicate refresh requests; terminal variance failure must clear pending UI and expose a
  usable structural-route fallback.
- **Seed every tie:** destination subset, cell stratum, frontier tie, and screenshot choice should all
  derive from the run seed.

### Lessons from exact performance-agent work

- **Separate three kinds of “first”:** process-first may compile; cache-first in a new process may
  deserialize compiled code; flow-first intentionally reuses earlier endpoint state. Record all
  three when investigating latency and use unprofiled wall time for the user-facing number.
- **Treat profiler time as attribution, not latency:** `cProfile` can inflate Python-heavy routing
  stages by 1.5-3x and changes thread/JIT behavior. Use it to rank functions, then confirm every
  claimed gain in a clean unprofiled child process.
- **Isolate small compiled accelerators:** editing one function in a large Numba module can invalidate
  unrelated cached kernels. Put independent route-card accelerators in their own module and exercise
  every real dtype/mutability signature during install-time warmup.
- **Shadow exact responses before and after:** normalize only explicitly ignored instrumentation
  fields, then compare the full response objects directly—including ordered routes, totals, stop
  ids, and geometry—not only a headline time. Do not add response checksums or integrity tokens to
  the benchmark. For array or indexing rewrites, retain a pure reference implementation and compare
  every output row with `numpy.array_equal`.
- **Index only on necessary predicates:** replace all-pairs structural comparisons with a bucket only
  when membership is provably required for equivalence/dominance. Keep a randomized adversarial test
  that compares the indexed result to the original reference by object identity and order.
- **Compile parent chains against an oracle:** faster predecessor materialization is safe only when
  every reachable stop's complete raw chain matches the reference across representative deadlines,
  not merely when best-time arrays match.
- **Measure nested cache payloads:** a count-bounded top-level LRU is not a memory bound when entries
  grow deadline trees, route geometry, branch geometry, or per-cell typicals. Cap those nested maps,
  account for retained scenarios and numpy buffers, and stress distinct-cell pokes as well as repeat
  hits.
- **Do not parallelize JIT boots between agents:** delegate static review, DOM analysis, and artifact
  grading in parallel; serialize child-process routing, server startup, and Python suites. A shared
  Numba cache can otherwise corrupt or produce meaningless contention timings.
- **Distinguish speed from displacement:** if an optimization moves work from pin to variance,
  startup, or a background task, report both endpoint and end-to-end flow latency. The product is
  faster only when the ready-to-use marker moves earlier without starving another interaction.
- **Measure thread scaling before rewriting an exact kernel:** the 24-draw committed kernel scaled
  almost linearly from one to four real local threads, then flattened by eight. That evidence is
  useful for attributing parallel work, but it does not authorize raising production threads on a
  one-OCPU host; record physical CPU allocation and test under representative contention.
- **Measure retained-state tax instead of assuming it is the problem:** lossless MC tail capture
  added only about 3 ms at two threads while retaining about 16 MiB and making pin replay cheap.
  Keep or remove such state based on endpoint time, hit rate, and bounded bytes—not architectural
  taste. In this case the repeated reverse profiles, not capture, remain the V4 target.
- **Reject performance samples taken under host saturation:** record load average, configured and
  observed Numba thread counts, threading layer, and competing processes next to every kernel run.
  A run at load 26 on 11 logical CPUs made a healthy parallel kernel look two to three times slower
  and nearly erased the difference between one and two workers. Treat that sample as invalid, not
  as evidence for a rewrite.
- **Profile phase scopes before comparing microbenchmarks:** an endpoint's “committed” phase can
  include draw allocation, walk scaling, deadline preparation, and materialization around a fast
  compiled extractor. Likewise, a planned-overlay phase includes candidate preparation, B3 traces,
  grouping, and bundle construction. Name the narrow scope and the endpoint scope separately.
- **Compare private route-analysis state across cache orders:** public-response equality can hide a
  changed private alternative bundle that later feeds pin enrichment. Differential tests should
  cover fresh trees, warmed trees, and multiple deadline-prebuild orders, then compare both private
  normalized structures and downstream public variance/pin responses.
- **Make test/reload lifecycle policy explicit:** a boot-only `init()` must reject in-flight work,
  invalidate every graph-derived cache before publishing new globals, and never blindly clear a
  flight registry whose old owner may still publish. If live reload is required, use generation
  keys and a real request barrier instead of a partial cache clear.
- **Warm optional compiled topology with synthetic ABI fixtures:** deployment must not depend on a
  current service-day route happening to contain a transfer/tail shape. Compile the real mixed-dtype
  signature directly, then use the real public route only as an integration smoke.

### Lessons from mobile request-agent work

- **Coalesce the lightweight request as well as enrichment:** a delayed `pin=1` prefetch can be
  perfectly bounded while Preview → Inspect still issues two ordinary itinerary calls. Key one
  abortable light-request owner by destination generation, cell, and routing parameters; let
  Preview and Inspect subscribe to it, and give rendering to the latest subscriber token.
- **Test the unresolved-promise transition:** the critical mobile race is Inspect before Preview's
  light response settles. A useful harness holds that response, asserts one light fetch, resolves
  it, and then asserts at most one enrichment plus the final card/cache owner. A test that begins
  from an already-cached light response misses the bug.
- **Choose one focus model per surface:** a pointer-opened nonmodal preview should announce itself
  without stealing focus. An explicitly invoked inspector may receive focus, but its role, trapping,
  close behavior, and restoration must agree. Do not label a full-screen region as modal while
  continuing to let the map behave as an independent surface.
- **Validate hybrid input in a real browser:** source-level `pointerType` fixtures do not prove what
  Leaflet emits after synthesized touch/click events. Test coarse touch, fine mouse, narrow desktop,
  and a hybrid device; if needed remember the last real pointerdown modality rather than guessing
  from viewport width.

## Example test matrices

### Bounded pre-merge matrix

Use for ordinary route-family/UI changes.

| Layer | Destinations | Speeds | Cells | Viewports | Cache |
| --- | ---: | --- | ---: | --- | --- |
| JS contract | fixtures | all symbolic | fixture set | DOM-free | n/a |
| API hotspot smoke | saved public Townsend destination | medium | 4 seeded | n/a | warm |
| Browser replay | saved Mission-to-Townsend hotspot | medium | 1 exact | 1280x800, 900x700, 390x844 | warm |
| Cross-speed API | saved hotspot | slow, medium, fast | 1 exact | n/a | warm |
| Full non-E2E suite | fixture/golden set | configured | all | n/a | server stopped |

Suggested commands:

```bash
node --test tests/test_viz.mjs
E2E_BASE_URL=http://127.0.0.1:8123 tests/e2e/run.sh test_route_families.py
# Stop the live server before this Python suite:
.venv/bin/python -m pytest tests/ --ignore=tests/e2e -q
```

### Broad scheduled/adversarial matrix

Use before a major route-family release or after a feed/service-date change.

| Layer | Destinations | Speeds | Cells | Viewports | Cache |
| --- | ---: | --- | ---: | --- | --- |
| API discovery | 6 seeded from the public catalog | slow, medium, fast | up to 5/config | n/a | warm |
| Frontier replay | non-dominated set, capped at 8-12 total | implicated speed plus adjacent speeds | exact IDs | desktop + mobile | warm |
| Tight-layout replay | top label/branch/scroll cases | implicated speed | 3-5 | 900x700 + 390x844 | warm |
| Race/state | 2 contrasting destinations | slow -> fast -> medium | 1 each | desktop + mobile | warm |
| Cold confirmation | saved failures and top 2 frontier cases | implicated speeds | exact IDs | failing viewport | cold then warm |
| Independent design review | top 3 densest valid cards | representative | exact IDs | all three | warm |

The current broad command stays within the harness's endpoint budgets:

```bash
ROUTE_FAMILY_HOTSPOT_SCAN=1 \
ROUTE_FAMILY_HOTSPOT_DESTS=6 \
ROUTE_FAMILY_HOTSPOT_SPEEDS=slow,med,fast \
ROUTE_FAMILY_HOTSPOT_PER_CONFIG=5 \
ROUTE_FAMILY_HOTSPOT_SEED=20260712 \
E2E_BASE_URL=http://127.0.0.1:8123 \
tests/e2e/run.sh test_route_families.py -k broad
```

If more than six destinations are needed across all three speeds, split the catalog into separate
bounded replay sets or restart after the rate-limit window. Do not remove the budget assertion.

## Actionable run checklist

## Completed evidence snapshot — 2026-08-07

Keep completed facts separate from the repeatable procedure above. The recorded
full suite completed with **288 Python tests and 83 JS tests**. That establishes
the tested correctness contracts; it does not certify latency under a clean host.

Same-cell cross-speed evidence retained these slow/medium/fast time values:

| Case | Slow / medium / fast | Confirmed route-contract evidence |
| --- | --- | --- |
| Townsend 1916 | 26 / 24 / 22 | M/K/L + 19 remained supported |
| Townsend 2066 | 38 / 35 / 33 | 22 > 19 remained true |
| Outer 2406 | 47 / 47 / 47 | every advertised service was backed by an itinerary |

The mobile review grade was **B** and desktop behavior was functional. The mobile
header-truncation and weak selected-cell-at-low-opacity findings have now been
implemented in `scripts/templates/index.html` and covered by the 83 JS tests.
Independent static review and final live postfix smoke pass. At mobile 320 and
390 widths, the smoke found no overflow or ellipsis. True preview dismissal
returns focus to the map without leaving focus in an `aria-hidden` region and
without affecting Inspect. At desktop 1280 with 20% heatmap opacity, the selected
cell painted a double outline; hover-only state painted none. The postfix smoke
recorded no console or page errors; artifacts are
`/tmp/sfci-ui-20260807/postfix-*.png`. The open findings are hybrid native-touch
`pointerType` behavior and 2477 px of More-section scroll debt. Classify these as
presentation/follow-up findings unless a saved replay shows a routing or API-to-DOM
violation.

Latency remains unproven: ordinary compute is around 100 ms in available results,
and neither sub-100-ms first enriched pin nor sub-100-ms exact variance has been
shown. A clean-host rebenchmark is still required; do not use contended-host timing
as a regression or success claim.

The 2026-08-09 raw-time calibration also exposed one retained cross-speed Pareto-label exception:
Bayview golden cell 2840 is 5 minutes at Slow and 6 at Medium because a faster walk-only reverse
label discards a still-feasible 23 transit continuation. The permanent test names that exact tuple
and rejects every additional exception. Agents should classify that saved replay as a known engine
limitation; any other faster-walk-longer result is a new contract failure.

### Before the run

- [ ] Confirm the intended application build and record dirty-worktree state.
- [ ] Confirm exactly one server process and the intended base URL.
- [ ] Save `/healthz`, including engine, semantic, walk mode, and service date.
- [ ] Choose bounded or broad matrix and freeze destination catalog, seed, and endpoint budgets.
- [ ] Create a unique ignored artifact directory; never reuse another agent's paths.
- [ ] Ensure fixtures use only neutral public destinations and clear browser persistence.
- [ ] Assign one API sampler and disjoint browser/review roles.

### API discovery

- [ ] Compute once per destination-speed pair.
- [ ] Fetch variance once per pair if needed.
- [ ] Select deterministic alternative-rich, fragile, time-stratified, and seeded cells.
- [ ] Fetch pinned itineraries sequentially.
- [ ] Enforce journey consistency and false-advertising gates.
- [ ] Store the full metric vector, calculate the Pareto frontier, then apply a stable tie-break.
- [ ] Save exact cell IDs, origin coordinates, API paths, payload snapshots, seed, and service date.

### Browser replay

- [ ] Open the coordinate permalink, not a geocoded address.
- [ ] Use real pointer, keyboard, and touch paths.
- [ ] Compare hover DOM to non-pinned API and pinned DOM to `pin=1` API.
- [ ] Compare family/branch/service keys, labels, times, counts, and active route drawing.
- [ ] Measure overflow, clipping, overlap, card containment, target size, and scroll reachability.
- [ ] Test desktop, tight desktop, mobile, and already-scrolled states.
- [ ] Collect console errors, page errors, failed requests, status codes, and stale-response evidence.
- [ ] Capture screenshots only for failures and representative frontier passes.

### Cross-speed and cache

- [ ] Compare slow, medium, and fast at the same destination and cell ID.
- [ ] Separate expected structural changes from suspicious identity churn and hard contract failures.
- [ ] Verify the DOM clears old services, times, focus, and route layers after each speed change.
- [ ] Repeat representative requests warm.
- [ ] Reboot once for cold confirmation; capture service/data identity again.
- [ ] Never overlap server JIT with Python tests or another server boot.

### Reporting and completion

- [ ] Classify every failure by routing, API, DOM, interaction, layout, design, harness, or environment.
- [ ] Include exact replay inputs, measurements, expected/actual, and artifact paths.
- [ ] Minimize routing failures into generic symbolic fixtures without production line literals.
- [ ] Rerun each fixed failure from its saved replay record.
- [ ] Run an independent adversarial review after major fixes.
- [ ] Run `node --test tests/test_viz.mjs`.
- [ ] Stop the server, then run `.venv/bin/python -m pytest tests/ --ignore=tests/e2e -q`.
- [ ] Restart the intended live server and repeat the bounded smoke check.
- [ ] Preserve durable lessons here; keep transient run results in ignored artifacts or a dated report.

## When to change this playbook

Update this document when the public response schema, route-card DOM, routing semantic, rate limits,
cache lifecycle, or browser entry points change. A new line name, destination, or one-off visual bug
does not by itself justify a new testing rule. Capture the reusable structural lesson and keep the
fixtures generic.

### Deployment evidence lesson

For a production smoke, record deployed build provenance, service-unit/preflight
result, public and origin health metadata, memory limits and observed usage, and
the exact endpoint versus server-log timing scopes. If the host cannot identify a
Git revision, retain a clean local export or saved file set for rollback. Treat
such timings as smoke evidence unless the host and measurement protocol are a
controlled SLA benchmark; record browser-smoke completion or its concrete blocker
separately. Test each configured third-party dependency from the actual service
host and service user, not merely by confirming that its API key or configuration
value is present. For a Caddy deployment, preserve SELinux policy, send JSON
access logs to stdout/journald, and when `admin off` is configured, validate the
Caddyfile and apply it with `systemctl restart` rather than reload.

## Field notes — 2026-08-09 agent-browser pressure tests

- Stagger agents that make heavy route requests. Parallelize artifact reading and visual grading,
  but serialize `/compute`, `/variance`, and enriched itinerary probes so the result remains
  attributable and the local router is not turned into the test subject.
- Device emulation can resize a viewport without making the application receive real touch input.
  Dispatch and observe `pointerType` during each mobile/hybrid scenario; do not infer modality from
  screen width or the browser's device flag alone.
- After a blank boot, hash-only navigation may not reapply workplace state. Load each public fixture
  through a fresh document or query-path entry, then assert the runtime destination before grading
  the selected cell.
- Exact structural branches may be genuinely distinct yet read as duplicate headline choices. Rank
  and collapse their top-level labels for comprehension, while retaining every branch under an
  explicit disclosure; this is presentation policy, not permission to discard routes.
- Inspect the network work as well as the final DOM: a correct end card can conceal duplicate fetches,
  delayed enrichment, or a stale response race. Screenshots and written grades must identify the
  actual selected cell, not only the intended coordinate.
- On short phones, test the combined controls sheet and route preview, not each in isolation; their
  stacking can hide map context, the close path, or the preview action.
- Keep source state and running-server state distinct in reports. A source edit is not a tested fix
  until the server serving the browser has been restarted on that build. Treat a lone favicon 404 as
  a low-noise browser artifact unless it affects product assets or interaction.
- Use deterministic, neutral public destinations and coordinates. They make tests replayable and
  avoid exposing a user's workplace or saved location.
- Successful checks that a shared K/L/M boarding corridor catalog was represented and that transfer
  instructions were explicit are useful examples of the contract. They are not line-name-specific
  specifications: future tests must assert the same generic catalog and interchange truth for any
  discovered service set.
- Server templates are loaded at boot. Restart the browser-serving process before treating a
  post-patch browser result as a verdict on the edited template.
- A bounded broad scan may exhaust its rate-limit allowance. Start its final verification in a fresh
  window rather than conflating the budget response with a route or UI failure.
- Exhaustive staged-disclosure comparison must deliberately open every lazy nested family before it
  compares API and DOM branch sets. Browser cache access is an LRU `Map`, so E2E reads must use
  `.get(...)`, not object-property access.
- For a late stale-fetch race, replay the deferred response without the already-aborted signal; the
  test must simulate a response that can still arrive after client cancellation, then prove it cannot
  win the UI.
- Canvas polygon taps need the actual polygon center and must wait for any pending map fit. An
  approximate coordinate can select a neighbor and turn a reliable test into a false diagnosis.
- In visual automation, a selector click that scrolls near the bottom of a Leaflet map can be
  intercepted by the attribution link even when the selector resolved correctly. Use the normal
  Playwright interaction test as the behavioral gate; for a screenshot-only disclosure replay,
  set the native `details.open` state explicitly and verify its DOM before capturing.
- Dense disclosure headings should be graded as a hierarchy, not just for text presence. A strong
  boarding-stop line plus muted direction/service metadata scans materially faster than one long
  sentence while preserving the same discovered structural truth.
- Keep route terminology exact: a pure walk-only result has zero transit rides, while a walk-finish
  route has one or more rides followed by walking. The final touch checks passed for Adjust followed
  by a new touch selection, and for the 390 px to 320 px mobile layout transition.
