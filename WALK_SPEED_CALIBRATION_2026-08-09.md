# Walk-speed calibration — 2026-08-09

The product uses fixed walking presets so a selected pace is deterministic across requests and
the router remains monotonic: slower walking cannot improve a walking leg. Google Maps walking
directions were used only as a calibration reference, never at runtime.

## Method

- Ten public San Francisco origin/destination pairs were queried in walking mode on 2026-08-09.
- The corpus intentionally includes short blocks, flat Mission/Embarcadero corridors, and hills in
  the Castro, Parnassus, Potrero, Glen Park, and Inner Sunset.
- Effective pace is `Google route distance / Google route duration`; it is a corpus check, not a
  claim that every individual will walk at that pace.

| Route | Distance | Duration | Effective pace | Category |
| --- | ---: | ---: | ---: | --- |
| 16th & Mission → Dolores Park | 959 m | 837 s | 4.12 km/h | short / cross-slope |
| Ferry Building → Embarcadero BART | 357 m | 327 s | 3.93 km/h | short / flat |
| 24th & Mission → 16th & Mission | 1,429 m | 1,139 s | 4.52 km/h | 1–2 km / flat |
| Castro Station → 16th & Mission | 1,474 m | 1,183 s | 4.49 km/h | 1–2 km / hill |
| Dolores Park → UCSF Parnassus | 3,138 m | 3,113 s | 3.63 km/h | 3 km / sustained hill |
| 20th & Wisconsin → 16th & Mission | 2,451 m | 2,134 s | 4.13 km/h | 2–3 km / Potrero hill |
| Inner Sunset → UCSF Parnassus | 719 m | 724 s | 3.58 km/h | short / hill |
| North Beach → Ferry Building | 1,526 m | 1,256 s | 4.37 km/h | 1–2 km / downhill |
| Glen Park Station → City College | 1,406 m | 1,261 s | 4.01 km/h | 1–2 km / hill |
| Japantown → Civic Center | 1,853 m | 1,590 s | 4.20 km/h | 1–2 km / downhill |

The observed range was 3.58–4.52 km/h, with a mean near 4.10 km/h. A separate ten-route spot
check over comparable SF corridors found a median near 4.25 km/h and hill values around 3.7–4.0.

## Selected product presets

| Preset | Pace | Rationale |
| --- | ---: | --- |
| Slow | 3.4 km/h | Deliberately conservative, below the hill-heavy reference observations. |
| Medium | 4.2 km/h | Typical SF walking pace, near the two bounded-corpus central estimates. |
| Fast | 5.2 km/h | A genuinely brisk pace, above the reference corpus but within the approved 5.1–5.4 band. |

The OSM graph remains baked at a **4.8 km/h reference speed**. At request time the engine applies
`4.8 / selected pace` to every access, transfer, egress, and pure-walk duration, so these product
speeds do not require a graph rebuild and preserve the ordering `slow time >= medium time >= fast
time`.

## Known routing-label limitation

Walking legs themselves are strictly monotonic. The full planned journey scan is ratcheted to one
named exception across 29,646 adjacent-speed comparisons: Bayview golden cell 2840 changes from
5 minutes at Slow to 6 at Medium. At Slow, a 23 transit label wins; at Medium, the reverse tree's
walk-only tail dominates and discards that still-feasible transit label before reconstruction.
The raw-time redesign reduced this from 13 exceptions to one and records the exact fixture in
`test_departafter_walkspeed_monotonicity`; any new exception fails the suite. Eliminating the last
one requires a Pareto reverse label that retains both walk-only and transit continuations, not a
speed fudge or a displayed-time clamp. Until that label model exists, agent QA must report this
single case explicitly rather than claiming perfect end-to-end monotonicity.
