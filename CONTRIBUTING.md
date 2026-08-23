# Contributing

Thanks for helping improve SF Commute Isochrone. This project maps modeled
door-to-door weekday commutes in San Francisco using public transit schedules
and a walking network. Contributions should make the model, UI, tests, or
documentation more accurate and easier to understand.

## Before opening a pull request

1. Search existing issues and pull requests.
2. Keep changes narrowly scoped. Explain any behavior or model change in the
   pull request description.
3. Do not commit downloaded `data/`, generated `out/`, `.venv/`, `.env`,
   caches, credentials, personal addresses, or private screenshots. Those are
   intentionally ignored.
4. Use public, non-sensitive locations in fixtures, screenshots, examples, and
   bug reports. A public landmark is preferable to a home or workplace.

## Development setup

Follow the [README](README.md) to create `.env`, install dependencies, and
download the local data needed for the full application. The data download
requires a 511 API token and is deliberately not part of repository CI.

The normal local checks are:

```bash
node --test tests/test_viz.mjs
.venv/bin/python -m pytest tests/ --ignore=tests/e2e -q
```

The browser suite needs a separately running local server and is documented in
[`tests/e2e/README.md`](tests/e2e/README.md). Avoid starting a test process
while another process is compiling Numba kernels against the same checkout.

## Design and routing changes

Changes that affect displayed times, route ranking, walking assumptions, or
route-family grouping need focused regression coverage. Describe the affected
public locations and the intended user-facing result. Do not encode a named
line as a special case when the behavior should follow from the transit
structure.

## Pull requests

Use the pull-request template. Keep commits readable, update documentation
when behavior or configuration changes, and call out incomplete verification.
By submitting a contribution, you agree that it may be licensed under the
[MIT License](LICENSE).
