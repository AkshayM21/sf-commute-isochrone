"""R5 ground-truth for the Phase-2 path layer: per-cell dominant line + sample breakdowns.

Boots R5 once (via the server module's recorded-path machinery — the existing _build_itin /
_dominant_line) and, for each validation workplace, records:
  - dominant[cellId]  = R5's color-by-line dominant route (the /attribution oracle)
  - sample breakdowns for a handful of cells (the /itinerary oracle)
into tests/raptor_golden/path_<name>.json. The Phase-2 RAPTOR path layer is validated against
these (JVM-free) by scripts/raptor_validate_paths.py.

This is the only JVM step for Phase-2 validation; rerun on a GTFS repull.
Usage: R5_MAX_MEMORY=4G EXACT_THREADS=6 .venv/bin/python scripts/raptor_path_oracle.py
"""
import os, sys, json
from pathlib import Path

_mem = os.environ.get("R5_MAX_MEMORY")
if _mem and "--max-memory" not in sys.argv:
    sys.argv += ["--max-memory", _mem]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
GOLDEN = ROOT / "tests" / "raptor_golden"
WORKPLACES = {
    "downtown": (37.7942, -122.3950), "sunset": (37.7558, -122.4942),
    "bayview": (37.7299, -122.3890), "westportal": (37.7405, -122.4663),
    "caltrain": (37.7766, -122.3933),
}
SAMPLE_CELLS = 40            # breakdowns to snapshot per workplace


def main():
    import server  # boots R5 once (USE_RAPTOR unset -> R5 path)
    gen = server._current_generation()
    for name, (lat, lon) in WORKPLACES.items():
        itins = server.prewarm_itineraries(lat, lon, gen)        # {cellId: itin} via R5 recorded paths
        dominant = {cid: server._dominant_line(it) for cid, it in itins.items()}
        ids = sorted(itins.keys(), key=lambda c: int(c))
        sample = {cid: itins[cid] for cid in ids[:: max(1, len(ids) // SAMPLE_CELLS)][:SAMPLE_CELLS]}
        out = GOLDEN / f"path_{name}.json"
        out.write_text(json.dumps({
            "name": name, "dest": [lat, lon], "service_date": str(server._SVC_DATE),
            "dominant": dominant, "sample_itins": sample,
        }, separators=(",", ":")))
        print(f"[path oracle {name}] {len(dominant)} cells, {len(sample)} sample itins -> {out.name}",
              flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
