#!/usr/bin/env python
"""
Which transit line carries each part of SF to the workplace?

For a medium grid, compute the fastest door-to-door itinerary (~8:35am) and record
the PRIMARY line (the transit leg with the most in-vehicle time) plus the full route
chain. Emit out/route_explorer.html:
  - cells colored by primary line (categorical)
  - click a cell -> full route + time
  - click a line in the legend -> show only that line's "commute-shed"
"""
import sys, json, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point, box
from r5py import TransportNetwork, DetailedItineraries, TransportMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itineraries import load_routes

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "out"
UTM = "EPSG:32610"
GRID_M = 400   # granular; DetailedItineraries per cell is heavy, so this run is slow (one-time)
from destination import DEST_LAT, DEST_LON, DEST_LABEL  # configurable; see .env


def build_grid(neigh):
    poly = neigh.to_crs(UTM).union_all()
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + GRID_M / 2, maxx, GRID_M)
    ys = np.arange(miny + GRID_M / 2, maxy, GRID_M)
    pts = [Point(x, y) for x in xs for y in ys]
    g = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    g = g[g.within(poly)].reset_index(drop=True)
    g["id"] = g.index.astype(str)
    return g.to_crs("EPSG:4326")


def route_shapes(gtfs_paths):
    """One representative (longest) shape per route, classified by mode, for an overlay."""
    import zipfile, io
    MODE = {"0": "metro", "1": "bart", "2": "bart", "5": "cable", "3": "bus"}
    feats = []
    for p in gtfs_paths:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            if "shapes.txt" not in names:
                continue
            shapes = pd.read_csv(io.BytesIO(z.read("shapes.txt")), dtype={"shape_id": str})
            trips = pd.read_csv(io.BytesIO(z.read("trips.txt")), dtype=str).dropna(subset=["shape_id"])
            routes = pd.read_csv(io.BytesIO(z.read("routes.txt")), dtype=str)
            rtype = dict(zip(routes.route_id, routes.route_type))
            rname = dict(zip(routes.route_id,
                             routes.get("route_short_name", routes["route_id"]).fillna(routes["route_id"])))
            cnt = shapes.groupby("shape_id").size()
            best = {}
            for sid, rid in zip(trips.shape_id, trips.route_id):
                if rid not in best or cnt.get(sid, 0) > cnt.get(best[rid], 0):
                    best[rid] = sid
            for rid, sid in best.items():
                pts = shapes[shapes.shape_id == sid].sort_values("shape_pt_sequence")
                coords = pts[["shape_pt_lon", "shape_pt_lat"]].astype(float).values.tolist()
                if len(coords) < 2:
                    continue
                feats.append({"type": "Feature",
                              "geometry": {"type": "LineString", "coordinates": coords},
                              "properties": {"mode": MODE.get(str(rtype.get(rid, "3")), "bus"),
                                             "name": str(rname.get(rid, rid))}})
    return {"type": "FeatureCollection", "features": feats}


def main():
    gtfs = [DATA / "muni_current.zip", DATA / "bart_gtfs.zip"]
    routes = load_routes(gtfs)
    neigh = gpd.read_file(DATA / "sf_neighborhoods.geojson").to_crs("EPSG:4326")
    g = build_grid(neigh)
    print(f"grid: {len(g)} cells @ {GRID_M}m")
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(DEST_LON, DEST_LAT)], crs="EPSG:4326")

    net = TransportNetwork(str(DATA / "osm_sf.pbf"), [str(p) for p in gtfs])
    di = DetailedItineraries(
        net, origins=g, destinations=dest,
        departure=dt.datetime(2026, 5, 20, 8, 35),
        transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
        snap_to_network=True, force_all_to_all=True)
    df = pd.DataFrame(di)
    df["tt"] = pd.to_timedelta(df["travel_time"]).dt.total_seconds() / 60
    df["wt"] = pd.to_timedelta(df["wait_time"]).dt.total_seconds() / 60

    def rn(rid, feed):
        if pd.isna(rid):
            return None
        try:
            k = str(int(float(rid)))
        except (ValueError, TypeError):
            k = str(rid)
        return routes.get(str(feed), {}).get(k, k)

    recs = {}
    for oid, gg in df.groupby("from_id"):
        tot = gg.groupby("option").apply(lambda x: x["tt"].sum() + x["wt"].sum(), include_groups=False)
        best = tot.idxmin()
        legs = gg[gg["option"] == best].sort_values("segment")
        chain, primary, pmax = [], None, -1
        for _, l in legs.iterrows():
            mode = str(l["transport_mode"]).split(".")[-1]
            if mode == "WALK":
                if l["tt"] >= 1:
                    chain.append(f"walk {l['tt']:.0f}m")
            else:
                nm = rn(l["route_id"], l.get("feed")) or mode
                chain.append(f"{nm} {l['tt']:.0f}m")
                if l["tt"] > pmax:
                    pmax, primary = l["tt"], nm
        recs[oid] = {"t": round(float(tot[best])), "chain": " → ".join(chain),
                     "prim": primary or "walk only"}

    # cells
    gm = g.to_crs(UTM)
    half = GRID_M / 2
    cells = g.copy()
    cells["olat"] = g.geometry.y.values
    cells["olon"] = g.geometry.x.values
    cells["geometry"] = gpd.GeoSeries(
        [box(p.x-half, p.y-half, p.x+half, p.y+half) for p in gm.geometry],
        crs=UTM).to_crs("EPSG:4326").values

    feats = []
    for _, r in cells.iterrows():
        rec = recs.get(r["id"])
        if not rec or rec["t"] is None or rec["t"] > 60:
            continue
        feats.append({
            "type": "Feature",
            "geometry": json.loads(gpd.GeoSeries([r["geometry"]]).to_json())["features"][0]["geometry"],
            "properties": {"t": rec["t"], "chain": rec["chain"], "prim": rec["prim"],
                           "olat": round(r["olat"], 5), "olon": round(r["olon"], 5)}})
    fc = {"type": "FeatureCollection", "features": feats}
    lines = route_shapes(gtfs)
    print(f"overlay: {len(lines['features'])} route shapes")

    # line frequency for legend ordering
    from collections import Counter
    cnt = Counter(f["properties"]["prim"] for f in feats)
    order = [k for k, _ in cnt.most_common()]
    print("primary lines:", dict(cnt.most_common(15)))

    html = TEMPLATE.replace("__DATA__", json.dumps(fc)) \
                   .replace("__ORDER__", json.dumps(order)) \
                   .replace("__LINES__", json.dumps(lines)) \
                   .replace("__DEST__", json.dumps([DEST_LAT, DEST_LON])) \
                   .replace("__LABEL__", DEST_LABEL)
    (OUT / "route_explorer.html").write_text(html)
    print(f"Wrote out/route_explorer.html  ({len(feats)} cells, {len(order)} distinct primary lines)")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Which line carries your commute → __LABEL__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--ink:#e8eaed;--mut:#9aa3af;--line:#272b34;--accent:#5ab0ff}
  *{box-sizing:border-box} html,body{margin:0;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  #map{position:absolute;inset:0;background:#0b0d10}
  .leaflet-container{background:#0b0d10}
  #panel{position:absolute;top:0;right:0;width:320px;height:100%;background:var(--panel);
    color:var(--ink);z-index:1000;display:flex;flex-direction:column;border-left:1px solid var(--line)}
  #panel h1{font-size:15px;margin:0;padding:15px 16px 3px}
  #panel .sub{color:var(--mut);font-size:12px;padding:0 16px 10px;line-height:1.45}
  #legend{flex:1;overflow:auto;border-top:1px solid var(--line)}
  .row{display:flex;align-items:center;gap:9px;padding:6px 16px;font-size:13px;cursor:pointer;border-bottom:1px solid #1c2027}
  .row:hover{background:#1f2530}
  .row.off{opacity:.32}
  .sw{width:14px;height:14px;border-radius:3px;flex:none}
  .row .nm{flex:1;font-weight:600} .row .ct{color:var(--mut);font-variant-numeric:tabular-nums}
  .bar{padding:9px 16px;border-top:1px solid var(--line);display:flex;gap:6px}
  .bar button{flex:1;background:#11141a;color:var(--ink);border:1px solid var(--line);padding:6px;border-radius:7px;font-size:12px;cursor:pointer}
  .bar button:hover{border-color:var(--accent)}
  .leaflet-popup-content-wrapper{background:#11141a;color:var(--ink);border:1px solid var(--line)}
  .leaflet-popup-tip{background:#11141a}
  .pop b{color:var(--accent)} .pop .ch{font-size:12.5px;line-height:1.5;margin-top:4px}
  .credit{padding:9px 16px;color:#5c6470;font-size:10.5px;border-top:1px solid var(--line)}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#ffd23f;border:1px solid #000;margin-right:4px}
</style></head>
<body>
<div id="map"></div>
<div id="panel">
  <h1>Which line carries you to __LABEL__?</h1>
  <div class="sub">Each cell is colored by the <b>primary line</b> in its fastest ~8:35am trip
    (the leg with the most ride time). <b>Click a cell</b> for the full route;
    <b>click a line</b> below to isolate its commute-shed. <span class="dot"></span>= __LABEL__.</div>
  <div class="bar"><button id="all">Show all</button><button id="none">Hide all</button></div>
  <div class="ovl" style="padding:9px 16px;border-top:1px solid var(--line);font-size:12.5px;display:flex;flex-wrap:wrap;gap:10px;align-items:center">
    <span style="color:var(--mut);width:100%;font-size:11px;text-transform:uppercase;letter-spacing:.06em">Overlay real lines</span>
    <label style="cursor:pointer"><input type="checkbox" data-m="bart"> <span style="color:#4363d8">BART</span></label>
    <label style="cursor:pointer"><input type="checkbox" data-m="metro"> <span style="color:#e6194B">Metro</span></label>
    <label style="cursor:pointer"><input type="checkbox" data-m="bus"> <span style="color:#f58231">Bus</span></label>
    <label style="cursor:pointer"><input type="checkbox" data-m="cable"> <span style="color:#3cb44b">Cable</span></label>
  </div>
  <div id="legend"></div>
  <div class="credit">r5py / R5 · Muni (511, current) + BART · fastest single 8:35am itinerary.
    "walk only" = faster on foot than any transit.</div>
</div>
<script>
const CELLS=__DATA__, ORDER=__ORDER__, LINES=__LINES__, DEST=__DEST__;
const gmaps=(olat,olon)=>`https://www.google.com/maps/dir/?api=1&origin=${olat},${olon}&destination=${DEST[0]},${DEST[1]}&travelmode=transit`;
const PALETTE=["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4",
 "#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#fffac8","#800000",
 "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9"];
const WALK="#7fd1ff";
const colorOf={}; ORDER.forEach((l,i)=>{colorOf[l]= l=="walk only"?WALK:PALETTE[i%PALETTE.length];});
const on={}; ORDER.forEach(l=>on[l]=true);

const map=L.map("map").setView([37.762,-122.43],12.3);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
 {maxZoom:19,attribution:"© OpenStreetMap, © CARTO"}).addTo(map);

function style(f){const p=f.properties;
  if(!on[p.prim]) return {fillOpacity:0,opacity:0,weight:0};
  return {fillColor:colorOf[p.prim]||"#999",fillOpacity:.74,color:"#0b0d10",weight:.3};}
const layer=L.geoJSON(CELLS,{style,onEachFeature:(f,l)=>{
  const p=f.properties;
  l.bindPopup(`<div class="pop"><b>${p.prim}</b> · ${p.t} min<div class="ch">${p.chain}</div>`+
    `<a href="${gmaps(p.olat,p.olon)}" target="_blank" style="color:#5ab0ff;display:inline-block;margin-top:6px">Open in Google Maps ↗</a></div>`);
  l.on("mouseover",()=>l.setStyle({weight:1.5,color:"#fff"}));
  l.on("mouseout",()=>layer.resetStyle(l));
}}).addTo(map);

// transit-line overlays (off by default; non-interactive so cell clicks pass through)
const LINESTYLE={bus:{color:"#f58231",weight:1.4,opacity:.55},metro:{color:"#e6194B",weight:2.6,opacity:.85},
  cable:{color:"#3cb44b",weight:2.2,opacity:.8},bart:{color:"#4363d8",weight:3.2,opacity:.85}};
const overlays={};
["bart","metro","bus","cable"].forEach(m=>{overlays[m]=L.geoJSON(
  {type:"FeatureCollection",features:LINES.features.filter(f=>f.properties.mode===m)},
  {style:()=>LINESTYLE[m],interactive:false});});
document.querySelectorAll(".ovl input").forEach(cb=>cb.onchange=()=>{
  const m=cb.dataset.m; if(cb.checked) overlays[m].addTo(map); else map.removeLayer(overlays[m]);});

L.marker(DEST,{icon:L.divIcon({className:"",html:'<div style="width:14px;height:14px;border-radius:50%;background:#ffd23f;border:2px solid #000;box-shadow:0 0 6px #000"></div>',iconSize:[14,14]})}).addTo(map);

const counts={}; CELLS.features.forEach(f=>counts[f.properties.prim]=(counts[f.properties.prim]||0)+1);
const leg=document.getElementById("legend");
function renderLegend(){leg.innerHTML=ORDER.map(l=>
  `<div class="row ${on[l]?"":"off"}" data-l="${l}">
     <span class="sw" style="background:${colorOf[l]}"></span>
     <span class="nm">${l}</span><span class="ct">${counts[l]} cells</span></div>`).join("");
  leg.querySelectorAll(".row").forEach(el=>el.onclick=()=>{
    const l=el.dataset.l;
    const onlyMe=Object.keys(on).every(k=>k==l?on[k]:!on[k]);
    if(onlyMe){ORDER.forEach(k=>on[k]=true);}        // toggle back to all
    else {ORDER.forEach(k=>on[k]=(k==l));}            // isolate this line
    layer.setStyle(style);renderLegend();});}
document.getElementById("all").onclick=()=>{ORDER.forEach(k=>on[k]=true);layer.setStyle(style);renderLegend();};
document.getElementById("none").onclick=()=>{ORDER.forEach(k=>on[k]=false);layer.setStyle(style);renderLegend();};
renderLegend();
</script></body></html>"""


if __name__ == "__main__":
    main()
