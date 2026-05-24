#!/usr/bin/env python
"""
Live commute-isochrone server: set ANY workplace address (or click/drag the pin)
and the map recomputes door-to-door times from a grid of SF origins.

The R5 network is loaded ONCE at startup and kept warm, so each recompute is just
the travel-time-matrix step (no network rebuild). Run:

    .venv/bin/python scripts/server.py        # then open http://127.0.0.1:8000

Env: GRID_M (default 250) trades detail for speed.
"""
import os, io, json, datetime as dt, urllib.request, urllib.parse
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point, box
from flask import Flask, request, jsonify

from r5py import TransportNetwork, TravelTimeMatrix, DetailedItineraries, TransportMode
from destination import DEST_LAT, DEST_LON, DEST_LABEL  # configurable; see .env
from itineraries import load_routes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UTM = "EPSG:32610"
GRID_M = int(os.environ.get("GRID_M", "250"))
DEP = dt.datetime(2026, 5, 20, 8, 35)
WINDOW = dt.timedelta(minutes=int(os.environ.get("WINDOW_MIN", "30")))
DEFAULT = (DEST_LAT, DEST_LON, DEST_LABEL)
GTFS = [DATA / "muni_current.zip", DATA / "bart_gtfs.zip"]
ROUTES = load_routes(GTFS)

print(f"[boot] building grid @ {GRID_M}m + loading R5 network (once)...")
NEIGH = gpd.read_file(DATA / "sf_neighborhoods.geojson").to_crs("EPSG:4326")
_poly = NEIGH.to_crs(UTM).union_all()
_minx, _miny, _maxx, _maxy = _poly.bounds
_pts = [Point(x, y) for x in np.arange(_minx+GRID_M/2, _maxx, GRID_M)
        for y in np.arange(_miny+GRID_M/2, _maxy, GRID_M)]
_g = gpd.GeoDataFrame(geometry=_pts, crs=UTM)
_g = _g[_g.within(_poly)].reset_index(drop=True)
_g["id"] = _g.index.astype(str)
GRID = _g.to_crs("EPSG:4326")[["id", "geometry"]]
ORIGIN_LL = {r.id: (r.geometry.y, r.geometry.x) for r in GRID.itertuples()}

# cell squares (sent to browser once) + neighborhood label per cell
_half = GRID_M / 2
_sq = gpd.GeoSeries([box(p.x-_half, p.y-_half, p.x+_half, p.y+_half) for p in _g.geometry],
                    crs=UTM).to_crs("EPSG:4326")
_cells = gpd.GeoDataFrame({"id": _g["id"].values}, geometry=_sq.values, crs="EPSG:4326")
_cells = gpd.sjoin(_cells, NEIGH[["name", "geometry"]], how="left", predicate="intersects") \
            .drop_duplicates("id").drop(columns="index_right")
CELLS_GEOJSON = json.loads(_cells.to_json())
for f in CELLS_GEOJSON["features"]:
    f["properties"] = {"id": f["properties"]["id"], "n": f["properties"].get("name")}

NET = TransportNetwork(str(DATA / "osm_sf.pbf"),
                       [str(DATA / "muni_current.zip"), str(DATA / "bart_gtfs.zip")])
print(f"[boot] ready: {len(GRID)} origins. Open http://127.0.0.1:8000")

app = Flask(__name__)


def compute(lat, lon):
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(lon, lat)], crs="EPSG:4326")
    ttm = TravelTimeMatrix(
        NET, origins=GRID, destinations=dest, snap_to_network=True,
        departure=DEP, departure_time_window=WINDOW, max_time=dt.timedelta(minutes=75),
        transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
        percentiles=[5, 50], speed_walking=4.8)
    ttm = pd.DataFrame(ttm)
    cells = {}
    for _, r in ttm.iterrows():
        b, rl = r["travel_time_p5"], r["travel_time_p50"]
        cells[str(r["from_id"])] = [None if pd.isna(b) else int(b),
                                    None if pd.isna(rl) else int(rl)]
    return cells


def fastest_itin(olat, olon, dlat, dlon):
    origin = gpd.GeoDataFrame({"id": ["o"]}, geometry=[Point(olon, olat)], crs="EPSG:4326")
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(dlon, dlat)], crs="EPSG:4326")
    di = DetailedItineraries(NET, origins=origin, destinations=dest, departure=DEP,
                             transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
                             snap_to_network=True, force_all_to_all=True)
    df = pd.DataFrame(di)
    if df.empty:
        return None
    df["tt"] = pd.to_timedelta(df["travel_time"]).dt.total_seconds() / 60
    df["wt"] = pd.to_timedelta(df["wait_time"]).dt.total_seconds() / 60
    tot = df.groupby("option").apply(lambda x: x["tt"].sum() + x["wt"].sum(), include_groups=False)
    best = tot.idxmin()
    legs = df[df["option"] == best].sort_values("segment")
    out, rides = [], 0
    for _, l in legs.iterrows():
        mode = str(l["transport_mode"]).split(".")[-1]
        if mode == "WALK":
            if l["tt"] >= 1:
                out.append({"mode": "walk", "line": None, "min": round(l["tt"])})
        else:
            rides += 1
            rid = l["route_id"]
            try:
                k = str(int(float(rid)))
            except (ValueError, TypeError):
                k = str(rid)
            line = ROUTES.get(str(l.get("feed")), {}).get(k, k)
            out.append({"mode": mode.lower(), "line": line,
                        "min": round(l["tt"]), "wait": round(l["wt"])})
    return {"total": round(float(tot[best])), "xfers": max(0, rides - 1), "legs": out}


@app.route("/itinerary")
def _itinerary():
    cid = request.args.get("id")
    if cid is not None and cid in ORIGIN_LL:
        olat, olon = ORIGIN_LL[cid]
    else:
        olat, olon = float(request.args["olat"]), float(request.args["olon"])
    dlat, dlon = float(request.args["dlat"]), float(request.args["dlon"])
    return jsonify(fastest_itin(olat, olon, dlat, dlon) or {"error": "no route"})


@app.route("/compute")
def _compute():
    lat = float(request.args["lat"]); lon = float(request.args["lon"])
    t0 = dt.datetime.now()
    cells = compute(lat, lon)
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[compute] ({lat:.4f},{lon:.4f}) {ms:.0f}ms")
    return jsonify({"dest": [lat, lon], "cells": cells, "ms": round(ms)})


@app.route("/geocode")
def _geocode():
    q = request.args.get("q", "")
    if not q.strip():
        return jsonify({"error": "empty"}), 400
    if "san francisco" not in q.lower() and "sf" not in q.lower():
        q = q + ", San Francisco, CA"
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "limit": 1})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sf-commute-isochrone/1.0"})
        d = json.load(urllib.request.urlopen(req, timeout=12))
        if d:
            return jsonify({"lat": float(d[0]["lat"]), "lon": float(d[0]["lon"]),
                            "label": d[0].get("display_name", q)})
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"error": "not found"}), 404


@app.route("/")
def _index():
    return PAGE.replace("__CELLS__", json.dumps(CELLS_GEOJSON)) \
               .replace("__DEFAULT__", json.dumps(DEFAULT))


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SF Commute Explorer — set your own workplace</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root{--panel:#171a21;--ink:#e8eaed;--mut:#9aa3af;--line:#272b34;--accent:#5ab0ff}
  *{box-sizing:border-box} html,body{margin:0;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  #map{position:absolute;inset:0;background:#0b0d10}.leaflet-container{background:#0b0d10}
  #panel{position:absolute;top:0;right:0;width:330px;height:100%;background:var(--panel);color:var(--ink);
    z-index:1000;display:flex;flex-direction:column;border-left:1px solid var(--line)}
  h1{font-size:15px;margin:0;padding:14px 16px 3px}.sub{color:var(--mut);font-size:12px;padding:0 16px 10px;line-height:1.45}
  .ctl{padding:9px 16px;border-top:1px solid var(--line)}
  .lab{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin-bottom:7px}
  .addr{display:flex;gap:6px}.addr input{flex:1;background:#0e1116;border:1px solid var(--line);color:var(--ink);
    padding:8px;border-radius:8px;font-size:13px}.addr button{background:var(--accent);color:#06121f;border:0;
    padding:0 12px;border-radius:8px;font-weight:600;cursor:pointer}
  .seg{display:flex;gap:6px}.seg button{flex:1;background:#11141a;color:var(--ink);border:1px solid var(--line);
    padding:7px;border-radius:8px;font-size:12.5px;cursor:pointer}.seg button.on{background:var(--accent);color:#06121f;font-weight:600;border-color:var(--accent)}
  input[type=range]{width:100%;accent-color:var(--accent)} .val{color:var(--accent);font-weight:600}
  #list{flex:1;overflow:auto;border-top:1px solid var(--line)}
  .nb{display:flex;justify-content:space-between;padding:7px 16px;font-size:13px;cursor:pointer;border-bottom:1px solid #1c2027}
  .nb:hover{background:#1f2530}.nb .t{font-variant-numeric:tabular-nums;font-weight:600}.nb small{color:var(--mut)}
  #legend{position:absolute;left:14px;bottom:16px;z-index:1000;background:rgba(20,23,29,.92);color:var(--ink);
    padding:11px 13px;border:1px solid var(--line);border-radius:10px;font-size:12px}
  #legend .bar{height:11px;width:240px;border-radius:3px;margin:7px 0 4px}
  #legend .sc{display:flex;justify-content:space-between;color:var(--mut);font-size:11px}
  #busy{position:absolute;top:12px;left:12px;z-index:1100;background:var(--accent);color:#06121f;font-weight:600;
    padding:7px 12px;border-radius:20px;font-size:12.5px;display:none;box-shadow:0 2px 10px rgba(0,0,0,.4)}
  .leaflet-tooltip.tt{background:#11141a;border:1px solid var(--line);color:var(--ink);font-size:12px}
  .credit{padding:9px 16px;color:#5c6470;font-size:10.5px;border-top:1px solid var(--line)}
</style></head>
<body>
<div id="map"></div><div id="busy">computing…</div>
<div id="panel">
  <h1>Commute to your workplace</h1>
  <div class="sub">Type an address, <b>click the map</b>, or <b>drag the pin</b> — the map recomputes
    walk+Muni+BART door-to-door times (~9am) from across SF.</div>
  <div class="ctl"><div class="lab">Workplace</div>
    <div class="addr"><input id="addr" placeholder="e.g. 1 Market St"><button id="go">Set</button></div>
    <div id="dest" style="color:var(--mut);font-size:11px;margin-top:6px"></div></div>
  <div class="ctl"><div class="lab">Time estimate</div>
    <div class="seg" id="metric"><button data-v="r" class="on">Realistic</button><button data-v="b">Best-case</button></div></div>
  <div class="ctl"><div class="lab">Sweet spot — green ≤ <span class="val" id="idval">25</span> min (recolors)</div>
    <input type="range" id="ideal" min="10" max="45" value="25"></div>
  <div class="ctl"><div class="lab">Max commute — hide above <span class="val" id="thrval">40</span> min (filters)</div>
    <input type="range" id="thr" min="10" max="70" value="40"></div>
  <div class="ctl" style="padding-bottom:5px"><div class="lab">Neighborhoods</div></div>
  <div id="list"></div>
  <div class="credit">R5/r5py · Muni (511) + BART · network kept warm; only the matrix recomputes.</div>
</div>
<div id="legend"><b id="legtitle">Realistic door-to-door (min)</b><div class="bar"></div>
  <div class="sc"></div></div>
<script>
const CELLS=__CELLS__, DEFAULT=__DEFAULT__;
let metric="r", ideal=25, thr=40, TT={}, NB={}, DESTLL=DEFAULT.slice(0,2);
const map=L.map("map").setView([37.762,-122.43],12.3);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",{maxZoom:19}).addTo(map);
const rgb=c=>`rgb(${c[0]},${c[1]},${c[2]})`;
function ramp(){const hi=ideal+25;return {hi,S:[[0,[0,104,55]],[ideal*.45,[26,152,80]],[ideal*.72,[102,189,99]],
  [ideal*.9,[166,217,106]],[ideal,[255,255,191]],[ideal+(hi-ideal)*.35,[253,174,97]],
  [ideal+(hi-ideal)*.7,[244,109,67]],[hi,[215,48,39]]]};}
function color(v){if(v==null)return null;const {S}=ramp();if(v<=0)return rgb(S[0][1]);
  for(let i=1;i<S.length;i++){if(v<=S[i][0]){const a=S[i-1],b=S[i],t=(v-a[0])/((b[0]-a[0])||1);
    return rgb(a[1].map((c,j)=>Math.round(c+(b[1][j]-c)*t)));}}return rgb(S.at(-1)[1]);}
function legend(){const {hi,S}=ramp();
  document.querySelector("#legend .bar").style.background="linear-gradient(90deg,"+
    S.map(s=>rgb(s[1])+" "+Math.round(s[0]/hi*100)+"%").join(",")+")";
  document.querySelector("#legend .sc").innerHTML=`<span>0</span><span>${ideal} (ideal)</span><span>${hi}+</span>`;
  document.getElementById("legtitle").textContent=(metric=="r"?"Realistic":"Best-case")+" door-to-door (min)";}
function val(id){const v=TT[id];return v?(metric=="r"?v[1]:v[0]):null;}
function style(f){const v=val(f.properties.id);if(v==null||v>thr)return{fillOpacity:0,opacity:0,weight:0};
  return{fillColor:color(v),fillOpacity:.72,weight:0};}
const layer=L.geoJSON(CELLS,{style,onEachFeature:(f,l)=>{
  l.on("mouseover",()=>{const v=val(f.properties.id);
    l.bindTooltip(`${f.properties.n||"—"}<br><b>${v==null?"—":v+" min"}</b><br><span style="color:#9aa3af">click for route</span>`,{className:"tt",sticky:true}).openTooltip();});
  l.on("click",async()=>{const v=val(f.properties.id);if(v==null)return;
    l.bindPopup("computing route…").openPopup();
    try{const r=await fetch(`/itinerary?id=${f.properties.id}&dlat=${DESTLL[0]}&dlon=${DESTLL[1]}`);const d=await r.json();
      if(d.error){l.setPopupContent("no transit route found");return;}
      const chain=d.legs.map(g=>g.line?`<b style="color:#5ab0ff">${g.line}</b> ${g.min}m`:`walk ${g.min}m`).join(" → ");
      l.setPopupContent(`<div style="font:12.5px sans-serif;max-width:235px"><b>${f.properties.n||""}</b> · ${d.total} min · ${d.xfers} transfer${d.xfers==1?"":"s"}<div style="margin-top:5px;line-height:1.65">${chain}</div><div style="color:#9aa3af;margin-top:5px;font-size:11px">typical fastest trip (~8:35am)</div></div>`);
    }catch(e){l.setPopupContent("error");}});
}}).addTo(map);
const pin=L.marker(DEFAULT.slice(0,2),{draggable:true}).addTo(map);
pin.on("dragend",()=>{const p=pin.getLatLng();run(p.lat,p.lng,"(dropped pin)");});
map.on("click",e=>{pin.setLatLng(e.latlng);run(e.latlng.lat,e.latlng.lng,"(map click)");});

function redraw(){layer.setStyle(style);legend();renderList();}
function renderList(){const rows=Object.entries(NB).filter(([k,v])=>v!=null).sort((a,b)=>a[1]-b[1]);
  document.getElementById("list").innerHTML=rows.map(([n,v])=>{const c=v<=thr?color(v):"#555";
    return `<div class="nb"><span><span style="color:${c}">●</span> ${n}</span><span class="t" style="color:${c}">${v}<small> min</small></span></div>`;}).join("");}
function aggregate(){const acc={};CELLS.features.forEach(f=>{const v=val(f.properties.id),n=f.properties.n;
  if(v==null||!n)return;(acc[n]=acc[n]||[]).push(v);});NB={};
  for(const n in acc){acc[n].sort((a,b)=>a-b);NB[n]=acc[n][Math.floor(acc[n].length/2)];}}
async function run(lat,lon,label){document.getElementById("busy").style.display="block";
  try{const r=await fetch(`/compute?lat=${lat}&lon=${lon}`);const d=await r.json();
    TT=d.cells;DESTLL=d.dest;document.getElementById("dest").textContent=(label||"")+`  ·  ${d.ms}ms`;
    aggregate();redraw();}catch(e){alert("compute failed: "+e);}finally{document.getElementById("busy").style.display="none";}}
function seg(id,set){document.querySelectorAll(`#${id} button`).forEach(b=>b.onclick=()=>{
  document.querySelectorAll(`#${id} button`).forEach(x=>x.classList.remove("on"));b.classList.add("on");set(b.dataset.v);aggregate();redraw();});}
seg("metric",v=>metric=v);
document.getElementById("ideal").oninput=e=>{ideal=+e.target.value;document.getElementById("idval").textContent=ideal;redraw();};
document.getElementById("thr").oninput=e=>{thr=+e.target.value;document.getElementById("thrval").textContent=thr;redraw();};
document.getElementById("go").onclick=async()=>{const q=document.getElementById("addr").value;
  if(!q)return;document.getElementById("busy").style.display="block";
  try{const r=await fetch(`/geocode?q=${encodeURIComponent(q)}`);const d=await r.json();
    if(d.lat){pin.setLatLng([d.lat,d.lon]);map.setView([d.lat,d.lon],13);run(d.lat,d.lon,d.label.split(",").slice(0,2).join(","));}
    else alert("address not found");}catch(e){alert(e);}finally{document.getElementById("busy").style.display="none";}};
document.getElementById("addr").addEventListener("keydown",e=>{if(e.key=="Enter")document.getElementById("go").click();});
run(DEFAULT[0],DEFAULT[1],DEFAULT[2]);   // initial
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), threaded=False)
