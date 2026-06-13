#!/usr/bin/env python
"""
Build a standalone interactive HTML map (out/commute_explorer.html) from the
full (Muni+BART+Caltrain) and Muni-only travel-time grids.

Controls in the page:
  - Metric:   Realistic (median, incl. wait)  vs  Best-case  ("conservative on/off")
  - Transit:  Muni + BART  vs  Muni only      (BART on/off)
  - Slider:   max commute minutes (hide cells above)
  - Neighborhood panel: ranked list; click to zoom
"""
import json
from pathlib import Path
import pandas as pd
import geopandas as gpd

from core import config, grid
from destination import DEST_LAT, DEST_LON, DEST_LABEL  # configurable; see .env


def load(tag):
    g = gpd.read_file(config.OUT / f"grid_traveltimes_{tag}.gpkg")[["id", "best_min", "real_min", "geometry"]]
    return g.rename(columns={"best_min": f"best_{tag}", "real_min": f"real_{tag}"})


def main():
    full = load("full")
    muni = load("munionly")[["id", "best_munionly", "real_munionly"]]
    g = full.merge(muni, on="id", how="left")

    neigh = grid.load_neighborhoods()
    g = gpd.sjoin(g, neigh[["name", "geometry"]], how="left", predicate="within").drop(columns="index_right")
    # a cell on a neighborhood border matches twice in the within-join -> duplicate features
    # (mirrors grid.attach_neighborhoods' drop_duplicates guard)
    g = g.drop_duplicates(subset="id")

    # square cells (200m) from the point grid
    cells = g.copy()
    cells["geometry"] = grid.square_cells(g.geometry, config.GRID_M).values

    def clean(v):
        return None if pd.isna(v) else round(float(v), 1)

    feats = []
    for _, r in cells.iterrows():
        if pd.isna(r["real_full"]) and pd.isna(r.get("real_munionly")):
            continue
        feats.append({
            "type": "Feature",
            "geometry": json.loads(gpd.GeoSeries([r["geometry"]]).to_json())["features"][0]["geometry"],
            "properties": {
                "n": (None if pd.isna(r["name"]) else r["name"]),
                "rf": clean(r["real_full"]), "bf": clean(r["best_full"]),
                "rm": clean(r.get("real_munionly")), "bm": clean(r.get("best_munionly")),
            }})
    fc = {"type": "FeatureCollection", "features": feats}

    # neighborhood aggregates (median per metric) + centroid
    rows = []
    for name, sub in g.groupby("name"):
        c = neigh[neigh["name"] == name]
        if not len(c):
            continue
        cen = c.to_crs(config.UTM).geometry.centroid.to_crs(config.WGS).iloc[0]
        rows.append({
            "name": name, "lat": round(cen.y, 5), "lon": round(cen.x, 5),
            "rf": clean(sub["real_full"].median()), "bf": clean(sub["best_full"].median()),
            "rm": clean(sub["real_munionly"].median()), "bm": clean(sub["best_munionly"].median()),
        })
    nb = [r for r in rows if r["rf"] is not None]

    # Inline the shared viz.js FIRST (so the page is self-contained and its time->color ramp /
    # gmaps link can't drift from the live server's), then substitute the data tokens.
    viz_text = (Path(__file__).resolve().parent / "assets" / "viz.js").read_text()
    html = TEMPLATE.replace("/*__VIZ__*/", viz_text) \
                   .replace("__DATA__", json.dumps(fc)) \
                   .replace("__NB__", json.dumps(nb)) \
                   .replace("__DEST__", json.dumps([DEST_LAT, DEST_LON])) \
                   .replace("__LABEL__", DEST_LABEL)
    (config.OUT / "commute_explorer.html").write_text(html)
    print(f"Wrote out/commute_explorer.html  ({len(feats)} cells, {len(nb)} neighborhoods)")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SF Commute Explorer → __LABEL__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--ink:#e8eaed;--mut:#9aa3af;--line:#272b34;--accent:#5ab0ff}
  *{box-sizing:border-box} html,body{margin:0;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  #map{position:absolute;inset:0;background:#0b0d10}
  .leaflet-container{background:#0b0d10}
  #panel{position:absolute;top:0;right:0;width:330px;height:100%;background:var(--panel);
    color:var(--ink);z-index:1000;display:flex;flex-direction:column;border-left:1px solid var(--line);box-shadow:-8px 0 24px rgba(0,0,0,.35)}
  #panel h1{font-size:15px;margin:0;padding:16px 16px 4px}
  #panel .sub{color:var(--mut);font-size:12px;padding:0 16px 12px;line-height:1.45}
  .ctl{padding:10px 16px;border-top:1px solid var(--line)}
  .ctl .lab{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin-bottom:7px}
  .seg{display:flex;gap:6px}
  .seg button{flex:1;background:#11141a;color:var(--ink);border:1px solid var(--line);
    padding:7px 8px;border-radius:8px;font-size:12.5px;cursor:pointer;transition:.12s}
  .seg button.on{background:var(--accent);color:#06121f;border-color:var(--accent);font-weight:600}
  .seg button:hover{border-color:var(--accent)}
  input[type=range]{width:100%;accent-color:var(--accent)}
  #thrval{color:var(--accent);font-weight:600}
  #list{flex:1;overflow:auto;border-top:1px solid var(--line)}
  .nb{display:flex;justify-content:space-between;gap:8px;padding:7px 16px;font-size:13px;
    cursor:pointer;border-bottom:1px solid #1c2027}
  .nb:hover{background:#1f2530} .nb .t{font-variant-numeric:tabular-nums;font-weight:600}
  .nb small{color:var(--mut)}
  #legend{position:absolute;left:14px;bottom:16px;z-index:1000;background:rgba(20,23,29,.92);
    color:var(--ink);padding:11px 13px;border:1px solid var(--line);border-radius:10px;font-size:12px}
  #legend .bar{height:11px;width:240px;border-radius:3px;margin:7px 0 4px;
    background:linear-gradient(90deg,#006837 0%,#1a9850 16%,#66bd63 28%,#a6d96a 40%,#d9ef8b 50%,#ffffbf 60%,#fdae61 70%,#f46d43 84%,#d73027 100%)}
  #legend .sc{display:flex;justify-content:space-between;color:var(--mut);font-size:11px}
  .leaflet-tooltip.tt{background:#11141a;border:1px solid var(--line);color:var(--ink);font-size:12px}
  .credit{padding:9px 16px;color:#5c6470;font-size:10.5px;border-top:1px solid var(--line)}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#ffd23f;border:1px solid #000;margin-right:5px}
</style></head>
<body>
<div id="map"></div>
<div id="legend">
  <b id="legtitle">Realistic door-to-door (min)</b>
  <div class="bar"></div>
  <div class="sc"><span>0</span><span>10</span><span>20</span><span>30</span><span>40</span><span>50+</span></div>
  <div style="margin-top:7px"><span class="dot"></span>__LABEL__</div>
</div>
<div id="panel">
  <h1>Commute → __LABEL__</h1>
  <div class="sub">Door-to-door by walking + transit, leaving ~8am on a weekday.
     Hover the map; click a neighborhood to zoom.</div>
  <div class="ctl"><div class="lab">Time estimate</div>
    <div class="seg" id="metric">
      <button data-v="r" class="on">Realistic (incl. wait)</button>
      <button data-v="b">Best-case</button></div></div>
  <div class="ctl"><div class="lab">Transit modes</div>
    <div class="seg" id="scenario">
      <button data-v="f" class="on">Muni + BART + Caltrain</button>
      <button data-v="m">Muni only</button></div></div>
  <div class="ctl"><div class="lab">Sweet spot — green ≤ <span id="idval">25</span> min <span style="color:#5c6470;text-transform:none;letter-spacing:0">(recolors only)</span></div>
    <input type="range" id="ideal" min="10" max="45" value="25" step="1"></div>
  <div class="ctl"><div class="lab">Max commute — hide above <span id="thrval">35</span> min <span style="color:#5c6470;text-transform:none;letter-spacing:0">(filters)</span></div>
    <input type="range" id="thr" min="10" max="60" value="35" step="1"></div>
  <div class="ctl" style="padding-bottom:6px"><div class="lab">Neighborhoods (by current estimate)</div></div>
  <div id="list"></div>
  <div class="credit">Engine: Conveyal R5 (r5py) · Muni (511.org) + BART + Caltrain · OSM walk network.
    Best-case = 5th pct over the ~8:35am departure window; realistic = median.</div>
</div>
<script>
/*__VIZ__*/
const CELLS=__DATA__, NB=__NB__, DEST=__DEST__;
const gmaps=(olat,olon)=>gmapsURL(olat,olon,DEST[0],DEST[1]);
let metric="r", scen="f", thr=35, ideal=25;
let SCALE=ramp(ideal).S;                     // current color scale (recomputed once per redraw)
const color=v=>colorScale(v,SCALE);
const key=()=>metric+scen; // rf, bf, rm, bm

const map=L.map("map",{zoomControl:true}).setView([37.773,-122.42],12.4);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  {maxZoom:19,attribution:"© OpenStreetMap, © CARTO"}).addTo(map);

function legendGradient(){
  const {hi,S}=ramp(ideal);
  const css=S.map(s=>`rgb(${s[1].join(",")}) ${Math.round(s[0]/hi*100)}%`).join(",");
  document.querySelector("#legend .bar").style.background=`linear-gradient(90deg,${css})`;
  document.querySelector("#legend .sc").innerHTML=
    `<span>0</span><span>${ideal} (ideal)</span><span>${hi}+</span>`;
}

function val(p){return p[key()];}
function style(f){
  const v=val(f.properties);
  if(v==null||v>thr) return {fillOpacity:0,opacity:0,weight:0};
  const c=color(v);
  return {fillColor:c,fillOpacity:.72,color:c,weight:0};
}
const layer=L.geoJSON(CELLS,{style,
  onEachFeature:(f,l)=>{l.on("mouseover",()=>{const p=f.properties,v=val(p);
    l.bindTooltip(`${p.n||"—"}<br><b>${v==null?"—":v+" min"}</b> `+
      `<small>(${metric=="r"?"realistic":"best"}, ${scen=="f"?"Muni+BART+Caltrain":"Muni only"})</small>`,
      {className:"tt",sticky:true}).openTooltip();});}
}).addTo(map);

L.marker(DEST,{icon:L.divIcon({className:"",html:'<div style="width:14px;height:14px;border-radius:50%;background:#ffd23f;border:2px solid #000;box-shadow:0 0 6px #000"></div>',iconSize:[14,14]})}).addTo(map).bindPopup("__LABEL__");

function redraw(){SCALE=ramp(ideal).S;layer.setStyle(style);legendGradient();
  document.getElementById("legtitle").textContent=
    (metric=="r"?"Realistic":"Best-case")+" door-to-door (min) · "+(scen=="f"?"Muni+BART+Caltrain":"Muni only");
  renderList();}
function renderList(){
  const k=key();
  const rows=NB.filter(n=>n[k]!=null).sort((a,b)=>a[k]-b[k]);
  document.getElementById("list").innerHTML=rows.map(n=>{
    const v=n[k]; const c=v<=thr?color(v):"#555";
    return `<div class="nb" data-lat="${n.lat}" data-lon="${n.lon}">
      <span><span style="color:${c}">●</span> ${n.name}
        <a href="${gmaps(n.lat,n.lon)}" target="_blank" title="Open transit directions in Google Maps"
           onclick="event.stopPropagation()" style="color:#5ab0ff;text-decoration:none;margin-left:4px">↗</a></span>
      <span class="t" style="color:${c}">${v}<small> min</small></span></div>`;}).join("");
  document.querySelectorAll(".nb").forEach(el=>el.onclick=()=>
    map.flyTo([+el.dataset.lat,+el.dataset.lon],14,{duration:.7}));
}
function seg(id,set){document.querySelectorAll(`#${id} button`).forEach(b=>b.onclick=()=>{
  document.querySelectorAll(`#${id} button`).forEach(x=>x.classList.remove("on"));
  b.classList.add("on");set(b.dataset.v);redraw();});}
seg("metric",v=>metric=v); seg("scenario",v=>scen=v);
document.getElementById("thr").oninput=e=>{thr=+e.target.value;
  document.getElementById("thrval").textContent=thr;redraw();};
document.getElementById("ideal").oninput=e=>{ideal=+e.target.value;
  document.getElementById("idval").textContent=ideal;redraw();};
redraw();
</script></body></html>"""


if __name__ == "__main__":
    main()
