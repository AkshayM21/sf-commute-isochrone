/* Map rendering boundary.
 *
 * The page/controller owns commute state, requests, and interaction policy.  This module owns
 * Leaflet objects and the deterministic geometry/style decisions used to paint them.  Leaflet is
 * deliberately injected so the pure decisions and layer lifecycle can be tested without a map.
 */

export const BASE_TILES = Object.freeze({
  // OpenStreetMap's standard raster endpoint is genuinely keyless.  Keep the same source for
  // both themes; app.css applies a restrained dark filter to the tile pane so appearance remains
  // theme-aware without requiring a second provider, a token, or a secret in the browser.
  dark: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  light: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
});

export const TILE_OPTS = Object.freeze({
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
});

export const ALT_CASING = Object.freeze(["#ff3db4", "#11c7c7", "#f59000", "#a16bff", "#7ed957", "#ff6b6b"]);
export const SEPARATOR = "rgba(12,14,18,.85)";
export const PRIMARY_KEY = "__primary__";
export const DASH_WALK = "2 6";
export const MIN_SPAN_DEG = 0.012;

export function focusHaloStyle(theme = "dark") {
  return { fillOpacity: 0, color: theme === "light" ? "#071625" : "#f7fbff", opacity: 1, weight: 7 };
}

export function focusStyle(theme = "dark") {
  return { fillOpacity: 0, color: theme === "light" ? "#fff" : "#071625", opacity: 1, weight: 3 };
}

export function cellStyle({ value, threshold = 40, mapcolors = "on", colorMode = "time", opacity = 1,
  color = () => null, line = null, lineColors = {} } = {}) {
  if (value == null || value > threshold || mapcolors === "off") return { fillOpacity: 0, opacity: 0, weight: 0 };
  if (colorMode === "line") {
    return line ? { fillColor: lineColors[line] || "#888", fillOpacity: Math.min(opacity + .04, .9), weight: 0 }
      : { fillColor: "#888", fillOpacity: .06, weight: 0 };
  }
  return { fillColor: color(value), fillOpacity: opacity, weight: 0 };
}

export function normalizePoints(points) {
  return (Array.isArray(points) ? points : []).filter((p) => Array.isArray(p) && p.length >= 2)
    .map((p) => [Number(p[0]), Number(p[1])])
    .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
}

export function normalizeGeometry(legs) {
  return (Array.isArray(legs) ? legs : []).map((leg) => ({ ...leg, pts: normalizePoints(leg && leg.pts) }))
    .filter((leg) => leg.pts.length > 1);
}

export function routeSegmentSpecs(legs, { routeColor = () => "#888", identityColor = "#fff", opacity = 1 } = {}) {
  const segs = normalizeGeometry(legs);
  const specs = [];
  segs.forEach((g) => {
    if (g.mode !== "walk") {
      specs.push({ kind: "casing", pts: g.pts, color: identityColor, weight: 10, opacity: .95 * opacity });
      specs.push({ kind: "separator", pts: g.pts, color: SEPARATOR, weight: 7, opacity: .9 * opacity });
    }
  });
  segs.forEach((g) => {
    if (g.mode === "walk") {
      specs.push({ kind: "walk-casing", pts: g.pts, color: identityColor, weight: 8, opacity: .95 * opacity });
      specs.push({ kind: "walk-separator", pts: g.pts, color: SEPARATOR, weight: 5, opacity: .9 * opacity });
      specs.push({ kind: "walk", pts: g.pts, color: "#9aa3af", weight: 3, dashArray: DASH_WALK, opacity: .98 * opacity });
    } else {
      specs.push({ kind: "transit", pts: g.pts, color: routeColor(g), weight: 4.5, opacity });
    }
  });
  return specs;
}

export function routeLabelSpecs(legs, { routeColor = () => "#888", firstTransitLabel = "" } = {}) {
  let index = 0;
  return normalizeGeometry(legs).filter((g) => g.mode !== "walk").map((g) => ({
    pts: g.pts, color: routeColor(g), name: index++ === 0 && firstTransitLabel ? firstTransitLabel : g.name, min: g.min,
  }));
}

export function paddedBounds(L, bounds, minSpan = MIN_SPAN_DEG) {
  if (!bounds || !bounds.isValid || !bounds.isValid()) return null;
  const dLat = bounds.getNorth() - bounds.getSouth(), dLon = bounds.getEast() - bounds.getWest();
  if (dLat >= minSpan || dLon >= minSpan) return bounds;
  const c = bounds.getCenter(), h = minSpan / 2;
  return L.latLngBounds([c.lat - h, c.lng - h], [c.lat + h, c.lng + h]);
}

export function fitOptions(insets = {}, { topPad = insets.top || 0, reducedMotion = false } = {}) {
  return {
    paddingTopLeft: [insets.left || 0, topPad],
    paddingBottomRight: [insets.right || 0, insets.bottom || 0],
    maxZoom: 15, animate: !reducedMotion, duration: reducedMotion ? 0 : 0.5,
  };
}

export function createMapRenderer({ L, map, getTheme = () => "dark", getCellStyle, getRouteColor = () => "#888",
  escapeHTML = (s) => String(s), getDestination = () => null, getViewInsets = () => ({}),
  getReducedMotion = () => false, onRouteDrawn = () => {} } = {}) {
  if (!L || !map) throw new TypeError("createMapRenderer requires injected Leaflet and map");
  const panes = { cells: "cellsPane", halo: "cellFocusHaloPane", focus: "cellFocusPane", route: "routePane" };
  map.createPane(panes.cells); map.getPane(panes.cells).style.zIndex = 400;
  map.createPane(panes.halo); map.getPane(panes.halo).style.zIndex = 414; map.getPane(panes.halo).style.pointerEvents = "none";
  map.createPane(panes.focus); map.getPane(panes.focus).style.zIndex = 415; map.getPane(panes.focus).style.pointerEvents = "none";
  map.createPane(panes.route); map.getPane(panes.route).style.zIndex = 430; map.getPane(panes.route).style.pointerEvents = "none";

  const focusHaloLayer = L.geoJSON(null, { pane: panes.halo, interactive: false, style: () => focusHaloStyle(getTheme()) }).addTo(map);
  const focusLayer = L.geoJSON(null, { pane: panes.focus, interactive: false, style: () => focusStyle(getTheme()) }).addTo(map);
  const routeLayer = L.layerGroup().addTo(map);
  const routeSvg = L.svg({ pane: panes.route }); map.addLayer(routeSvg);
  let baseTheme = getTheme() === "light" ? "light" : "dark";
  let baseLayer = L.tileLayer(BASE_TILES[baseTheme], TILE_OPTS).addTo(map);
  let destinationMarker = null;
  let drawn = null;

  function createCells(features, { onEachFeature } = {}) {
    return L.geoJSON(features, { pane: panes.cells, style: getCellStyle, onEachFeature }).addTo(map);
  }
  function createOverlays(lines, modeColors) {
    const styles = { bus: { color: modeColors.bus, weight: 1.3, opacity: .5 }, metro: { color: modeColors.metro, weight: 2.4, opacity: .85 }, cable: { color: modeColors.cable, weight: 2, opacity: .8 }, bart: { color: modeColors.bart, weight: 3, opacity: .85 } };
    const overlays = {};
    ["bart", "metro", "bus", "cable"].forEach((mode) => {
      overlays[mode] = L.geoJSON({ type: "FeatureCollection", features: (lines.features || []).filter((f) => f.properties.mode === mode) },
        { style: () => styles[mode], interactive: false });
    });
    return overlays;
  }
  function setTheme(theme) {
    const next = theme === "light" ? "light" : "dark";
    if (next !== baseTheme) {
      if (BASE_TILES[next] !== BASE_TILES[baseTheme]) {
        map.removeLayer(baseLayer); baseLayer = L.tileLayer(BASE_TILES[next], TILE_OPTS).addTo(map); baseLayer.bringToBack();
      }
      baseTheme = next;
    }
    focusHaloLayer.setStyle(() => focusHaloStyle(next)); focusLayer.setStyle(() => focusStyle(next));
  }
  function showCellFocus(feature) { focusHaloLayer.clearLayers(); focusLayer.clearLayers(); if (feature) { focusHaloLayer.addData(feature); focusLayer.addData(feature); } }
  function clearCellFocus() { focusHaloLayer.clearLayers(); focusLayer.clearLayers(); }
  function setDestinationMarker(latlng) {
    if (!latlng) { if (destinationMarker) map.removeLayer(destinationMarker); destinationMarker = null; return null; }
    if (destinationMarker) destinationMarker.setLatLng(latlng);
    else destinationMarker = L.marker(latlng).addTo(map);
    return destinationMarker;
  }
  function clearRoute() { routeLayer.clearLayers(); drawn = null; onRouteDrawn(null); }
  function placePills(specs) {
    const placed = [], minGap = 26;
    specs.forEach((spec) => {
      const mid = Math.floor(spec.pts.length / 2), order = [mid];
      for (let step = 1; step < spec.pts.length; step++) { if (mid + step < spec.pts.length) order.push(mid + step); if (mid - step >= 0) order.push(mid - step); }
      let chosen = null;
      for (const i of order) { const ll = L.latLng(spec.pts[i][0], spec.pts[i][1]), cp = map.latLngToContainerPoint(ll); if (!placed.some((p) => Math.hypot(p.cp.x - cp.x, p.cp.y - cp.y) < minGap)) { chosen = { cp, latlng: [spec.pts[i][0], spec.pts[i][1]], spec }; break; } }
      if (!chosen) { const ll = L.latLng(spec.pts[mid][0], spec.pts[mid][1]); chosen = { cp: map.latLngToContainerPoint(ll), latlng: [ll.lat, ll.lng], spec }; }
      placed.push(chosen);
    });
    return placed;
  }
  function rectAround(cp, w, h, cx, cy) { const pad = 4, l = cp.x - w * cx, t = cp.y - h * cy; return { l: l - pad, t: t - pad, r: l + w + pad, b: t + h + pad }; }
  function drawOne(legs, identityColor, totalMin, badgeMin, opts = {}) {
    if (badgeMin == null) badgeMin = totalMin;
    if (!opts.append) routeLayer.clearLayers();
    const segs = normalizeGeometry(legs); if (!segs.length) { if (!opts.append) drawn = null; return { segs: [], placed: [], badgeLL: null }; }
    const specs = routeSegmentSpecs(segs, { routeColor: getRouteColor, identityColor, opacity: opts.opacity == null ? 1 : opts.opacity });
    specs.forEach((s) => routeLayer.addLayer(L.polyline(s.pts, { renderer: routeSvg, color: s.color, weight: s.weight, opacity: s.opacity, dashArray: s.dashArray, interactive: false, lineCap: "round", lineJoin: "round" })));
    const showLabels = opts.labels !== false, showBadge = opts.badge !== false;
    const pills = showLabels ? placePills(routeLabelSpecs(segs, { routeColor: getRouteColor, firstTransitLabel: opts.firstTransitLabel })) : [];
    const badgeLL = segs[0].pts[0];
    if (showBadge) routeLayer.addLayer(L.marker(badgeLL, { interactive: false, keyboard: false, zIndexOffset: 300, icon: L.divIcon({ className: "rtwrap", iconSize: null, html: `<span class="rtbadge">~${badgeMin} min</span>` }) }));
    if (showLabels) pills.forEach((p) => routeLayer.addLayer(L.marker(p.latlng, { interactive: false, keyboard: false, zIndexOffset: 200, icon: L.divIcon({ className: "rtwrap", iconSize: null, html: `<span class="rtlab" style="background:${p.spec.color}">${escapeHTML(p.spec.name)} · ${p.spec.min}m</span>` }) })));
    const avoid = [];
    segs.forEach((g) => g.pts.forEach((p) => { const cp = map.latLngToContainerPoint(L.latLng(p[0], p[1])); avoid.push({ x: cp.x, y: cp.y }); }));
    const blocks = pills.map((p) => rectAround(map.latLngToContainerPoint(L.latLng(p.latlng[0], p.latlng[1])), 64, 20, .5, .5));
    if (showBadge) blocks.push(rectAround(map.latLngToContainerPoint(L.latLng(badgeLL[0], badgeLL[1])), 82, 24, .5, 1.35));
    const destination = getDestination();
    if (destination) blocks.push(rectAround(map.latLngToContainerPoint(L.latLng(destination[0], destination[1])), 25, 41, .5, 1));
    const result = { segs, placed: pills, badgeLL: showBadge ? badgeLL : null, avoid, blocks };
    onRouteDrawn(result);
    return result;
  }
  function fitBoundsUnoccluded(bounds) {
    const b = paddedBounds(L, bounds); if (!b) return;
    const insets = getViewInsets(); let topPad = insets.top || 0; const destination = getDestination(); const span = b.getNorth() - b.getSouth();
    if (destination && span > 0 && (b.getNorth() - destination[0]) < span * .15) topPad = Math.max(topPad, 41 + 12);
    map.fitBounds(b, fitOptions(insets, { topPad, reducedMotion: getReducedMotion() }));
  }
  function avoidForOptions(options) {
    const avoid = [], blocks = [];
    (options || []).forEach((legs) => normalizeGeometry(legs).forEach((g) => g.pts.forEach((p) => {
      const cp = map.latLngToContainerPoint(L.latLng(p[0], p[1])); avoid.push({ x: cp.x, y: cp.y });
    })));
    const destination = getDestination();
    if (destination) blocks.push(rectAround(map.latLngToContainerPoint(L.latLng(destination[0], destination[1])), 25, 41, .5, 1));
    return { avoid, blocks };
  }
  function remove() { clearRoute(); setDestinationMarker(null); [focusHaloLayer, focusLayer, routeSvg, baseLayer].forEach((l) => map.removeLayer(l)); }
  return { panes, focusHaloLayer, focusLayer, routeLayer, routeSvg, get baseLayer() { return baseLayer; }, get baseTheme() { return baseTheme; }, createCells, createOverlays, setTheme, setDestinationMarker, showCellFocus, clearCellFocus, clearRoute, drawOne, avoidForOptions, fitBoundsUnoccluded, remove, cellStyle: getCellStyle };
}
