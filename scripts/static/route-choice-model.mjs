/* Pure route-choice data model shared by the route inspector and its tests.
 *
 * The browser supplies only the small pieces of presentation state which are
 * intentionally live (metric, identity colours, and the dominant-line label).
 * Keeping those at the call boundary makes this module usable without a DOM.
 */

export function stableRouteKey(family, branch, fallback) {
  if (family && family.key != null && branch && branch.key != null) {
    return "struct:" + JSON.stringify([String(family.key), String(branch.key)]);
  }
  return fallback;
}

export function routeChoiceKey(option, fallback) {
  const choiceKey = option && option.choice_key;
  if (typeof choiceKey === "string" && choiceKey) return choiceKey;
  return stableRouteKey(option && option.family, option && option.branch, fallback);
}

export function daPick(o, fallback, metric = "b") {
  const journey = metric === "r" ? (o && o.typical) : (o && o.best);
  const selected = journey || fallback || {};
  const out = { total: selected.total, min: selected.total, legs: selected.legs, real: null };
  if ("geom" in selected || "geom" in (fallback || {})) {
    out.geom = "geom" in selected ? selected.geom : fallback.geom;
  }
  return out;
}

export function breakdownFrag(d, departAfter) {
  if (departAfter && d && d.frag != null) return d.frag;
  return d && d.var && d.var.frag != null ? d.var.frag : 0;
}

export function normalizeBD(d, { departAfter = false, metric = "b" } = {}) {
  if (!departAfter || !d || d.error) return d;
  const journey = daPick(d, { total: d.total, xfers: d.xfers, legs: d.legs, geom: d.geom }, metric);
  const active = metric === "r" ? d.typical : d.best;
  const out = Object.assign({}, d, {
    total: journey.total,
    xfers: active && active.xfers != null ? active.xfers : d.xfers,
    legs: journey.legs,
    geom: journey.geom,
    real: null,
    frag: breakdownFrag(d, departAfter),
  });
  out.alts = (d.alts || []).map((a) => {
    const picked = daPick(a, undefined, metric);
    return { line: a.line, min: picked.min, total: picked.total, legs: picked.legs,
      real: null, frag: a.frag != null ? a.frag : null };
  });
  return out;
}

export function optDA(o, metric = "b") {
  if (!o || !o.da) return null;
  return metric === "r" ? (o.da.typical || o.da.best) : (o.da.best || o.da.typical);
}

export function optLegs(o, metric = "b") {
  const journey = optDA(o, metric);
  return journey ? (journey.legs || []) : ((o && o.legs) || []);
}

export function optTotal(o, metric = "b") {
  const journey = optDA(o, metric);
  return journey ? journey.total : (o && o.total);
}

export function optRead(o, metric = "b") {
  if (o && o.da) {
    const head = optTotal(o, metric);
    const fragKnown = o.da.frag != null;
    const typical = o.da.typical || o.da.best || {};
    return { showR: metric === "r", head, waitExtra: 0,
      frag: fragKnown ? Number(o.da.frag) : null, fragKnown,
      badDayBase: Number(typical.total != null ? typical.total : head) };
  }
  const real = o && o.real != null ? o.real : null;
  const showR = metric === "r" && real != null;
  const head = showR ? real : (o && o.total);
  const fragKnown = real != null && o && o.frag != null;
  return { showR, head, waitExtra: showR ? Math.max(0, real - o.total) : 0,
    frag: fragKnown ? Number(o.frag) : null, fragKnown,
    badDayBase: Number(real != null ? real : head) };
}

export function uniq(a) {
  const out = [];
  (a || []).forEach((x) => { if (x && !out.includes(x)) out.push(x); });
  return out;
}

export function tlegs(o, metric = "b") {
  return optLegs(o, metric).filter((g) => g && g.mode === "transit" && (g.name || g.line));
}
export function lname(g) { return String((g && g.name) || (g && g.line) || ""); }
export function lineNames(o, metric = "b") { return tlegs(o, metric).map(lname); }

export function routeServices(raw, fallbackLines) {
  const out = [], seen = new Set();
  (Array.isArray(raw) ? raw : []).forEach((service, i) => {
    if (!service || typeof service !== "object") return;
    const name = String(service.name || ""); if (!name) return;
    const key = String(service.key || `display:${name}:${i}`);
    if (seen.has(key)) return;
    seen.add(key); out.push({ ...service, key, name });
  });
  if (!out.length) (fallbackLines || []).forEach((line, i) => {
    const name = String(line || ""); if (!name) return;
    const key = `legacy-line:${name}:${i}`;
    if (!seen.has(key)) { seen.add(key); out.push({ key, name }); }
  });
  return out;
}
export function serviceNames(meta) {
  return uniq(routeServices(meta && meta.services, meta && meta.lines).map((s) => s.name));
}

export function routeGrouping(o, metric = "b") {
  const family = o && o.family, branch = o && o.branch;
  if (family && family.key != null && branch && branch.key != null) {
    const flines = uniq(Array.isArray(family.lines) ? family.lines.map(String) : []);
    const blines = uniq(Array.isArray(branch.lines) ? branch.lines.map(String) : []);
    const tags = uniq(Array.isArray(family.tags) ? family.tags.map(String) : []);
    const fservices = routeServices(family.services, flines);
    const bservices = routeServices(branch.services, []);
    const serviceKeys = uniq(Array.isArray(branch.serviceKeys) ? branch.serviceKeys.map(String) :
      bservices.map((service) => service.key));
    return {
      family: { key: String(family.key), name: String(family.name || flines.join(" / ") || "Route family"),
        sub: String(family.sub || "route family"), lines: flines, tags, services: fservices },
      branch: { key: String(branch.key), name: String(branch.name || "Route option"),
        kind: String(branch.kind || "route"), lines: blines, services: bservices, serviceKeys },
    };
  }
  const names = uniq(lineNames(o, metric));
  const label = String((o && o.line) || names.join(" > ") || (o && o.isPrimary ? "Primary route" : "Alternate route"));
  const key = "legacy:" + String((o && o.key) || "route");
  return {
    family: { key, name: label, sub: (o && o.isPrimary) ? "primary route" : "alternate route",
      lines: names, tags: [], services: routeServices([], names) },
    branch: { key: key + ":branch", name: names.length ? label : "Walk finish",
      kind: names.length ? "route" : "walk", lines: names, services: [], serviceKeys: [] },
  };
}

export function buildCompare(d, {
  departAfter = false, metric = "b", primaryKey = "__primary__",
  primaryColor = "#fff", altCasing = [], dominantLine = (legs) => {
    let best = null, bestMin = -1;
    (legs || []).forEach((g) => { if (g.mode === "transit" && g.name && g.min > bestMin) {
      bestMin = g.min; best = g.name;
    }});
    return best || "walk only";
  },
} = {}) {
  const out = [], da = departAfter && d && !d.error;
  if (d && d.geom) {
    if (da) {
      out.push({ key: routeChoiceKey(d, primaryKey), choiceKey: d.choice_key || null,
        line: dominantLine((d.typical || d.best || { geom: d.geom }).geom || d.geom),
        identityColor: primaryColor, isPrimary: true, family: d.family || null, branch: d.branch || null,
        da: { best: d.best ? { total: d.best.total, legs: d.best.geom } : null,
          typical: d.typical ? { total: d.typical.total, legs: d.typical.geom } : { total: d.total, legs: d.geom },
          frag: d.frag != null ? d.frag : (d.var && d.var.frag != null ? d.var.frag : null) } });
    } else {
      out.push({ key: routeChoiceKey(d, primaryKey), choiceKey: d.choice_key || null,
        line: dominantLine(d.geom), identityColor: primaryColor, legs: d.geom, total: d.total,
        real: d.real != null ? d.real : null, frag: d.var ? d.var.frag : null, isPrimary: true,
        family: d.family || null, branch: d.branch || null });
    }
  }
  const altHasTransit = (a) => {
    const lists = da ? [(a.best || {}).legs, (a.typical || {}).legs] : [a.legs];
    return lists.some((legs) => (legs || []).some((g) => g && g.mode === "transit"));
  };
  const legSig = (legs) => (legs || []).map((g) => {
    const name = (g && (g.line || g.name)) || "";
    const min = g && Number.isFinite(+g.min) ? Math.round(+g.min) : "";
    const wait = g && Number.isFinite(+g.wait) ? Math.round(+g.wait) : "";
    const pts = g && Array.isArray(g.pts) ? g.pts.map((p) => Array.isArray(p) ? p.map(Number).join(",") : "").join(";") : "";
    return [g && g.mode || "", name, min, wait, pts].join(":");
  }).join("|");
  const optSig = (o) => {
    if (!o) return "";
    const grouping = o.family && o.family.key != null && o.branch && o.branch.key != null
      ? "||group:" + JSON.stringify([String(o.family.key), String(o.branch.key)]) : "";
    if (o.da) {
      const best = o.da.best ? legSig(o.da.best.legs) : "";
      const typical = o.da.typical ? legSig(o.da.typical.legs) : "";
      return [best || typical, typical || best].join("||") + grouping;
    }
    return legSig(o.legs) + grouping;
  };
  (d && d.alts || []).filter(altHasTransit).forEach((a, k) => {
    const key = routeChoiceKey(a, "alt" + k);
    if (da) out.push({ key, choiceKey: a.choice_key || null, line: a.line,
      identityColor: altCasing.length ? altCasing[k % altCasing.length] : undefined, isPrimary: false,
      family: a.family || null, branch: a.branch || null,
      da: { best: a.best || null, typical: a.typical || null, frag: a.frag != null ? a.frag : null } });
    else out.push({ key, choiceKey: a.choice_key || null, line: a.line,
      identityColor: altCasing.length ? altCasing[k % altCasing.length] : undefined, legs: a.legs, total: a.min,
      real: a.real != null ? a.real : null, frag: a.frag != null ? a.frag : null, isPrimary: false,
      family: a.family || null, branch: a.branch || null });
  });
  const seen = new Set(), deduped = []; let altSlot = 0;
  out.forEach((o) => {
    const sig = optSig(o) + (o.choiceKey ? "||choice:" + o.choiceKey : "");
    if (sig && seen.has(sig)) return;
    if (sig) seen.add(sig);
    if (!o.isPrimary && altCasing.length) o.identityColor = altCasing[(altSlot++) % altCasing.length];
    deduped.push(o);
  });
  return deduped;
}

export function formatMinutes(value) {
  const minutes = Number(value);
  if (!Number.isFinite(minutes)) return "—";
  if (minutes > 0 && minutes < 1) return "<1 min";
  const whole = Math.round(minutes);
  if (Math.abs(minutes - whole) < 0.05) return `${whole} min`;
  return `${Math.round(minutes * 10) / 10} min`;
}
export function naturalList(items) {
  const values = uniq((items || []).map(String));
  if (values.length < 2) return values[0] || "transit";
  if (values.length === 2) return `${values[0]} or ${values[1]}`;
  return `${values.slice(0, -1).join(", ")}, or ${values[values.length - 1]}`;
}
export function displayStopName(name) {
  return String(name || "").replace(/\s*&\s*/g, " & ").replace(/\s+/g, " ").trim();
}
export function legPhysicalMin(g) {
  const physical = Number(g && g.physical_min), fallback = Number(g && g.min);
  return Number.isFinite(physical) && physical >= 0 ? physical : (Number.isFinite(fallback) && fallback >= 0 ? fallback : 0);
}
export function legScheduleAllowanceMin(g) {
  if (!g || g.mode !== "walk") return 0;
  const explicit = [g.schedule_allowance_min, g.allowance_min, g.slack_min].map(Number)
    .find((v) => Number.isFinite(v) && v > 0);
  if (explicit != null) return explicit;
  const total = Number(g.min), physical = Number(g.physical_min);
  return Number.isFinite(total) && Number.isFinite(physical) ? Math.max(0, total - physical) : 0;
}
export function optionWalk(o, metric = "b") { return optLegs(o, metric).filter((g) => g && g.mode === "walk").reduce((s, g) => s + legPhysicalMin(g), 0); }
export function optionScheduleAllowance(o, metric = "b") { return optLegs(o, metric).filter((g) => g && g.mode === "walk").reduce((s, g) => s + legScheduleAllowanceMin(g), 0); }
export function optionTransfers(o, metric = "b") { return Math.max(0, tlegs(o, metric).length - 1); }

export function buildFamilies(compareList, metric = "b") {
  const reads = (compareList || []).map((o) => ({ o, r: optRead(o, metric) })).sort((a, b) => a.r.head - b.r.head);
  const byKey = new Map();
  reads.forEach((x) => {
    const grouping = routeGrouping(x.o, metric), key = grouping.family.key;
    if (!byKey.has(key)) byKey.set(key, { key, meta: grouping.family, opts: [], branches: new Map() });
    const family = byKey.get(key); family.opts.push(x);
    const branchKey = grouping.branch.key;
    if (!family.branches.has(branchKey)) family.branches.set(branchKey, { key: branchKey, meta: grouping.branch, opts: [] });
    family.branches.get(branchKey).opts.push(x);
  });
  return Array.from(byKey.values()).map((family) => {
    family.branches = Array.from(family.branches.values()).map((branch) => {
      branch.opts.sort((a, b) => a.r.head - b.r.head); return branch;
    }).sort((a, b) => a.opts[0].r.head - b.opts[0].r.head);
    return family;
  }).sort((a, b) => a.opts[0].r.head - b.opts[0].r.head);
}

export function hoverAltChipData(rawD, options = {}) {
  if (!rawD || rawD.error) return [];
  return buildCompare(rawD, options).filter((o) => !o.isPrimary)
    .map((o) => ({ line: o.line, min: optRead(o, options.metric || "b").head, color: o.identityColor }));
}
