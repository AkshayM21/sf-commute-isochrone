// Unit tests for the shared browser viz helpers (scripts/assets/viz.js): the
// time->color ramp, the Google Maps deep link, and the transit-mode palette.
//
// viz.js is plain browser script that declares globals (no module exports), so
// we read its source, append a tiny expression that collects the bindings, and
// eval it in this Node context. That keeps the file untouched while letting us
// assert on its functions directly.
//
// Run:  node --test tests/test_viz.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const VIZ_PATH = join(__dirname, "..", "scripts", "assets", "viz.js");

// Load viz.js once: eval the source, then return its top-level bindings.
const src = readFileSync(VIZ_PATH, "utf8");
const viz = eval(src + "\n;({ ramp, colorScale, gmapsURL, MODECOLOR, rgb });");
const { ramp, colorScale, gmapsURL, MODECOLOR } = viz;

const IDEAL = 25;

// --------------------------------------------------------------------------- //
// colorScale                                                                  //
// --------------------------------------------------------------------------- //
test("colorScale: v=0 returns the first (deep green) anchor", () => {
  const S = ramp(IDEAL).S;
  assert.equal(colorScale(0, S), "rgb(0,104,55)");
});

test("colorScale: v<=0 clamps to the first anchor", () => {
  const S = ramp(IDEAL).S;
  assert.equal(colorScale(-10, S), "rgb(0,104,55)");
});

test("colorScale: v=ideal hits the pale-yellow pivot [255,255,191]", () => {
  const S = ramp(IDEAL).S;
  assert.equal(colorScale(IDEAL, S), "rgb(255,255,191)");
});

test("colorScale: v>=ideal+25 hits the red anchor [215,48,39]", () => {
  const S = ramp(IDEAL).S;
  assert.equal(colorScale(IDEAL + 25, S), "rgb(215,48,39)");
  // Beyond the top anchor stays clamped to red.
  assert.equal(colorScale(IDEAL + 100, S), "rgb(215,48,39)");
});

test("colorScale: a midpoint linearly interpolates between two anchors", () => {
  const S = ramp(IDEAL).S;
  // First segment: v=0 -> [0,104,55], v=ideal*0.45=11.25 -> [26,152,80].
  // Midpoint v=5.625 (t=0.5) -> componentwise average, rounded.
  const v = (IDEAL * 0.45) / 2;
  const expected =
    `rgb(${Math.round((0 + 26) / 2)},` +
    `${Math.round((104 + 152) / 2)},` +
    `${Math.round((55 + 80) / 2)})`;
  assert.equal(colorScale(v, S), expected);
});

test("colorScale: null value returns null (uncomputed cell)", () => {
  const S = ramp(IDEAL).S;
  assert.equal(colorScale(null, S), null);
});

test("ramp: produces an 8-stop scale with hi = ideal+25", () => {
  const r = ramp(IDEAL);
  assert.equal(r.hi, IDEAL + 25);
  assert.equal(r.S.length, 8);
  // First/last anchors are the green and red endpoints.
  assert.deepEqual(r.S[0], [0, [0, 104, 55]]);
  assert.deepEqual(r.S[r.S.length - 1], [IDEAL + 25, [215, 48, 39]]);
});

test("ramp: the ideal stop is the pale-yellow pivot", () => {
  const r = ramp(IDEAL);
  const idealStop = r.S.find((s) => s[0] === IDEAL);
  assert.ok(idealStop, "ramp must contain a stop exactly at `ideal`");
  assert.deepEqual(idealStop[1], [255, 255, 191]);
});

test("ramp: a different ideal moves the pivot and hi", () => {
  const r = ramp(15);
  assert.equal(r.hi, 40);
  assert.equal(colorScale(15, r.S), "rgb(255,255,191)");
  assert.equal(colorScale(40, r.S), "rgb(215,48,39)");
});

// --------------------------------------------------------------------------- //
// gmapsURL                                                                     //
// --------------------------------------------------------------------------- //
test("gmapsURL: builds a Google Maps transit deep link with a depart-at time", () => {
  // viz.js uses Google's UNOFFICIAL data= format (the ?api=1 form can't set a depart time):
  //   /dir/<origin>/<dest>/@<dest>,14z/data=...!7e2(depart-at)!8j<epoch>!3e3(transit)
  const url = gmapsURL(37.78, -122.41, 37.7955, -122.3937);
  assert.match(url, /^https:\/\/www\.google\.com\/maps\/dir\/37\.78,-122\.41\/37\.7955,-122\.3937\//);
  assert.match(url, /!7e2!8j\d+!3e3$/);   // depart-at + epoch + transit mode
});

test("gmapsURL: always requests the transit travel mode", () => {
  const url = gmapsURL(1, 2, 3, 4);
  assert.match(url, /!3e3$/);             // transit
  assert.match(url, /\/dir\/1,2\/3,4\//); // origin / destination in the path
});

// --------------------------------------------------------------------------- //
// MODECOLOR                                                                    //
// --------------------------------------------------------------------------- //
test("MODECOLOR: has bart/metro/bus/cable keys", () => {
  for (const key of ["bart", "metro", "bus", "cable"]) {
    assert.ok(key in MODECOLOR, `MODECOLOR missing key: ${key}`);
    assert.match(MODECOLOR[key], /^#[0-9a-fA-F]{6}$/, `${key} not a hex color`);
  }
});

// --------------------------------------------------------------------------- //
// DEPART-AFTER rendering: active-journey normalization (Stage 4)               //
// --------------------------------------------------------------------------- //
// The depart-after /itinerary contract carries TWO distinct journeys per cell —
// `best` (p5) and `typical` (p50), each with its OWN legs that sum to its OWN total
// (no "+wait" reconciliation). The frontend NORMALIZES that both-journey response
// into the ACTIVE journey the existing renderers expect, RE-PICKED on the metric
// toggle. These pure pick/normalize helpers live in scripts/templates/index.html
// (no browser deps), so we slice them out of the template and exercise them with
// synthetic data — deterministic, no server needed.
//
// We build the helpers under DEPARTAFTER=true (a depart-after page) so normalizeBD
// is active, and assert: Best-case picks `best`, Typical picks `typical`, each with
// real=null (no reconciliation) + the carried frag; alts + compare options re-pick
// by the live metric too.
const TEMPLATE_PATH = join(__dirname, "..", "scripts", "templates", "index.html");
function _daHelpers() {
  const tsrc = readFileSync(TEMPLATE_PATH, "utf8");
  const fn = (name) => {
    const sig = "function " + name + "(";
    const i = tsrc.indexOf(sig);
    if (i < 0) throw new Error("template fn not found: " + name);
    let depth = 0, started = false;
    for (let k = i; k < tsrc.length; k++) {
      const c = tsrc[k];
      if (c === "{") { depth++; started = true; }
      else if (c === "}") { depth--; if (started && depth === 0) return tsrc.slice(i, k + 1); }
    }
    throw new Error("template fn unterminated: " + name);
  };
  const defs = ["bdFrag", "_daPick", "normalizeBD", "optDA", "optLegs", "optTotal",
    "optRead", "buildCompare"].map(fn).join("\n");
  const harness = `
"use strict";
let metric="b";
const REAL={}, VAR={};
const PRIMARY_KEY="__primary__";
const ALT_CASING=["#ff3db4","#11c7c7","#f59000","#a16bff"];
const DEPARTAFTER=true;            /* exercise the depart-after branch */
function dominantLine(legs){let b=null,m=-1;(legs||[]).forEach(g=>{if(g.mode==="transit"&&g.name&&g.min>m){m=g.min;b=g.name;}});return b||"walk only";}
function primaryCasing(){return "#fff";}
${defs}
return {set metric(v){metric=v;}, get metric(){return metric;},
        bdFrag, normalizeBD, optRead, optLegs, optTotal, buildCompare};`;
  return new Function(harness)();
}

// A synthetic depart-after /itinerary?pin=1 breakdown: best (p5=20) and typical (p50=26)
// are DIFFERENT journeys (different legs that sum to their own totals), plus a primary
// frag and two alts each carrying both journeys + its own frag.
function _daFixture() {
  return {
    name: "Test NB", olat: 37.7, olon: -122.4,
    total: 26, xfers: 1,
    legs: [{ mode: "walk", line: null, min: 3 },
           { mode: "transit", line: "N", min: 18, wait: 5 }],         // sums to 26
    geom: [{ mode: "walk", min: 3 }, { mode: "transit", name: "N", min: 18 }],
    typical: { total: 26, xfers: 1,
      legs: [{ mode: "walk", min: 3 }, { mode: "transit", name: "N", min: 18 }],
      geom: [{ mode: "walk", min: 3 }, { mode: "transit", name: "N", min: 18 }] },
    best: { total: 20, xfers: 1,
      legs: [{ mode: "walk", min: 2 }, { mode: "transit", name: "N", min: 16 }],
      geom: [{ mode: "walk", min: 2 }, { mode: "transit", name: "N", min: 16 }] },
    frag: 6,
    alts: [
      { line: "K", best: { total: 22, legs: [{ mode: "transit", name: "K", min: 20 }] },
                    typical: { total: 28, legs: [{ mode: "transit", name: "K", min: 24 }] }, frag: 4 },
      { line: "J", best: { total: 24, legs: [{ mode: "transit", name: "J", min: 22 }] },
                    typical: { total: 30, legs: [{ mode: "transit", name: "J", min: 26 }] }, frag: 2 },
    ],
  };
}

test("depart-after: normalizeBD picks best (p5) in Best-case, typical (p50) in Typical", () => {
  const H = _daHelpers();
  const d = _daFixture();
  H.metric = "b";
  let n = H.normalizeBD(d);
  assert.equal(n.total, 20, "Best-case total = best.total (p5)");
  assert.equal(n.legs, d.best.legs, "Best-case legs = best journey legs");
  assert.equal(n.real, null, "no reconciliation under depart-after (real=null)");
  H.metric = "r";
  n = H.normalizeBD(d);
  assert.equal(n.total, 26, "Typical total = typical.total (p50)");
  assert.equal(n.legs, d.typical.legs, "Typical legs = typical journey legs");
  assert.equal(n.real, null, "no reconciliation under depart-after (real=null)");
});

test("depart-after: normalizeBD carries primary frag for the bad-day chip (both modes)", () => {
  const H = _daHelpers();
  const d = _daFixture();
  for (const m of ["b", "r"]) {
    H.metric = m;
    assert.equal(H.normalizeBD(d).frag, 6, `frag carried in metric=${m}`);
  }
});

test("depart-after: normalizeBD flattens each alt to the active metric's {line,min,legs}", () => {
  const H = _daHelpers();
  const d = _daFixture();
  H.metric = "b";
  let alts = H.normalizeBD(d).alts;
  assert.deepEqual(alts.map((a) => [a.line, a.min]), [["K", 22], ["J", 24]], "alts -> best totals");
  H.metric = "r";
  alts = H.normalizeBD(d).alts;
  assert.deepEqual(alts.map((a) => [a.line, a.min]), [["K", 28], ["J", 30]], "alts -> typical totals");
});

test("depart-after: optRead/optLegs/optTotal re-pick by the live metric (compare strips)", () => {
  const H = _daHelpers();
  const d = _daFixture();
  // Build the compare list ONCE (under Best-case) — like renderPin — then toggle the metric and
  // confirm a STALE list still resolves the active journey (drawSelected re-picks at draw time).
  H.metric = "b";
  const list = H.buildCompare(d);
  const prim = list[0], altK = list[1];
  assert.equal(H.optTotal(prim), 20, "primary best-case total");
  assert.equal(H.optTotal(altK), 22, "alt K best-case total");
  assert.equal(H.optRead(prim).head, 20, "optRead head = best-case");
  assert.equal(H.optRead(prim).waitExtra, 0, "no +wait reconciliation");
  assert.equal(H.optRead(prim).frag, 0, "no per-route frag in Best-case");
  // Toggle to Typical — the SAME (stale) list must now resolve the typical journeys + per-route frag.
  H.metric = "r";
  assert.equal(H.optTotal(prim), 26, "primary typical total after toggle (no re-fetch)");
  assert.equal(H.optTotal(altK), 28, "alt K typical total after toggle");
  assert.equal(H.optRead(prim).head, 26, "optRead head = typical");
  assert.equal(H.optRead(prim).frag, 6, "per-route frag shown in Typical");
  assert.equal(H.optRead(altK).frag, 4, "alt K per-route frag in Typical");
  // legs flip to the active journey too (distinct geometry per metric).
  assert.equal(H.optLegs(prim), d.typical.legs, "primary legs = typical journey legs in Typical");
});

// --------------------------------------------------------------------------- //
// summaryVM: the ONE active-journey view-model for the header (both semantics)  //
// --------------------------------------------------------------------------- //
// The arrive-by/depart-after split in summaryHTML was unified into a single view-model, summaryVM,
// that resolves {head, headLab, headTilde, otherVal, otherLab, otherTilde, frag, stuck, showSig}
// once per metric so summaryHTML carries NO if(DEPARTAFTER) branch. The KEY semantic difference is
// encoded in the VM: ARRIVE-BY best-case+typical are the SAME journey (typical = +wait, so it's
// APPROXIMATE → "~" tilde, and the chip is gated on `real` landing); DEPART-AFTER are DISTINCT
// journeys (each exact, no "~", chip shows in both modes). These pure helpers live in the template;
// slice them out and assert the resolved fields per semantic × metric. (bdFrag is a dependency.)
function _summaryVM(departafter) {
  const tsrc = readFileSync(TEMPLATE_PATH, "utf8");
  const fn = (name) => {
    const sig = "function " + name + "(";
    const i = tsrc.indexOf(sig);
    if (i < 0) throw new Error("template fn not found: " + name);
    let depth = 0, started = false;
    for (let k = i; k < tsrc.length; k++) {
      const c = tsrc[k];
      if (c === "{") { depth++; started = true; }
      else if (c === "}") { depth--; if (started && depth === 0) return tsrc.slice(i, k + 1); }
    }
    throw new Error("template fn unterminated: " + name);
  };
  const defs = ["bdFrag", "summaryVM"].map(fn).join("\n");
  const harness = `
"use strict";
let metric="b";
const DEPARTAFTER=${departafter ? "true" : "false"};
${defs}
return {set metric(v){metric=v;}, get metric(){return metric;}, summaryVM};`;
  return new Function(harness)();
}

test("summaryVM (arrive-by): same journey — typical gets the ~ tilde, chip gated on real", () => {
  const H = _summaryVM(false);
  // Before /variance: real undefined → typical headline NOT shown; Best-case headline, chip hidden.
  const pre = { total: 22, xfers: 1 };
  H.metric = "r";
  let vm = H.summaryVM(pre);
  assert.equal(vm.head, 22, "no real → headline stays best-case total");
  assert.equal(vm.headLab, "best-case");
  assert.equal(vm.headTilde, "", "best-case headline never gets ~");
  assert.equal(vm.showSig, false, "chip hidden until realistic lands");
  // After /variance: real=28 (committed typical), var carries frag/stuck.
  const post = { total: 22, xfers: 1, real: 28, var: { frag: 6, stuck: 0.03 } };
  vm = H.summaryVM(post);                          // Typical
  assert.equal(vm.head, 28, "Typical headline = real (committed typical)");
  assert.equal(vm.headLab, "typical");
  assert.equal(vm.headTilde, "~", "arrive-by typical is approximate → ~");
  assert.equal(vm.otherVal, 22, "other = best-case total");
  assert.equal(vm.otherLab, "best-case");
  assert.equal(vm.otherTilde, "");
  assert.equal(vm.frag, 6); assert.equal(vm.stuck, 0.03); assert.equal(vm.showSig, true);
  H.metric = "b";                                  // Best-case: other = typical, gets ~
  vm = H.summaryVM(post);
  assert.equal(vm.head, 22); assert.equal(vm.headLab, "best-case"); assert.equal(vm.headTilde, "");
  assert.equal(vm.otherVal, 28); assert.equal(vm.otherLab, "typical"); assert.equal(vm.otherTilde, "~");
});

test("summaryVM (depart-after): distinct journeys — no ~, chip shows in both modes", () => {
  const H = _summaryVM(true);
  // normalizeBD flattens the active journey onto d.total; raw best/typical pair stays for `other`.
  const dTyp = { total: 26, xfers: 1, best: { total: 20 }, typical: { total: 26 }, frag: 6,
    var: { frag: 5, stuck: 0.04 } };
  H.metric = "r";
  let vm = H.summaryVM(dTyp);
  assert.equal(vm.head, 26, "Typical head = active (flattened) total");
  assert.equal(vm.headLab, "typical");
  assert.equal(vm.headTilde, "", "depart-after typical is exact → no ~");
  assert.equal(vm.otherVal, 20, "other = best journey total");
  assert.equal(vm.otherLab, "best-case"); assert.equal(vm.otherTilde, "");
  assert.equal(vm.frag, 6, "primary per-route frag wins over cell overlay");
  assert.equal(vm.showSig, true, "depart-after chip shows in both modes");
  // Best-case: active flattened total is the best journey; other = typical (still no ~).
  const dBest = { total: 20, xfers: 2, best: { total: 20 }, typical: { total: 26 }, frag: 6,
    var: { frag: 5, stuck: 0.04 } };
  H.metric = "b";
  vm = H.summaryVM(dBest);
  assert.equal(vm.head, 20); assert.equal(vm.headLab, "best-case"); assert.equal(vm.headTilde, "");
  assert.equal(vm.otherVal, 26); assert.equal(vm.otherLab, "typical"); assert.equal(vm.otherTilde, "");
  assert.equal(vm.showSig, true);
});
