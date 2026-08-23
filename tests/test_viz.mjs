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
const TSRC = readFileSync(TEMPLATE_PATH, "utf8");
// Slice a `function name(...){...}` declaration out of the LIVE template source (brace-balanced),
// so the tests exercise the shipped code, not a drifting copy. Shared by every harness below.
function templateFn(name) {
  const sig = "function " + name + "(";
  const i = TSRC.indexOf(sig);
  if (i < 0) throw new Error("template fn not found: " + name);
  let depth = 0, started = false;
  for (let k = i; k < TSRC.length; k++) {
    const c = TSRC[k];
    if (c === "{") { depth++; started = true; }
    else if (c === "}") { depth--; if (started && depth === 0) return TSRC.slice(i, k + 1); }
  }
  throw new Error("template fn unterminated: " + name);
}
// Extract a one-line `const NAME=...;` declaration from the live template. Extracting the palette
// keeps the test harness aligned with the shipped identity colors.
function templateConst(name) {
  const m = TSRC.match(new RegExp("const " + name + "=[^;\\n]*;"));
  if (!m) throw new Error("template const not found: " + name);
  return m[0];
}
// Evaluate extracted const(s) and return the LAST one's value.
function templateConstValue(...names) {
  const decls = names.map(templateConst).join("\n");
  return new Function(decls + "\nreturn " + names[names.length - 1] + ";")();
}
const T_ALT_CASING = templateConstValue("ALT_CASING");   // the live identity-casing palette

// Startup state is deliberately classified in the synchronous <head> script so a returning user
// or shared commute link never paints the first-run address card before normal boot begins.
function _startupClassifier() {
  return new Function(`"use strict";${templateFn("classifyStartupState")};return classifyStartupState;`)();
}

test("startup classifier: valid shared wp wins, validates both coordinates, and otherwise uses valid saved workplace", () => {
  const classify = _startupClassifier();
  const saved = JSON.stringify({ lat: 37.7749, lon: -122.4194, label: "Private label" });
  assert.equal(classify("#wp=37.79,-122.39,Shared%20workplace", saved), "restoring");
  assert.equal(classify("#wp=37.79,,Malformed", saved), "restoring",
    "an invalid hash falls back to the user's valid saved workplace");
  assert.equal(classify("#wp=37.79,,Malformed", "not json"), "onboarding");
  assert.equal(classify("#wp=37.79,-122.39,First&wp=,0,Last", saved), "restoring",
    "the last wp follows the late hash parser and valid saved data remains the fallback");
  assert.equal(classify("#wp=NaN,-122.39,Nope", JSON.stringify({ lat: 37.7, lon: "-122.4" })), "onboarding");
});

test("startup restore shell is selected before paint and failures remain actionable", () => {
  const head = TSRC.slice(0, TSRC.indexOf('<link rel="stylesheet"'));
  assert.match(head, /document\.documentElement\.dataset\.startup=classifyStartupState\(location\.hash,wp\)/,
    "the privacy-safe state word is set synchronously before CSS loads");
  assert.doesNotMatch(head, /dataset\.startup\s*=\s*[^;]*(?:lat|lon|label)/,
    "the root startup attribute never leaks a saved workplace into the DOM");
  assert.match(TSRC, /:root\[data-startup="restoring"\] #prompt \.ob-onboarding/);
  assert.match(TSRC, /:root\[data-startup="restoring"\] #prompt \.ob-restoring/);
  assert.match(TSRC, /id="startup-title">Opening your commute/);
  assert.match(TSRC, /id="startupretry"[^>]*>Retry/);
  assert.match(TSRC, /id="startupchange"[^>]*>Change workplace/);
  assert.match(TSRC, /document\.getElementById\("startupretry"\)\.onclick=retryStartupRestore/);
  assert.match(TSRC, /document\.getElementById\("startupchange"\)\.onclick=chooseStartupWorkplace/);

  const restore = templateFn("setWorkplace");
  assert.match(restore, /const startup=!!opts\.startup/);
  assert.match(restore, /if\(startup\)\{startupRetry=null;setStartupPrompt\("restoring"/);
  assert.match(restore, /if\(!r\.ok\)throw new Error/,
    "HTTP failures cannot masquerade as a usable commute payload");
  assert.match(restore, /startupRetry=\(\)=>setWorkplace\(lat,lon,label,Object\.assign\(\{\},opts,\{startup:true\}\)\)/);
  assert.match(restore, /setStartupPrompt\("error","We couldn’t open this commute/);
  assert.match(restore, /if\(myGen===GEN&&startup\)hideStartupPrompt\(\)/,
    "the shell leaves only after the restored route has painted");
});

function _daHelpers() {
  const defs = ["stableRouteKey", "routeChoiceKey", "bdFrag", "_daPick", "normalizeBD", "optDA", "optLegs", "optTotal",
    "optRead", "buildCompare", "hoverAltChipData"].map(templateFn).join("\n");
  const harness = `
"use strict";
let metric="b";
const REAL={}, VAR={};
const PRIMARY_KEY="__primary__";
${templateConst("ALT_CASING")}
const DEPARTAFTER=true;            /* exercise the depart-after branch */
function dominantLine(legs){let b=null,m=-1;(legs||[]).forEach(g=>{if(g.mode==="transit"&&g.name&&g.min>m){m=g.min;b=g.name;}});return b||"walk only";}
function primaryCasing(){return "#fff";}
${defs}
return {set metric(v){metric=v;}, get metric(){return metric;},
        stableRouteKey, bdFrag, normalizeBD, optRead, optLegs, optTotal, buildCompare,
        hoverAltChipData, ALT_CASING};`;
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
  assert.equal(H.optRead(prim).frag, 6,
    "bad-day data remains available while the headline shows Best-case");
  assert.equal(H.optRead(prim).badDayBase, 26,
    "bad-day stays anchored to the scheduled journey, not the Best-case headline");
  // Toggle to Typical — the SAME (stale) list must now resolve the typical journeys + per-route frag.
  H.metric = "r";
  assert.equal(H.optTotal(prim), 26, "primary typical total after toggle (no re-fetch)");
  assert.equal(H.optTotal(altK), 28, "alt K typical total after toggle");
  assert.equal(H.optRead(prim).head, 26, "optRead head = typical");
  assert.equal(H.optRead(prim).frag, 6, "per-route frag shown in Typical");
  assert.equal(H.optRead(altK).frag, 4, "alt K per-route frag in Typical");
  // legs flip to the active journey too (distinct geometry per metric). The primary strip's legs
  // are the GEOM legs (drawable: carry `pts`), NOT the text legs — see the drawable test below.
  assert.equal(H.optLegs(prim), d.typical.geom, "primary legs = typical journey GEOM legs in Typical");
});

test("depart-after: buildCompare carries server family and branch metadata onto every option", () => {
  const H = _daHelpers();
  const d = _daFixture();
  d.family = { key: "primary-corridor", name: "Aurora", sub: "primary corridor", lines: ["Aurora"], tags: [] };
  d.branch = { key: "primary-finish", name: "Walk finish", kind: "walk", lines: [] };
  d.alts[0].family = { key: "alternate-corridor", name: "Borealis", sub: "alternate corridor", lines: ["Borealis"], tags: ["backup"] };
  d.alts[0].branch = { key: "cedar-tail", name: "Transfer to Cedar", kind: "transit", lines: ["Cedar"] };

  const [primary, alt] = H.buildCompare(d);
  assert.equal(primary.family, d.family);
  assert.equal(primary.branch, d.branch);
  assert.equal(alt.family, d.alts[0].family);
  assert.equal(alt.branch, d.alts[0].branch);
});

test("structural alternative keys survive lightweight-to-pinned reordering and insertion", () => {
  const H=_daHelpers(),first=_daFixture(),second=_daFixture();
  const annotate=(alt,family,branch)=>Object.assign(alt,{family:{key:family},branch:{key:branch}});
  annotate(first.alts[0],"north","walk");annotate(first.alts[1],"south","tail");
  const inserted=annotate({line:"X",best:{total:25,legs:[{mode:"transit",name:"X",min:25}]},
    typical:{total:31,legs:[{mode:"transit",name:"X",min:31}]}},"west","transfer");
  annotate(second.alts[0],"north","walk");annotate(second.alts[1],"south","tail");
  second.alts=[second.alts[1],inserted,second.alts[0]];

  const keys=list=>Object.fromEntries(list.filter(option=>!option.isPrimary).map(option=>[option.line,option.key]));
  const light=keys(H.buildCompare(first)),pinned=keys(H.buildCompare(second));
  assert.equal(light.K,pinned.K);
  assert.equal(light.J,pinned.J);
  assert.match(light.K,/^struct:/);
  assert.doesNotMatch(pinned.X,/^alt/,
    "a late inserted structural route cannot steal a positional selection slot");
  assert.equal(H.stableRouteKey(null,null,"alt7"),"alt7","legacy metadata retains the positional fallback");
});

test("public choice keys keep same-family, same-branch exact choices distinct", () => {
  const H=_daHelpers(),d=_daFixture();
  d.choice_key="choice:primary";
  d.family={key:"north"};d.branch={key:"walk"};
  d.alts=d.alts.map((alt,index)=>Object.assign(alt,{
    choice_key:`choice:north-walk-${index}`,
    family:{key:"north"},branch:{key:"walk"},
  }));

  const list=H.buildCompare(d);
  assert.deepEqual(list.map(option=>option.key),[
    "choice:primary","choice:north-walk-0","choice:north-walk-1"],
    "the public structural key, not the broad family/branch group, is the selection identity");
  assert.equal(new Set(list.map(option=>option.key)).size,list.length,
    "every exact server option remains a distinct selectable compare row");
});

// --------------------------------------------------------------------------- //
// BUG 1 regression: the depart-after PRIMARY strip's legs must be DRAWABLE      //
// --------------------------------------------------------------------------- //
// The compare card showed the primary strip selected but drew NO route on the map, because the
// primary's da.best/da.typical used d.best.legs / d.typical.legs (the TEXT legs: {mode,line,min,wait}
// — NO `pts`), while alts correctly used geom (legs == geom, with `pts`). drawSelected draws
// optLegs(opt); text legs have no points -> nothing renders. The fix builds the primary from the
// GEOM legs (matching the alts + arrive-by), so optLegs(primary) returns legs that carry `pts`.
// This test FAILS on the pre-fix code (text legs lack `pts`) and PASSES after.
function _daFixtureWithPts() {
  // A depart-after /itinerary?pin=1 breakdown shaped like the real server response: best (p5) and
  // typical (p50) journeys each carry SEPARATE `legs` (text, NO pts) and `geom` (with pts).
  const txtBest = [{ mode: "walk", line: null, min: 2 },
                   { mode: "transit", line: "N", min: 16, wait: 4 }];
  const geomBest = [{ mode: "walk", name: null, min: 2, pts: [[37.70, -122.40], [37.71, -122.40]] },
                    { mode: "transit", name: "N", min: 16, wait: 4, tmode: "metro",
                      pts: [[37.71, -122.40], [37.78, -122.41]] }];
  const txtTyp = [{ mode: "walk", line: null, min: 3 },
                  { mode: "transit", line: "N", min: 18, wait: 5 }];
  const geomTyp = [{ mode: "walk", name: null, min: 3, pts: [[37.70, -122.40], [37.715, -122.40]] },
                   { mode: "transit", name: "N", min: 18, wait: 5, tmode: "metro",
                     pts: [[37.715, -122.40], [37.78, -122.41]] }];
  return {
    name: "Test NB", olat: 37.7, olon: -122.4,
    total: 26, xfers: 1, legs: txtTyp, geom: geomTyp,
    typical: { total: 26, xfers: 1, legs: txtTyp, geom: geomTyp },
    best: { total: 20, xfers: 1, legs: txtBest, geom: geomBest },
    frag: 6,
    alts: [
      { line: "K",
        best: { total: 22, legs: [{ mode: "transit", name: "K", min: 20, pts: [[37.70, -122.40], [37.79, -122.41]] }] },
        typical: { total: 28, legs: [{ mode: "transit", name: "K", min: 24, pts: [[37.70, -122.40], [37.79, -122.41]] }] },
        frag: 4 },
    ],
  };
}
function _legsDrawable(legs) {
  // A drawable leg list: non-empty, and every leg carries a non-empty `pts` polyline.
  return Array.isArray(legs) && legs.length > 0 &&
    legs.every((g) => Array.isArray(g.pts) && g.pts.length > 0);
}
test("BUG1 depart-after: primary strip legs are DRAWABLE (carry pts) in BOTH metric modes", () => {
  const H = _daHelpers();
  const d = _daFixtureWithPts();
  const prim = H.buildCompare(d)[0];
  for (const m of ["b", "r"]) {
    H.metric = m;
    const legs = H.optLegs(prim);
    assert.ok(_legsDrawable(legs),
      `primary legs must carry pts (drawable) in metric=${m}; got ` + JSON.stringify(legs));
  }
});
test("BUG1 depart-after: a transit alt strip is also drawable (unchanged by the fix)", () => {
  const H = _daHelpers();
  const d = _daFixtureWithPts();
  const altK = H.buildCompare(d)[1];
  for (const m of ["b", "r"]) {
    H.metric = m;
    assert.ok(_legsDrawable(H.optLegs(altK)), `alt legs drawable in metric=${m}`);
  }
});

test("compare list drops exact duplicate route rows", () => {
  const H = _daHelpers();
  const d = _daFixtureWithPts();
  d.alts.unshift({
    line: "duplicate primary",
    best: { total: d.best.total, legs: d.best.geom },
    typical: { total: d.typical.total, legs: d.typical.geom },
    frag: d.frag,
  });
  const list = H.buildCompare(d);
  assert.equal(list.length, 2, "primary duplicate alt should be suppressed");
  assert.equal(list[0].isPrimary, true, "primary remains first");
  assert.equal(list[1].line, "K", "real alt remains after duplicate suppression");
  assert.equal(list[1].identityColor, T_ALT_CASING[0], "remaining alts get contiguous colors");
});

test("compare list preserves text-identical routes from distinct authoritative families", () => {
  const H = _daHelpers();
  const d = _daFixtureWithPts();
  d.family = { key: "east-corridor", name: "Crosstown", sub: "primary corridor",
    lines: ["Crosstown"], tags: [] };
  d.branch = { key: "east-finish", name: "Walk finish", kind: "walk", lines: [] };
  d.alts.unshift({
    line: "same display journey, different corridor",
    best: { total: d.best.total, legs: d.best.geom },
    typical: { total: d.typical.total, legs: d.typical.geom },
    frag: d.frag,
    family: { key: "west-corridor", name: "Crosstown", sub: "alternate corridor",
      lines: ["Crosstown"], tags: [] },
    branch: { key: "west-finish", name: "Walk finish", kind: "walk", lines: [] },
  });

  const list = H.buildCompare(d);
  assert.equal(list.length, 3, "different server family keys must defeat the legacy text dedupe");
  assert.deepEqual(list.slice(0, 2).map((o) => o.family.key), ["east-corridor", "west-corridor"]);
});

function _routeFamilyHelpers() {
  const defs = [
    "uniq", "tlegs", "lname", "lineNames", "routeServices", "serviceNames",
    "routeGrouping", "displayOptions", "buildFamilies",
  ].map(templateFn).join("\n");
  const harness = `
"use strict";
let compareList=[];
function optLegs(o){return o.legs||[];}
function optRead(o){return {head:o.total||0};}
${defs}
return {
  setCompareList(v){compareList=v;},
  displayOptions, routeGrouping, buildFamilies,
};`;
  return new Function(harness)();
}

function _groupedOption({ key, line, total, familyKey, branchKey, branchName, branchKind, branchLines }) {
  return {
    key, line, total, isPrimary: key === "primary",
    legs: [{ mode: "transit", name: line, min: total }],
    family: { key: familyKey, name: "Aurora / Borealis", sub: "shared boarding corridor",
      lines: ["Aurora", "Borealis"], tags: ["2 services", "1 transfer"],
      services: [
        { key: "service-a", name: "Aurora", shown: true, branchKeys: [branchKey] },
        { key: "service-b", name: "Borealis", shown: false, branchKeys: [branchKey] },
      ] },
    branch: { key: branchKey, name: branchName, kind: branchKind, lines: branchLines,
      services: [{ key: "service-a", name: "Aurora" }, { key: "service-b", name: "Borealis" }],
      serviceKeys: ["service-a", "service-b"] },
  };
}

test("route families: client groups exclusively by server family and branch keys", () => {
  const H = _routeFamilyHelpers();
  const opts = [
    _groupedOption({ key: "primary", line: "Aurora", total: 20, familyKey: "northbound-corridor",
      branchKey: "walk-finish", branchName: "Walk after Aurora", branchKind: "walk", branchLines: [] }),
    _groupedOption({ key: "alt-a", line: "Borealis", total: 22, familyKey: "northbound-corridor",
      branchKey: "walk-finish", branchName: "Walk after Borealis", branchKind: "walk", branchLines: [] }),
    _groupedOption({ key: "alt-b", line: "Aurora", total: 24, familyKey: "northbound-corridor",
      branchKey: "harbor-tail", branchName: "Transfer to Harbor", branchKind: "transit", branchLines: ["Harbor"] }),
  ];
  H.setCompareList(opts);
  const families = H.buildFamilies();

  assert.equal(families.length, 1);
  assert.equal(families[0].key, "northbound-corridor");
  assert.equal(families[0].opts.length, 3);
  assert.deepEqual(families[0].branches.map((b) => b.key), ["walk-finish", "harbor-tail"]);
  assert.equal(families[0].branches[0].opts.length, 2, "matching server branch keys compact together");
  assert.equal(families[0].branches[1].meta.name, "Transfer to Harbor");
  assert.deepEqual(families[0].branches[1].meta.lines, ["Harbor"]);
});

test("route families: server labels, lines, and tags are preserved as presentation metadata", () => {
  const H = _routeFamilyHelpers();
  const option = _groupedOption({ key: "primary", line: "Aurora", total: 20,
    familyKey: "corridor-key", branchKey: "tail-key", branchName: "Transfer to Cedar",
    branchKind: "transit", branchLines: ["Cedar"] });
  const grouping = H.routeGrouping(option);

  assert.deepEqual(grouping.family, {
    key: "corridor-key", name: "Aurora / Borealis", sub: "shared boarding corridor",
    lines: ["Aurora", "Borealis"], tags: ["2 services", "1 transfer"],
    services: [
      { key: "service-a", name: "Aurora", shown: true, branchKeys: ["tail-key"] },
      { key: "service-b", name: "Borealis", shown: false, branchKeys: ["tail-key"] },
    ],
  });
  assert.deepEqual(grouping.branch, {
    key: "tail-key", name: "Transfer to Cedar", kind: "transit", lines: ["Cedar"],
    services: [{ key: "service-a", name: "Aurora" }, { key: "service-b", name: "Borealis" }],
    serviceKeys: ["service-a", "service-b"],
  });
});

test("route families: branch-qualified boarding services survive client grouping", () => {
  const H = _routeFamilyHelpers();
  const option = _groupedOption({ key: "primary", line: "Aurora", total: 20,
    familyKey: "corridor-key", branchKey: "tail-key", branchName: "Transfer to Cedar",
    branchKind: "transit", branchLines: ["Cedar"] });
  option.branch.services = [{ key: "service-b", name: "Borealis" }];
  option.branch.serviceKeys = ["service-b"];

  const grouping = H.routeGrouping(option);
  assert.deepEqual(grouping.family.services.map((service) => service.name), ["Aurora", "Borealis"]);
  assert.deepEqual(grouping.branch.services, [{ key: "service-b", name: "Borealis" }]);
  assert.deepEqual(grouping.branch.serviceKeys, ["service-b"]);
});

test("route families: identical line labels remain separate when server family keys differ", () => {
  const H = _routeFamilyHelpers();
  const east = _groupedOption({ key: "primary", line: "Crosstown", total: 20,
    familyKey: "eastbound", branchKey: "east-walk", branchName: "Walk finish",
    branchKind: "walk", branchLines: [] });
  const west = _groupedOption({ key: "alt-a", line: "Crosstown", total: 21,
    familyKey: "westbound", branchKey: "west-walk", branchName: "Walk finish",
    branchKind: "walk", branchLines: [] });
  H.setCompareList([east, west]);

  assert.deepEqual(H.buildFamilies().map((f) => f.key), ["eastbound", "westbound"]);
});

test("route families: missing metadata falls back to one unique family per route", () => {
  const H = _routeFamilyHelpers();
  const a = { key: "primary", line: "Crosstown", total: 20, isPrimary: true,
    legs: [{ mode: "transit", name: "Crosstown", min: 20 }] };
  const b = { key: "alt-a", line: "Crosstown", total: 21, isPrimary: false,
    legs: [{ mode: "transit", name: "Crosstown", min: 21 }] };
  H.setCompareList([a, b]);
  const families = H.buildFamilies();

  assert.equal(families.length, 2, "legacy routes must not be guessed into one corridor");
  assert.deepEqual(families.map((f) => f.key), ["legacy:primary", "legacy:alt-a"]);
  assert.equal(families[0].meta.name, "Crosstown");
});

test("route families: frontend performs no second-pass route suppression", () => {
  const H = _routeFamilyHelpers();
  const direct = { key: "primary", line: "Local", total: 20, isPrimary: true,
    legs: [{ mode: "transit", name: "Local", min: 20 }] };
  const longer = { key: "alt-a", line: "Local > Connector > Tail", total: 30, isPrimary: false,
    legs: ["Local", "Connector", "Tail"].map((name) => ({ mode: "transit", name, min: 5 })) };
  H.setCompareList([direct, longer]);

  assert.deepEqual(H.displayOptions(), [direct, longer],
    "the server owns structural dominance; every returned option stays renderable");
});

test("route families: removed line-specific classifier symbols cannot return", () => {
  const forbidden = [
    "MARKET_TRUNK", "MARKET_BRANCH_LINES", "isMarketTrunkLine", "familyIsMarket",
    "familyIs22", "familyIs5549", "isLowValueFeeder", "familyKeyFor", "tuneFamilyMeta",
    "familyMeta", "branchKeyFor", "branchName", "branchTransitSequence",
    "marketTrunkLeg", "marketFamilyLines", "marketBranchTrunks",
  ];
  forbidden.forEach((name) => assert.ok(!TSRC.includes(name), `removed client heuristic returned: ${name}`));
  ["fam_market", "fam_22", "fam_55_49"].forEach((key) =>
    assert.ok(!TSRC.includes(key), `line-specific family key returned: ${key}`));
});

// --------------------------------------------------------------------------- //
// BUG 3 regression: an "uncolored" cell must still be interactive             //
// --------------------------------------------------------------------------- //
// A user reported that hovering a cell that isn't lit up (above the Max-commute slider `thr`, OR
// genuinely unreachable) does nothing. The old code gated the hover/click/loadBreak paths on
// `v==null||v>thr` — so a cell DIMMED only by the slider (still reachable, has a real journey) was
// fully non-interactive. The fix gates interactivity ONLY on genuine unreachability (`v==null`):
// the slider controls heatmap COLORING, not whether a reachable cell can be inspected; an
// unreachable cell shows a clear "No transit route" breakdown instead of a silent "—".
// These are SOURCE-SCAN guards (the gates live inline in Leaflet event handlers, not as exported
// functions): they FAIL on the pre-fix template (which had the `v>thr` interactivity gates) and
// pass after. The server-side contract these rely on is in tests/test_api.py
// (test_itinerary_works_for_uncolored_cells).
test("BUG3: interactivity gates no longer block dimmed-but-reachable cells (no `v>thr` early-return)", () => {
  const tsrc = readFileSync(TEMPLATE_PATH, "utf8");
  // The three buggy gates were:
  //   loadBreak:  if(v==null||v>thr){setHTML("—");ready(null);return;}
  //   mouseover:  {const v=val(f.properties.id);if(v==null||v>thr)return;}
  //   click:      {const v=val(f.properties.id);if(v==null||v>thr)return;}
  // None of these `||v>thr` interactivity early-returns may survive (the heatmap `style()` keeps its
  // own `v>thr` test — that DIMS the cell and is correct, so we only forbid the INTERACTIVITY forms).
  assert.ok(!tsrc.includes('if(v==null||v>thr)return;'),
    "found a leftover `if(v==null||v>thr)return;` interactivity gate — a dimmed reachable cell would be inert");
  assert.ok(!tsrc.includes('if(v==null||v>thr){setHTML("—")'),
    "found the leftover loadBreak `v>thr` gate that showed a bare '—' for a reachable dimmed cell");
});
test("BUG3: an unreachable cell renders the clear no-route message (not a silent '—')", () => {
  const tsrc = readFileSync(TEMPLATE_PATH, "utf8");
  // bdHTML renders the user-facing no-route copy; loadBreak must route an unreachable (v==null) cell
  // INTO it (we render directly from TT — no /itinerary round-trip), so the hover shows the message.
  assert.match(tsrc, /No transit route within ~75 min\./,
    "the no-route user message must exist in the template");
  assert.match(tsrc, /if\(v==null\)\{setHTML\(bdHTML\(\{error:"no route"\}\)\)/,
    "loadBreak must render the no-route breakdown for an unreachable (v==null) cell");
});
test("BUG3: the heatmap style() still dims above-thr cells (slider behavior preserved)", () => {
  const tsrc = readFileSync(TEMPLATE_PATH, "utf8");
  // The Max-commute slider must STILL hide above-thr cells in the heatmap — only INTERACTIVITY was
  // decoupled. style() keeps its `v>thr` fillOpacity:0 test.
  assert.match(tsrc, /function style\(f\)\{[^}]*if\(v==null\|\|v>thr\)return\{fillOpacity:0/,
    "style() must keep dimming above-thr cells (the slider still controls coloring)");
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
  const defs = ["bdFrag", "summaryVM"].map(templateFn).join("\n");
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

test("summaryVM (depart-after): scheduled journey — no ~, chip shows in both modes", () => {
  const H = _summaryVM(true);
  // normalizeBD flattens the active journey onto d.total; raw best/typical pair stays for `other`.
  // The legacy `typical` JSON key is displayed to users as "scheduled" in planned depart-after.
  const dTyp = { total: 26, xfers: 1, best: { total: 20 }, typical: { total: 26 }, frag: 6,
    var: { frag: 5, stuck: 0.04 } };
  H.metric = "r";
  let vm = H.summaryVM(dTyp);
  assert.equal(vm.head, 26, "Scheduled head = active (flattened) total");
  assert.equal(vm.headLab, "scheduled");
  assert.equal(vm.headTilde, "", "depart-after scheduled is exact → no ~");
  assert.equal(vm.otherVal, 20, "other = best journey total");
  assert.equal(vm.otherLab, "best-case"); assert.equal(vm.otherTilde, "");
  assert.equal(vm.frag, 6, "primary per-route frag wins over cell overlay");
  assert.equal(vm.showSig, true, "depart-after chip shows in both modes");
  // Best-case: active flattened total is the best journey; other = scheduled (still no ~).
  const dBest = { total: 20, xfers: 2, best: { total: 20 }, typical: { total: 26 }, frag: 6,
    var: { frag: 5, stuck: 0.04 } };
  H.metric = "b";
  vm = H.summaryVM(dBest);
  assert.equal(vm.head, 20); assert.equal(vm.headLab, "best-case"); assert.equal(vm.headTilde, "");
  assert.equal(vm.otherVal, 26); assert.equal(vm.otherLab, "scheduled"); assert.equal(vm.otherTilde, "");
  assert.equal(vm.showSig, true);
});

// --------------------------------------------------------------------------- //
// C1 regression: /variance must land under DEPART-AFTER (realistic:{})         //
// --------------------------------------------------------------------------- //
// loadVariance used to store the payload only when `d.realistic` was non-empty — but the DEFAULT
// depart-after semantic serves {realistic:{}, variance:{...}}, so VAR/REAL never populated: no
// bad-day chips on hover, and a card pinned before the MC completed showed "loading alternatives…"
// forever (varianceReady() could never flip true). The gate now lives in the pure, sliceable
// predicate applyVariancePayload: EITHER map non-empty counts as "the MC landed".
function _varianceHelpers() {
  const defs = ["resetVarianceState", "setVarianceState", "varianceReady", "varianceFailed",
    "varianceSettled", "applyVariancePayload"].map(templateFn).join("\n");
  const harness = `
"use strict";
let REAL={}, VAR={}, GEN=1, varianceState={status:"idle",gen:0,error:""};
${defs}
resetVarianceState(GEN);
return {applyVariancePayload, varianceReady, varianceFailed, varianceSettled,
        ready(){return setVarianceState("ready",GEN);},
        get REAL(){return REAL;}, get VAR(){return VAR;}};`;
  return new Function(harness)();
}

test("C1: a depart-after payload (realistic:{}, variance populated) populates VAR + flips varianceReady", () => {
  const H = _varianceHelpers();
  assert.equal(H.varianceReady(), false, "nothing landed yet");
  const applied = H.applyVariancePayload({ realistic: {}, variance: { 12: { frag: 4, stuck: 0, alt: ["K"] } } });
  assert.equal(applied, true, "depart-after payload must be accepted");
  assert.deepEqual(H.REAL, {}, "REAL stays empty under depart-after (no realistic override)");
  assert.equal(H.VAR[12].frag, 4, "VAR populated from the variance map");
  H.ready();
  assert.equal(H.varianceReady(), true, "varianceReady flips → pinned cards stop saying 'loading alternatives…'");
});

test("C1: an arrive-by payload populates BOTH maps", () => {
  const H = _varianceHelpers();
  const applied = H.applyVariancePayload({ realistic: { 7: 31 }, variance: { 7: { frag: 6, stuck: 0.02 } } });
  assert.equal(applied, true);
  assert.equal(H.REAL[7], 31);
  assert.equal(H.VAR[7].frag, 6);
  H.ready();
  assert.equal(H.varianceReady(), true);
});

test("C1: a genuinely empty payload is rejected and leaves state untouched", () => {
  const H = _varianceHelpers();
  assert.equal(H.applyVariancePayload({ realistic: {}, variance: {} }), false);
  assert.equal(H.applyVariancePayload({}), false);
  assert.equal(H.applyVariancePayload(null), false);
  assert.deepEqual(H.REAL, {});
  assert.deepEqual(H.VAR, {});
  assert.equal(H.varianceReady(), false);
});

test("C1: loadVariance routes through the predicate and refreshes the drawn route when the MC lands", () => {
  const lv = templateFn("loadVariance");   // slices the async fn body (sig matches inside `async function`)
  assert.ok(!lv.includes("d.realistic&&Object.keys(d.realistic).length"),
    "stale realistic-only gate found — depart-after /variance would be dropped again");
  assert.ok(lv.includes("applyVariancePayload(d)"), "loadVariance must gate via applyVariancePayload");
  assert.ok(lv.includes("refreshDrawnRoute()"),
    "an open pinned card must refresh (gain its alts) when the MC lands");
});

function _varianceLifecycleHarness() {
  const defs=["createLRU","resetVarianceState","setVarianceState","varianceReady","varianceFailed",
    "varianceSettled","applyVariancePayload","failVariance","retryAfter"]
    .map(templateFn).join("\n")+"\n"+
    templateFn("loadVariance").replace(/^function /,"async function ");
  const harness=`
"use strict";
let GEN=1,DESTLL=[37.7,-122.4],REAL={},VAR={},varianceState={status:"idle",gen:0,error:""};
let routePin=7,pinFeature={properties:{id:7}},maxxfers="any",walkspeed="med";
const requests=[],sleeps=[],toasts=[],refreshes=[],queue=[];let deferredResolve=null;
${defs}
const BDCACHE=createLRU(32);BDCACHE.set(7,{_pin:false,_pinPending:true,geom:[{mode:"walk"}]},7);
function ridesParam(){return "";} function speedParam(){return "";}
function response(status,body,retry){return {status,ok:status>=200&&status<300,
  headers:{get(name){return name==="Retry-After"?(retry||null):null;}},json:async()=>body};}
function fetch(url){requests.push(url);const item=queue.shift();if(!item)return Promise.reject(new Error("missing response"));
  if(item.error)return Promise.reject(item.error);
  if(item.defer)return new Promise(resolve=>{deferredResolve=resolve;});
  return Promise.resolve(response(item.status,item.body,item.retry));}
function sleep(ms){sleeps.push(ms);return Promise.resolve();}
function toast(msg){toasts.push(msg);}
function aggregate(){refreshes.push("aggregate");} function redraw(){refreshes.push("redraw");}
function refreshOpenInfo(){refreshes.push("open");}
function refreshDrawnRoute(){const cached=BDCACHE.get(routePin);
  refreshes.push("draw:"+(cached?String(cached._pinPending):"missing"));}
resetVarianceState(GEN);
return {
  load:loadVariance,requests,sleeps,toasts,refreshes,
  enqueue(status,body,retry){queue.push({status,body,retry});},
  enqueueError(){queue.push({error:new Error("network down")});},
  defer(){queue.push({defer:true});},
  resolve(status,body,retry){deferredResolve(response(status,body,retry));},
  nextGeneration(){GEN++;resetVarianceState(GEN);},
  get state(){return Object.assign({},varianceState);},
  get REAL(){return REAL;},get VAR(){return VAR;},
  get pending(){const d=BDCACHE.get(7);return d&&d._pinPending;}
};`;
  return new Function(harness)();
}

test("variance lifecycle retries 429 via Retry-After, then fails and releases a pending pin", async () => {
  const H=_varianceLifecycleHarness();
  H.enqueue(429,null,"0.025");H.enqueue(500,null);
  await H.load();
  assert.deepEqual(H.requests.map(u=>u.split("?")[0]),["/variance","/variance"]);
  assert.deepEqual(H.sleeps,[25],"429 honors Retry-After before the sole retry");
  assert.equal(H.state.status,"failed");assert.equal(H.state.gen,1);
  assert.equal(H.pending,false,"terminal variance failure clears the lightweight pin's wait flag");
  assert.deepEqual(H.refreshes,["open","draw:false"],
    "failure re-renders degraded state and releases the normal structural-enrichment owner");
  assert.match(H.toasts.at(-1),/Bad-day estimates unavailable/);
});

test("variance lifecycle also honors Retry-After for a transient 503", async () => {
  const H=_varianceLifecycleHarness();
  H.enqueue(503,null,"0.04");H.enqueue(200,{realistic:{},variance:{7:{frag:2}}});
  await H.load();
  assert.deepEqual(H.sleeps,[40]);assert.equal(H.requests.length,2);
  assert.equal(H.state.status,"ready");
});

test("variance lifecycle exposes loading, commits ready on success, and preserves normal refresh", async () => {
  const H=_varianceLifecycleHarness();H.defer();
  const pending=H.load();
  assert.equal(H.state.status,"loading","state changes before the network response settles");
  H.resolve(200,{realistic:{},variance:{7:{frag:3,stuck:0,alt:["K"]}}});
  await pending;
  assert.equal(H.state.status,"ready");
  assert.equal(H.VAR[7].frag,3);
  assert.deepEqual(H.refreshes,["aggregate","redraw","open","draw:true"],
    "the established success paint and active-route refresh remain intact");
  assert.deepEqual(H.toasts,[]);
});

test("empty and thrown variance responses terminate as failed", async () => {
  const empty=_varianceLifecycleHarness();empty.enqueue(200,{realistic:{},variance:{}});await empty.load();
  assert.equal(empty.state.status,"failed","an empty payload cannot leave the lifecycle loading");
  const thrown=_varianceLifecycleHarness();thrown.enqueueError();await thrown.load();
  assert.equal(thrown.state.status,"failed","a fetch exception cannot leave the lifecycle loading");
});

test("a stale variance failure cannot mutate the next destination generation", async () => {
  const H=_varianceLifecycleHarness();H.defer();const old=H.load();
  assert.equal(H.state.status,"loading");
  H.nextGeneration();
  assert.deepEqual(H.state,{status:"idle",gen:2,error:""});
  H.resolve(500,null);await old;
  assert.deepEqual(H.state,{status:"idle",gen:2,error:""},
    "the old destination cannot publish a terminal state into the new generation");
  assert.equal(H.pending,true,"a stale failure cannot clear the new generation's cache state");
  assert.deepEqual(H.toasts,[]);assert.deepEqual(H.refreshes,[]);
  assert.match(templateFn("setWorkplace"),/const myGen=\+\+GEN;[\s\S]*resetVarianceState\(myGen\)/,
    "every new destination and settings recompute resets variance under its new generation");
});

function _failedVarianceUpgradeHarness() {
  const defs=["createLRU","varianceReady","varianceFailed","varianceSettled","routeRequestKey","pinRequestKey","upgradePinnedBreakdown"]
    .map(templateFn).join("\n");
  const harness=`
"use strict";
let GEN=1,varianceState={status:"failed",gen:1,error:"unavailable"},routePin=7;
let DESTLL=[37.7,-122.4],maxxfers="any",walkspeed="med",pinUpgradeSeq=0,pinUpgradeKey="",pinPrefetchKey="";
let pinFeature={properties:{id:7,n:"Pinned area"}},REAL={},VAR={};
const fetches=[],renders=[];let resolvePin;
${defs}
const BDCACHE=createLRU(32);BDCACHE.set(7,{_pin:false,_pinPending:false,geom:[{mode:"walk"}]},7);
function ridesParam(){return "";}function speedParam(){return "";}
function fetch(url){fetches.push(url);return new Promise(resolve=>{resolvePin=resolve;});}
function renderPin(_f,d){renders.push({pin:d._pin,pending:d._pinPending});}
const pinBody={scrollTop:0};function requestAnimationFrame(fn){fn();}
const f=pinFeature;
return {fetches,renders,run(){upgradePinnedBreakdown(f);},
  resolve(){resolvePin({ok:true,json:async()=>({geom:[{mode:"walk"}],alts:[{line:"planned"}]})});},
  cached(){return BDCACHE.get(7);}};`;
  return new Function(harness)();
}

test("failed variance still permits exactly one planned pin enrichment", async () => {
  const H=_failedVarianceUpgradeHarness();H.run();H.run();
  assert.equal(H.fetches.length,1,"the existing pinUpgradeKey remains the sole request owner");
  assert.match(H.fetches[0],/&pin=1/);
  H.resolve();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(H.cached()._pin,true);
  assert.equal(H.cached()._pinPending,false);
  assert.deepEqual(H.renders.at(-1),{pin:true,pending:false});
  const compare=templateFn("compareHTML");
  assert.match(compare,/Bad-day estimates are unavailable\. Route choices are still shown\./,
    "the inspector explains the degraded metrics without hiding structural choices");
});

function _touchPrefetchHarness() {
  const defs=["createLRU","routeRequestKey","pinRequestKey","cancelPinPrefetch","launchPinPrefetch",
    "schedulePinPrefetch","upgradePinnedBreakdown"].map(templateFn).join("\n");
  const harness=`
"use strict";
let GEN=1,EXACT=true,routePin=null,maxxfers="any",walkspeed="med",pinUpgradeSeq=0,pinUpgradeKey="";
let pinPrefetchTimer=null,pinPrefetchSeq=0,pinPrefetchKey="",pinPrefetchAbort=null,pinPrefetchId=null;
let DESTLL=[37.7,-122.4],REAL={},VAR={},pinFeature=null;
const TOUCH=true,PIN_PREFETCH_DELAY=180,fetches=[],renders=[],deferreds=[];
let timer=null;
const console={warn(){}};
const f={properties:{id:7,n:"Tapped area"}};
let touchFeature=f;
const touchPeek={classList:{contains(name){return name==="open";}}};
${defs}
const BDCACHE=createLRU(32);BDCACHE.set(7,{_pin:false,_pinPending:false,geom:[{mode:"walk"}]},7);
function varianceSettled(){return true;}function ridesParam(){return "";}function speedParam(){return "";}
function fetch(url,opts){fetches.push({url,opts});return new Promise((resolve,reject)=>{deferreds.push({resolve,reject});});}
function setTimeout(fn){timer=fn;return 1;}function clearTimeout(){timer=null;}
function renderPin(_f,d){renders.push({pin:d._pin,pending:d._pinPending});}
const pinBody={scrollTop:0};function requestAnimationFrame(fn){fn();}
return {schedule(){return schedulePinPrefetch(f);},promote(){routePin=7;pinFeature=f;return schedulePinPrefetch(f,true);},
  fire(){if(timer){const fn=timer;timer=null;fn();}},dismiss(){cancelPinPrefetch();},
  nextGeneration(){GEN++;},
  resolve(i=0){deferreds[i].resolve({ok:true,json:async()=>({geom:[{mode:"walk"}],alts:[{line:"planned"}]})});},
  reject(i=0){deferreds[i].reject(new Error("network"));},
  fetches,renders,cached(){return BDCACHE.get(7);},pendingTimer(){return !!timer;}};`;
  return new Function(harness)();
}

test("touch preview warms one bounded enriched route and promotes that same request on Inspect", async () => {
  const H=_touchPrefetchHarness();
  assert.equal(H.schedule(),true);
  assert.equal(H.fetches.length,0,"prefetch waits briefly so quick exploratory taps stay cheap");
  assert.equal(H.pendingTimer(),true);
  H.promote();
  assert.equal(H.fetches.length,1,"Inspect flushes the pending same-cell prefetch instead of starting another request");
  assert.match(H.fetches[0].url,/&pin=1/);
  H.schedule();
  assert.equal(H.fetches.length,1,"the selected cell owns at most one enriched request");
  H.resolve();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(H.cached()._pin,true);
  assert.deepEqual(H.renders.at(-1),{pin:true,pending:false},"a promoted prefetch upgrades the open inspector");
});

test("dismissing a touch preview cancels its delayed prefetch before it consumes a route request", () => {
  const H=_touchPrefetchHarness();H.schedule();H.dismiss();H.fire();
  assert.equal(H.fetches.length,0,"dismissal prevents background work for an abandoned cell");
});

test("a stale touch prefetch cannot upgrade a newer routing generation", async () => {
  const H=_touchPrefetchHarness();
  H.schedule();H.fire();
  H.nextGeneration();H.schedule();H.fire();
  assert.equal(H.fetches.length,2,"a new generation gets one new request after cancelling the old owner");
  H.resolve(0);await new Promise(resolve=>setImmediate(resolve));
  assert.equal(H.cached()._pin,false,"the cancelled generation cannot publish after its late response");
  H.resolve(1);await new Promise(resolve=>setImmediate(resolve));
  assert.equal(H.cached()._pin,true,"the current generation remains eligible to enrich the inspector");
});

test("a failed touch prefetch releases ownership so Inspect can make one retry", async () => {
  const H=_touchPrefetchHarness();H.schedule();H.fire();
  H.reject();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(H.cached()._pinPending,false,"failure clears the lightweight card's pending indicator");
  H.promote();
  assert.equal(H.fetches.length,2,"Inspect retries only after the failed owner has released its key");
  H.resolve(1);await new Promise(resolve=>setImmediate(resolve));
  assert.equal(H.cached()._pin,true);
});

// --------------------------------------------------------------------------- //
// Recommendation-first route choices + focused route drawing                   //
// --------------------------------------------------------------------------- //
// The route inspector has one native button per authoritative branch. The server-selected
// primary is the recommendation, and the selected exact route is drawn last at full strength
// over a small set of practical alternatives rendered as quiet ghosts.
function _drawCompareHelpers() {
  const defs = ["effectiveChoiceKey", "compactMapChoices", "drawCompareRoutes"].map(templateFn).join("\n");
  const harness = `
"use strict";
const PRIMARY_KEY="__primary__";
const COMPACT_MAP_ROUTE_LIMIT=4;
let routePin=7, DRAWN=null;
let selKey=PRIMARY_KEY,previewKey=null,showAllRoutes=false;
let compareList=[], CHOICES=[];
const draws=[];
const routeLayer={clearLayers(){draws.length=0;}};
function clearRoute(){routeLayer.clearLayers();DRAWN=null;}
function primaryCasing(){return "#fff";}
function routeChoices(){return CHOICES;}
function recommendedChoice(choices){return choices.find(c=>c.o.isPrimary)||choices[0]||null;}
function practicalChoices(choices,recommended){return choices.filter(c=>c!==recommended).slice(0,3);}
function featuredChoices(choices,recommended){return [recommended,...practicalChoices(choices,recommended)].filter(Boolean);}
function findOpt(key){return compareList.find(o=>o.key===key)||null;}
function optLegs(o){return o.legs;}
function optTotal(o){return o.total;}
function branchServiceRows(choice){return (choice&&choice.branch&&choice.branch.meta&&choice.branch.meta.services)||[];}
function serviceNames(meta){return (meta&&meta.services||[]).map(service=>service.name).filter(Boolean);}
function rebuildAvoidForOptions(){}
function drawOne(legs,color,total,badge,opts){draws.push({color,name:legs&&legs[0]&&legs[0].name,
  opacity:(opts&&opts.opacity!=null)?opts.opacity:1,
  labels:!!(opts&&opts.labels), badge:!!(opts&&opts.badge)});}
${defs}
return {draws, drawCompareRoutes,
  setChoices(c){CHOICES=c;compareList=c.map(x=>x.o);},
  setSelection(key){selKey=key;previewKey=null;},
  setPreview(key){previewKey=key;},
  setExpanded(value){showAllRoutes=value;},
  get DRAWN(){return DRAWN;}};`;
  return new Function(harness)();
}
function _choiceFixture() {
  const choice = (key, primary, color, total, family, branch) => ({
    o: { key, isPrimary: primary, identityColor: color, total,
         legs: [{ mode: "transit", name: key, min: total, pts: [[0, 0], [1, 1]] }] },
    r: { head: total, frag: 0 }, family: { key: family }, branch: { key: branch } });
  return [
    choice("__primary__", true, "#fff", 20, "famA", "brP"),
    choice("alt0", false, T_ALT_CASING[0], 24, "famA", "brT"),
    choice("alt1", false, T_ALT_CASING[1], 26, "famB", "brB"),
  ];
}

test("route focus: primary selection draws last/full while practical alternatives are .13 ghosts", () => {
  const H = _drawCompareHelpers();
  H.setChoices(_choiceFixture());
  H.drawCompareRoutes();
  assert.deepEqual(H.draws.map(d=>d.opacity), [.13, .13, 1]);
  assert.ok(H.draws.slice(0,-1).every(d=>!d.labels&&!d.badge), "ghosts carry no labels or badge");
  assert.equal(H.draws.at(-1).color, "#fff", "server primary is the selected full-strength route");
  assert.equal(H.draws.at(-1).labels, true);
  assert.equal(H.draws.at(-1).badge, true);
  assert.deepEqual(H.DRAWN, {id:7,multi:true,key:"__primary__",famKey:"famA",
    branchKey:"brP",ghostCount:2});
});

test("route focus: a previewed alternative becomes the sole full-strength last draw", () => {
  const H = _drawCompareHelpers();
  H.setChoices(_choiceFixture());
  H.setPreview("alt1");
  H.drawCompareRoutes();
  assert.deepEqual(H.draws.map(d=>d.opacity), [.13, .13, 1]);
  assert.equal(H.draws.at(-1).color, T_ALT_CASING[1]);
  assert.equal(H.DRAWN.key, "alt1");
  assert.equal(H.DRAWN.famKey, "famB");
  assert.equal(H.DRAWN.branchKey, "brB");
});

test("route focus: an Additional exact choice stays full-strength in compact and expanded map modes", () => {
  const H=_drawCompareHelpers();
  const choices=_choiceFixture();
  for(let i=2;i<5;i++)choices.push({
    o:{key:`alt${i}`,isPrimary:false,identityColor:T_ALT_CASING[i%T_ALT_CASING.length],total:28+i,
      legs:[{mode:"transit",name:`alt${i}`,min:28+i,pts:[[0,0],[1,1]]}]},
    r:{head:28+i,frag:0},family:{key:"famA"},branch:{key:"brT"},
  });
  H.setChoices(choices);H.setSelection("alt4");
  H.drawCompareRoutes();
  assert.equal(H.DRAWN.key,"alt4","a route retained under Additional remains selectable");
  assert.equal(H.draws.at(-1).opacity,1,"the selected Additional route is always drawn in full");
  assert.equal(H.draws.length,4,"compact mode retains the selection and recommendation within its four-route ceiling");

  H.setExpanded(true);H.drawCompareRoutes();
  assert.equal(H.draws.length,choices.length,"expanded map mode draws every exact choice");
  assert.equal(H.draws.at(-1).opacity,1,"expanded mode still leaves the selection full-strength");
});

test("route focus: five choices with an Additional selection still leave one route for Show all", () => {
  const H=_drawCompareHelpers();
  const choices=_choiceFixture();
  for(let i=2;i<4;i++)choices.push({
    o:{key:`alt${i}`,isPrimary:false,identityColor:T_ALT_CASING[i%T_ALT_CASING.length],total:28+i,
      legs:[{mode:"transit",name:`alt${i}`,min:28+i,pts:[[0,0],[1,1]]}]},
    r:{head:28+i,frag:0},family:{key:"famA"},branch:{key:"brT"},
  });
  H.setChoices(choices);H.setSelection("alt3");
  H.drawCompareRoutes();
  assert.equal(H.draws.length,4,"compact mode never appends a fifth route");
  assert.deepEqual(H.draws.map(draw=>draw.name),["__primary__","alt0","alt1","alt3"],
    "the selected route and recommendation keep the best remaining featured choices");
  assert.equal(H.DRAWN.key,"alt3","the Additional selection survives the compact cap");
  assert.equal(H.draws.at(-1).opacity,1);

  H.setExpanded(true);H.drawCompareRoutes();
  assert.equal(H.draws.length,5,"Show all reveals the one route omitted by compact mode");
});

function _mapRouteToggleHelpers() {
  const defs=["compactMapChoices","mapRouteToggleLabel","mapRouteToggleHTML"].map(templateFn).join("\n");
  const harness=`
"use strict";
const COMPACT_MAP_ROUTE_LIMIT=4;
let CHOICES=[],showAllRoutes=false;
function routeChoices(){return CHOICES;}
function practicalChoices(choices,recommended){return choices.filter(c=>c!==recommended).slice(0,3);}
function featuredChoices(choices,recommended){return [recommended,...practicalChoices(choices,recommended)].filter(Boolean);}
${defs}
return {
  html(choices,selectedKey){CHOICES=choices;const recommended=choices.find(c=>c.o.isPrimary)||choices[0]||null;
    const selected=choices.find(c=>c.o.key===selectedKey)||recommended;
    return mapRouteToggleHTML(choices,recommended,selected);},
  setExpanded(value){showAllRoutes=value;},
};`;
  return new Function(harness)();
}

test("map route toggle is hidden when compact mode already contains every exact choice", () => {
  const H=_mapRouteToggleHelpers(),choices=_choiceFixture();
  choices.push({o:{key:"alt2",isPrimary:false},r:{head:28,frag:0},family:{key:"famC"},branch:{key:"brC"}});
  assert.equal(H.html(choices,"alt2"),"","four exact routes need no no-op expansion control");

  choices.push({o:{key:"alt3",isPrimary:false},r:{head:30,frag:0},family:{key:"famD"},branch:{key:"brD"}});
  assert.match(H.html(choices,"alt3"),/>Show all 5 routes on map<\/button>/,
    "a selected Additional route still leaves one exact choice for expansion");
  H.setExpanded(true);
  assert.match(H.html(choices,"alt3"),/>Show featured routes on map<\/button>/);
});

function _routeDisclosureHelpers() {
  const defs = ["routeTitle", "representativeChoices", "practicalChoices", "featuredChoices", "moreRouteChoices", "familyRouteChoices",
    "choiceBoardingParts", "choiceBoardingContext", "boardingGroupLabel", "boardingHeadingHTML", "additionalChoicesHTML"]
    .map(templateFn).join("\n");
  const harness = `
"use strict";
let CHOICES=[];
function routeChoices(){return CHOICES;}
function recommendedChoice(choices){return choices.find(c=>c.o.isPrimary)||choices[0]||null;}
function optLegs(o){return o.legs||[];}
function displayStopName(name){return String(name||"").replace(/\\s+/g," ").trim();}
function routeRowHTML(c,_recommended,where){return \`<button class="route-choice \${where}" data-key="\${c.o.key}">\${c.o.key}</button>\`;}
function escapeHTML(s){return String(s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
${defs}
return {
  setChoices(v){CHOICES=v;},
  practicalChoices,featuredChoices,moreRouteChoices,familyRouteChoices,additionalChoicesHTML,boardingGroupLabel,boardingHeadingHTML,
};`;
  return new Function(harness)();
}

function _disclosureChoices() {
  return [
    ["__primary__",true,20,"family-a","branch-primary"],
    ["alt0",false,21,"family-a","branch-a"],
    ["alt1",false,22,"family-b","branch-b"],
    ["alt2",false,23,"family-c","branch-c"],
    ["alt3",false,24,"family-d","branch-d"],
    ["alt4",false,25,"family-e","branch-e"],
  ].map(([key,isPrimary,head,family,branch])=>({
    o:{key,isPrimary},r:{head,frag:0},
    family:{key:family,meta:{name:family}},branch:{key:branch,meta:{name:branch}},
  }));
}

test("route disclosure: featured and more keys are disjoint and exhaust every route choice", () => {
  const H=_routeDisclosureHelpers(),choices=_disclosureChoices();
  H.setChoices(choices);
  const recommended=choices[0];
  const featured=H.featuredChoices(choices,recommended).map(c=>c.o.key);
  const more=H.moreRouteChoices(choices,recommended).map(c=>c.o.key);

  assert.deepEqual(featured,["__primary__","alt0","alt1","alt2"]);
  assert.deepEqual(more,["alt3","alt4"]);
  assert.deepEqual(featured.filter(key=>more.includes(key)),[],"no route button may be duplicated");
  assert.deepEqual([...featured,...more].sort(),choices.map(c=>c.o.key).sort(),
    "featured + more is the exhaustive branch-choice union");
});

test("route disclosure: visually identical nearby finishes do not crowd the headline choices", () => {
  const H=_routeDisclosureHelpers();
  const choices=[
    ["primary",true,22,"36","transfer to 43 / 6"],
    ["walk-a",false,22,"36","walk after 36"],
    ["walk-b",false,23,"36","walk after 36"],
    ["walk-c",false,23,"36","walk after 36"],
    ["route-37",false,24,"37","transfer to N"],
    ["route-43",false,25,"6 / 43","walk after 6 / 43"],
  ].map(([key,isPrimary,head,familyName,branchName],index)=>({
    o:{key,isPrimary},r:{head,frag:0},
    family:{key:"family-"+index,meta:{name:familyName}},
    branch:{key:"branch-"+index,meta:{name:branchName}},
  }));
  H.setChoices(choices);
  const practical=H.practicalChoices(choices,choices[0]);
  assert.deepEqual(practical.map(c=>c.o.key),["walk-a","route-37","route-43"],
    "one representative walk finish leaves headline room for meaningfully different routes");
  assert.deepEqual(H.moreRouteChoices(choices,choices[0]).map(c=>c.o.key),["walk-b","walk-c"],
    "the exact sibling finishes remain available under More rather than being discarded");
});

test("route disclosure: one additional-choices disclosure groups every remaining exact route by structural boarding context", () => {
  const H=_routeDisclosureHelpers(),choices=_disclosureChoices();
  H.setChoices(choices);
  const recommended=choices[0],more=H.moreRouteChoices(choices,recommended);
  const html=H.additionalChoicesHTML(more,recommended);
  assert.deepEqual([...html.matchAll(/data-key="([^"]*)"/g)].map(match=>match[1]).sort(),
    more.map(choice=>choice.o.key).sort(),"all remaining exact route keys stay reachable");
  assert.deepEqual([...html.matchAll(/data-family="([^"]*)"/g)].map(match=>match[1]).sort(),
    more.map(choice=>choice.family.key).sort(),"each authoritative family has one visible group");
  assert.doesNotMatch(html,/<details|More finishes|More route choices/,
    "the opened additional list has no nested disclosure or repetitive finish label");
});

test("route disclosure: duplicate structural family names gain stop or direction context", () => {
  const H=_routeDisclosureHelpers();
  const choices=[
    ["a","alpha","Northbound", "A Street", "Downtown"],
    ["b","beta","Northbound", "B Street", "Uptown"],
  ].map(([key,family,name,board,toward])=>({o:{key,isPrimary:false,legs:[{mode:"transit",board:{name:board},toward}]},r:{head:20,frag:0},
    family:{key:family,meta:{name}},branch:{key:key,meta:{name:"Walk finish"}}}));
  const html=H.additionalChoicesHTML(choices,choices[0]);
  assert.match(html,/boarding-place">Board at A Street</);
  assert.match(html,/boarding-detail">Toward Downtown, Services: Northbound/);
  assert.match(html,/boarding-place">Board at B Street/);
  assert.match(html,/boarding-detail">Toward Uptown, Services: Northbound/);
});

test("route disclosure: one family is split by discovered boarding stop and direction", () => {
  const H=_routeDisclosureHelpers();
  const choices=[
    ["north-a","A Street","Downtown"],
    ["north-b","B Street","Uptown"],
  ].map(([key,board,toward])=>({o:{key,isPrimary:false,legs:[{mode:"transit",board:{name:board},toward}]},r:{head:20,frag:0},
    family:{key:"north",meta:{name:"Northbound"}},branch:{key,meta:{name:"Walk finish"}}}));
  const html=H.additionalChoicesHTML(choices,choices[0]);
  assert.equal((html.match(/data-family="north"/g)||[]).length,2,
    "a single service family never mixes separate boarding contexts in one outer group");
  assert.match(html,/boarding-place">Board at A Street/);
  assert.match(html,/boarding-detail">Toward Downtown, Services: Northbound/);
  assert.match(html,/boarding-place">Board at B Street/);
  assert.match(html,/boarding-detail">Toward Uptown, Services: Northbound/);
});

test("route disclosure: refresh preserves only the two native disclosure states", () => {
  const render=templateFn("renderPin"),toggle=templateFn("handleRouteDisclosureToggle");
  assert.match(TSRC,/document\.addEventListener\("toggle",handleRouteDisclosureToggle,true\)/);
  assert.ok(render.indexOf("captureRouteDisclosureState()")<render.indexOf("setPinHTML(pinHTML(d))"),
    "refresh snapshots native disclosure state before replacing pinbody markup");
  assert.ok(render.indexOf("restoreRouteDisclosureState()")>render.indexOf("setPinHTML(pinHTML(d))"),
    "the replacement restores native disclosure state immediately afterward");
  assert.match(toggle,/allRoutesOpen=details\.open/);
  assert.match(toggle,/directionsOpen=details\.open/);
  assert.doesNotMatch(TSRC,/expert-family|family-more|More finishes on/);
  assert.match(render,/restoreRouteFocusKey/);
  assert.match(render,/row\.focus\(\{preventScroll:true\}\)/);
});

test("route selection fallback survives rerender and is announced through the dedicated polite status node", () => {
  const render=templateFn("renderPin"),announce=templateFn("announceRouteSelection"),compare=templateFn("compareHTML");
  assert.match(render,/selectionReplaced=!selectionLocked\|\|!findChoice\(selKey\)/,
    "only an unlocked initial selection or a removed exact route may be replaced by enrichment");
  assert.match(render,/if\(selectionLocked&&!findChoice\(selKey\)\)selectionLocked=false/,
    "a user lock is released only when its exact route disappeared");
  assert.match(render,/pendingSelectionAnnouncement=`Route changed to /);
  assert.ok(render.indexOf("setPinHTML(pinHTML(d))")<render.indexOf("fallbackAnnouncement=pendingSelectionAnnouncement"),
    "fallback copy is published only after replacement inspector markup contains its live node");
  assert.match(render,/pinCard\.querySelector\("#route-selection-status"\)/);
  assert.match(compare,/id="route-selection-status" class="sr-only" role="status" aria-live="polite"/);
  assert.match(announce,/formatMinutes\(choice\.r\.head\)/);
});

test("route inspector exposes Choices and a requested Route Plan without making Map a peer view", () => {
  const compare=templateFn("compareHTML"),row=templateFn("routeRowHTML"),selected=templateFn("selectedRouteHTML");
  const pin=templateFn("pinHTML");
  assert.match(compare,/id="route-choices-panel"/);
  assert.match(selected,/id="route-plan-panel"/);
  assert.match(compare,/id="view-route-plan"/,
    "Choices keeps an explicit Route Plan request control");
  assert.match(TSRC,/aria-controls="route-plan-panel"/,
    "the Route Plan control identifies the region it exposes");
  assert.match(TSRC,/aria-expanded="\$\{[^}]*planOpen[^}]*\}"/,
    "the Route Plan control reports its requested open state");
  assert.doesNotMatch(pin,/data-inspector-view="map"|data-view="map"/,
    "Map is permanent context, never a third mutually-exclusive inspector view");
  assert.match(compare,/Recommended route/);
  assert.match(compare,/Good alternatives/);
  assert.match(compare,/See \$\{more\.length\} additional route choice/);
  assert.doesNotMatch(row,/routeActionsHTML/,"route rows must not contain inline directions");
  assert.match(selected,/<details class="selected-directions" id="route-directions"/);
  assert.match(selected,/Step-by-step directions/);
  assert.match(selected,/direction-sequence/);
  assert.match(selected,/<ol class="route-directions">/);
});

test("opening Additional materializes its deferred route rows without a card rerender", () => {
  const toggle=templateFn("handleRouteDisclosureToggle");
  assert.match(toggle,/if\(details\.open\)/);
  assert.match(toggle,/panel\.innerHTML=additionalChoicesHTML\(/);
  assert.match(toggle,/moreRouteChoices\(choices,recommended\)/);
});

function _lockRowHelpers() {
  const defs=["effectiveChoiceKey","lockRow"].map(templateFn).join("\n");
  const harness=`
"use strict";
let routePin=7,selKey="__primary__",previewKey=null,directionsOpen=true,selectionLocked=false,draws=0,marks=0,refreshes=0,announcements=0;
const valid=new Set(["__primary__","alt0","alt1"]);
function findChoice(key){return valid.has(key)?{o:{key}}:null;}
function drawSelected(){draws++;}
function markRouteChoices(){marks++;}
function refreshSelectedRoutePanel(){refreshes++;}
function announceRouteSelection(){announcements++;}
${defs}
return {
  lockRow,
  setState(selected,preview=null){selKey=selected;previewKey=preview;},
  get state(){return {selKey,previewKey,directionsOpen,selectionLocked,draws,marks,refreshes,announcements};},
};`;
  return new Function(harness)();
}

test("route locking reuses an already-previewed draw and redraws only a genuinely new key", () => {
  const H=_lockRowHelpers();
  H.setState("__primary__","alt1");
  H.lockRow("alt1");
  assert.deepEqual(H.state,{selKey:"alt1",previewKey:null,directionsOpen:true,selectionLocked:true,draws:0,marks:1,refreshes:1,announcements:1},
    "locking the currently previewed route keeps its draw, preserves open directions, and records an explicit user lock");

  H.lockRow("alt1");
  assert.equal(H.state.draws,0,"re-clicking the locked route stays a no-op");

  H.lockRow("alt0");
  assert.deepEqual(H.state,{selKey:"alt0",previewKey:null,directionsOpen:true,selectionLocked:true,draws:1,marks:2,refreshes:2,announcements:2},
    "changing route keys redraws, updates the Plan, preserves directions, and announces exactly once");
});

function _selectionEnrichmentHelpers() {
  const harness=`
"use strict";
let routePin=7,selKey="manual",previewKey=null,selectionLocked=true,metric="r";
let mapChoiceKey=null,recommendedChoiceKey=null,recommendedChoiceKeys={},compareList=[],restoreRouteFocusKey=null;
let pendingSelectionAnnouncement="",resetPinScroll=false,pendingFitId=null;
const PRIMARY_KEY="primary",pinBody={scrollTop:0},pinCard={querySelector(){return null;}};
const choices=[
  {o:{key:"map",isPrimary:true},r:{head:22}},
  {o:{key:"recommended",isPrimary:false},r:{head:22}},
  {o:{key:"manual",isPrimary:false},r:{head:24}},
];
function captureRouteDisclosureState(){} function buildCompare(){return choices.map(c=>c.o);}
function routeChoices(){return choices;} function findChoice(key){return choices.find(c=>c.o.key===key)||null;}
function recommendedChoice(){return findChoice(recommendedChoiceKey)||findChoice(mapChoiceKey)||choices[0];}
function routeTitle(choice){return choice.o.key;} function formatMinutes(value){return value+" min";}
function setPinHTML(){} function pinHTML(){return "";} function restoreRouteDisclosureState(){}
function maybeRoute(){} function requestAnimationFrame(fn){fn();} function fitToCompare(){}
${templateFn("renderPin")}
return {
  render(d){renderPin({properties:{id:7}},d);},
  setLocked(v){selectionLocked=v;},
  setMetric(v){metric=v;},
  get state(){return {selKey,mapChoiceKey,recommendedChoiceKey,selectionLocked};}
};`;
  return new Function(harness)();
}

test("manual route selection survives a later enriched server recommendation", () => {
  const H=_selectionEnrichmentHelpers();
  H.render({map_choice_key:"map",recommended_choice_key:"recommended"});
  assert.deepEqual(H.state,{selKey:"manual",mapChoiceKey:"map",recommendedChoiceKey:"recommended",selectionLocked:true},
    "the server may refine its recommendation without overriding an explicit choice");
  H.setLocked(false);
  H.render({map_choice_key:"map",recommended_choice_key:"recommended"});
  assert.equal(H.state.selKey,"recommended",
    "an untouched initial selection adopts the server recommendation when enrichment lands");
});

test("an untouched selection follows the recommendation for the active time mode", () => {
  const H=_selectionEnrichmentHelpers();
  H.setLocked(false);
  const payload={map_choice_key:"map",recommended_choice_key:"recommended",
    recommended_choice_keys:{r:"recommended",b:"map"}};
  H.render(payload);
  assert.equal(H.state.selKey,"recommended");
  H.setMetric("b");
  H.render(payload);
  assert.equal(H.state.selKey,"map");
});

function _routeChoiceHelpers() {
  const defs = ["choiceMatchesKey", "routeChoices", "recommendedChoice", "mapChoice", "routeServices", "serviceNames", "naturalList",
    "branchServiceRows", "formatMinutes", "legPhysicalMin", "legScheduleAllowanceMin", "optionWalk", "optionScheduleAllowance",
    "optionTransfers", "routeTitle", "representativeChoices", "routeReasonParts", "routeReason", "routeBadDay", "routeFactsHTML", "routeTradeoffsHTML",
    "practicalChoices", "featuredChoices", "moreRouteChoices", "displayStopName", "routeActionsHTML", "routeRowHTML"].map(templateFn).join("\n");
  const harness = `
"use strict";
let FAMILIES=[];
let selKey="__primary__",previewKey=null,mapChoiceKey=null,recommendedChoiceKey=null;
function buildFamilies(){return FAMILIES;}
function optLegs(o){return o.legs||[];}
function optTotal(o){return o.total;}
function tlegs(o){return optLegs(o).filter(g=>g&&g.mode==="transit"&&(g.name||g.line));}
function lname(g){return String((g&&g.name)||(g&&g.line)||"");}
function uniq(a){const out=[];(a||[]).forEach(x=>{if(x&&!out.includes(x))out.push(x);});return out;}
function escapeHTML(s){return String(s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function primaryCasing(){return "#fff";}
${defs}
return {setFamilies(f){FAMILIES=f;},setKeys(map,recommended){mapChoiceKey=map;recommendedChoiceKey=recommended;},routeChoices,recommendedChoice,mapChoice,branchServiceRows,
  practicalChoices,featuredChoices,moreRouteChoices,optionWalk,optionScheduleAllowance,routeTitle,routeActionsHTML,routeRowHTML};`;
  return new Function(harness)();
}

function _routeChoiceFamilies() {
  const x=(key,primary,total,line)=>({o:{key,isPrimary:primary,total,identityColor:"#fff",
    legs:[{mode:"transit",name:line,min:total,board:{name:"Fixture Platform"},
      toward:"Representative Terminal"}]},r:{head:total,frag:0,fragKnown:true,badDayBase:total}});
  const p=x("__primary__",true,27,"Aurora");
  const sibling=x("alt0",false,22,"Borealis");
  const tail=x("alt1",false,25,"Cedar");
  const other=x("alt2",false,24,"Delta");
  const famA={key:"north",meta:{name:"Aurora / Borealis",services:[]},opts:[p,sibling,tail],branches:[]};
  famA.branches=[
    {key:"walk",meta:{name:"Walk finish",services:[
      {key:"svc-a",name:"Aurora"},{key:"svc-b",name:"Borealis"},{key:"svc-c",name:"Comet"}]},opts:[p,sibling]},
    {key:"tail",meta:{name:"Transfer to Cedar",services:[]},opts:[tail]},
  ];
  const famB={key:"south",meta:{name:"Delta",services:[]},opts:[other],branches:[
    {key:"delta-walk",meta:{name:"Walk finish",services:[]},opts:[other]},
  ]};
  return [famA,famB];
}

test("route choices: server recommendation can differ from the map choice without hiding either exact route", () => {
  const H=_routeChoiceHelpers();H.setFamilies(_routeChoiceFamilies());
  H.setKeys("__primary__","alt0");
  const choices=H.routeChoices();
  assert.equal(choices.length,4,"non-representative exact options remain reachable in Additional choices");
  assert.ok(choices.some(choice=>choice.o.key==="alt0"),"exact sibling key survives client rendering");
  const recommended=H.recommendedChoice(choices);
  assert.equal(H.mapChoice(choices).o.key,"__primary__","the map keeps its own generating itinerary identity");
  assert.equal(recommended.o.key,"alt0","the server's practical recommendation is authoritative over map-primary order");
  const row=H.routeRowHTML(recommended,recommended,"recommended");
  assert.match(row,/^<button type="button" class="[^"]*route-choice/);
  assert.equal((row.match(/<button\b/g)||[]).length,1,"a generated route row has one native button");
  assert.doesNotMatch(row,/role="button"/);
  assert.doesNotMatch(row,/aria-label=/,
    "the native button's visible directions must remain its accessible content");
  assert.equal((row.match(/<\/button>/g)||[]).length,1);
});

test("route choices use compact labeled fact nodes instead of one dense punctuation-separated sentence", () => {
  const H=_routeChoiceHelpers();H.setFamilies(_routeChoiceFamilies());
  H.setKeys("__primary__","alt0");
  const recommended=H.recommendedChoice();
  const row=H.routeRowHTML(recommended,recommended,"recommended");
  assert.match(row,/data-choice-key="alt0"/);
  assert.deepEqual([...row.matchAll(/data-fact="([^"]+)"/g)].map(match=>match[1]),
    ["walk","transfers","bad-day"],
    "a choice carries the comparison facts users need before opening its Route Plan");
  assert.match(row,/route-fact-label">Walking/);
  assert.match(row,/route-fact-label">Transfers/);
  assert.match(row,/route-fact-label">Bad day/);
  assert.match(row,/data-route-reason/);
  assert.doesNotMatch(row,/ · /,
    "fact grouping is visual/semantic, not a brittle centered-dot prose run-on");
});

test("route choices distinguish unavailable bad-day data from a real zero-delay estimate", () => {
  const H=_routeChoiceHelpers();
  const families=_routeChoiceFamilies();
  const choice=families[0].opts[0];
  choice.r.frag=null;choice.r.fragKnown=false;
  H.setFamilies(families);
  let exact=H.routeChoices().find(item=>item.o.key==="__primary__");
  let row=H.routeRowHTML(exact,H.recommendedChoice(),"recommended");
  assert.match(row,/route-fact-value">Not available/);
  choice.r.frag=0;choice.r.fragKnown=true;
  exact=H.routeChoices().find(item=>item.o.key==="__primary__");
  row=H.routeRowHTML(exact,H.recommendedChoice(),"recommended");
  assert.match(row,/route-fact-value">27 min/);
});

test("route choices: featured rows use one representative per branch while Additional retains exact siblings", () => {
  const H=_routeChoiceHelpers();H.setFamilies(_routeChoiceFamilies());
  const choices=H.routeChoices(),recommended=H.recommendedChoice(choices);
  assert.deepEqual(H.featuredChoices(choices,recommended).map(choice=>choice.o.key),["__primary__","alt2","alt1"],
    "the compact top level remains one representative per practical branch");
  assert.deepEqual(H.moreRouteChoices(choices,recommended).map(choice=>choice.o.key),["alt0"],
    "the exact sibling is retained for Additional selection and map drawing");
});

test("route choices: branch-qualified service metadata produces a generic multi-service boarding set", () => {
  const H=_routeChoiceHelpers();H.setFamilies(_routeChoiceFamilies());
  const choice=H.routeChoices().find(c=>c.branch.key==="walk");
  assert.deepEqual(H.branchServiceRows(choice).map(s=>s.name),["Aurora","Borealis","Comet"]);
  const actions=H.routeActionsHTML(choice);
  assert.match(actions,/Board Aurora, Borealis, or Comet/);
  assert.doesNotMatch(actions,/Board Aurora ·/,"exact representative must not hide proven sibling services");
  assert.doesNotMatch(actions,/Representative Terminal/,
    "one representative headsign must not be advertised for every cataloged sibling service");
});

test("route choices: a walk-finish title does not repeat the exact discovered service catalog", () => {
  const H=_routeChoiceHelpers();
  assert.equal(H.routeTitle({
    family:{meta:{name:"Aurora / Borealis / Cedar"}},
    branch:{meta:{name:"walk after Aurora / Borealis / Cedar"}},
  }),"Aurora / Borealis / Cedar → walk to destination");
  assert.equal(H.routeTitle({
    family:{meta:{name:"Aurora / Borealis"}},
    branch:{meta:{name:"walk after Aurora"}},
  }),"Aurora / Borealis → walk after Aurora",
  "a branch backed by only part of the corridor catalog remains explicit");
});

test("route actions preserve a real same-service reboard", () => {
  const H=_routeChoiceHelpers();
  const choice={o:{legs:[{mode:"transit",name:"Aurora",min:3},
    {mode:"transit",name:"Aurora",min:4},{mode:"transit",name:"Borealis",min:5}]},
    branch:{meta:{services:[]}},family:{meta:{services:[]}}};
  const html=H.routeActionsHTML(choice);
  assert.match(html,/Board Aurora/);
  assert.match(html,/Reboard Aurora/);
  assert.match(html,/Transfer to Borealis/);
});

test("route actions include alight stops and name the destination of every walk", () => {
  const H=_routeChoiceHelpers();
  const choice={o:{legs:[
    {mode:"walk",min:4},
    {mode:"transit",name:"Aurora",min:3,board:{name:"First Platform"},
      alight:{name:"Transfer Landing& 1st"},toward:"North Terminal"},
    {mode:"walk",min:2},
    {mode:"transit",name:"Borealis",min:5,board:{name:"Second Platform"},
      alight:{name:"Workplace Stop"},toward:"South Terminal"},
    {mode:"walk",min:6},
  ]},branch:{meta:{services:[]}},family:{meta:{services:[]}}};
  const html=H.routeActionsHTML(choice);

  assert.match(html,/Walk 4 min to First Platform/,
    "the access walk names the first boarding place");
  assert.match(html,/Ride 3 min, get off at Transfer Landing &amp; 1st/,
    "a ride keeps its normalized alight instruction inside the same numbered action");
  assert.match(html,/Walk 2 min to Second Platform/,
    "a transfer walk names the next boarding place");
  assert.match(html,/Ride 5 min, get off at Workplace Stop/);
  assert.match(html,/Walk 6 min to your workplace/,
    "the final walk retains its destination");
  assert.equal((html.match(/class="route-step"/g)||[]).length,5,
    "alight detail must not inflate the numbered step count");
  assert.match(html,/<small class="route-detail">Ride 3 min, get off at Transfer Landing &amp; 1st<\/small>/,
    "alight copy is native text, not an aria-only duplicate");
});

test("route actions separate physical walking from schedule allowance", () => {
  const H=_routeChoiceHelpers();
  const choice={o:{legs:[
    {mode:"walk",min:7,physical_min:3,schedule_allowance_min:4},
    {mode:"transit",name:"Aurora",min:11,board:{name:"First Platform"}},
  ]},branch:{meta:{services:[]}},family:{meta:{services:[]}}};
  assert.equal(H.optionWalk(choice.o),3,"physical_min controls all displayed walking totals");
  assert.equal(H.optionScheduleAllowance(choice.o),4,"pre-boarding allowance is preserved separately");
  const html=H.routeActionsHTML(choice);
  assert.match(html,/Walk 3 min to First Platform/);
  assert.match(html,/Allow 4 min before boarding/);
  assert.doesNotMatch(html,/Walk 7 min/);
});

test("minute formatter keeps fractional route truth readable without changing ranking values", () => {
  const format=new Function(`"use strict";${templateFn("formatMinutes")};return formatMinutes;`)();
  assert.equal(format(12),"12 min");
  assert.equal(format(12.04),"12 min","near-whole values stay compact");
  assert.equal(format(12.25),"12.3 min");
  assert.equal(format(.6),"<1 min");
  assert.equal(format("bad"),"—");
  const row=templateFn("routeRowHTML"),actions=templateFn("routeActionsHTML"),selected=templateFn("selectedRouteHTML"),pin=templateFn("pinHTML");
  [row,actions,selected,pin].forEach(source=>assert.match(source,/formatMinutes\(/));
});

// --------------------------------------------------------------------------- //
// C5 regression: hover "also serves" chip colors == the deduped pinned colors  //
// --------------------------------------------------------------------------- //
// The hover chips used to color by the RAW d.alts index while the pinned compare list dedups
// (walk-only filter + duplicate suppression) and reassigns CONTIGUOUS ALT_CASING slots — so a
// chip's dot could show a different color than the same route's pinned dot / drawn casing.
// hoverAltChipData now derives the chips from the same buildCompare output; structural dominance
// is already reflected in the server's returned alternative set.
test("C5: hover chips are deduped and take the compare list's contiguous slot colors", () => {
  const H = _daHelpers();
  const d = _daFixtureWithPts();
  // A duplicate-of-primary alt sits FIRST in d.alts: raw-index coloring would give it slot 0 and
  // push the real alt K to slot 1, while the pinned card dedups it away and gives K slot 0.
  d.alts.unshift({
    line: "duplicate primary",
    best: { total: d.best.total, legs: d.best.geom },
    typical: { total: d.typical.total, legs: d.typical.geom },
    frag: d.frag,
  });
  H.metric = "b";
  const chips = H.hoverAltChipData(d);
  assert.equal(chips.length, 1, "the duplicate-of-primary alt is deduped out of the hover chips too");
  assert.equal(chips[0].line, "K");
  assert.equal(chips[0].color, H.ALT_CASING[0],
    "chip dot = the deduped slot color (== pinned strip/family dot + drawn casing), not the raw index");
  assert.equal(chips[0].min, 22, "chip minutes follow the selected metric read (best-case)");
  H.metric = "r";
  assert.equal(H.hoverAltChipData(d)[0].min, 28, "chip minutes re-pick under Typical");
});

test("C5: hover chips list is empty for an error/absent breakdown (no crash)", () => {
  const H = _daHelpers();
  assert.deepEqual(H.hoverAltChipData(null), []);
  assert.deepEqual(H.hoverAltChipData({ error: "no route" }), []);
});

// --------------------------------------------------------------------------- //
// C6 regression: bad-day absolute anchors on the typical/scheduled total       //
// --------------------------------------------------------------------------- //
// badAbs was head+frag, but `head` follows the metric selector — in Best-case mode the bad-day
// absolute understated by (typical − best). frag is p90−p50 of the committed MC, so the absolute
// is typical.total + frag regardless of the selected metric.
function _summaryHTMLHelpers(departafter) {
  const defs = ["escapeHTML", "fragChip", "bdFrag", "summaryVM", "summaryHTML"].map(templateFn).join("\n");
  const harness = `
"use strict";
let metric="b";
const DEPARTAFTER=${departafter ? "true" : "false"};
${defs}
return {set metric(v){metric=v;}, summaryHTML};`;
  return new Function(harness)();
}

test("C6: depart-after bad-day absolute = typical.total + frag in BOTH metric modes", () => {
  const H = _summaryHTMLHelpers(true);
  const base = { name: "X", xfers: 1, best: { total: 20 }, typical: { total: 26 }, frag: 6,
    var: { frag: 5, stuck: 0 } };
  H.metric = "b";                                       // Best-case selected: head=20, but bad-day
  let html = H.summaryHTML(Object.assign({}, base, { total: 20 }));  // (normalized best shape)
  assert.match(html, /bad-day <b[^>]*>32m<\/b>/, "26 (typical) + 6 (frag) = 32, not 20+6=26");
  assert.ok(!/bad-day <b[^>]*>26m/.test(html), "must not anchor on the best-case headline");
  H.metric = "r";                                       // Typical selected: unchanged (26+6)
  html = H.summaryHTML(Object.assign({}, base, { total: 26 }));
  assert.match(html, /bad-day <b[^>]*>32m<\/b>/);
});

// --------------------------------------------------------------------------- //
// C7 regression: the STATIC metric-help tip is the arrive-by copy              //
// --------------------------------------------------------------------------- //
// The static HTML buttons say Realistic/Best-case (arrive-by labels); applyCfg only rewrites the
// labels + tip under DEPARTAFTER. The static data-tip used to carry depart-after "Scheduled" copy,
// contradicting the visible buttons on an arrive-by boot.
test("C7: static metric tip matches the static Realistic/Best-case labels; DEPARTAFTER override intact", () => {
  const m = TSRC.match(/aria-label="What is Time estimate\?"[^>]*data-tip="([^"]*)"/);
  assert.ok(m, "the metric help button with a data-tip must exist");
  assert.ok(!/Scheduled/.test(m[1]), "static tip must not carry the depart-after 'Scheduled' copy");
  assert.match(m[1], /Best-case/);
  assert.match(m[1], /Realistic/);
  // the runtime DEPARTAFTER override still swaps in the Scheduled copy
  assert.ok(TSRC.includes('metricHelp.dataset.tip='), "applyCfg must still override the tip under DEPARTAFTER");
  assert.match(TSRC, /Scheduled is the normal scheduled trip/);
});

// --------------------------------------------------------------------------- //
// C3 regression: the pinned card's × cannot scroll away                        //
// --------------------------------------------------------------------------- //
// #pincard used to be the scroller (overflow-y:auto) with the × absolutely positioned INSIDE it,
// so scrolling a tall compare card carried the close button off-screen. Now the card is a
// non-scrolling shell (flex column, overflow clipped) and only #pinbody scrolls.
test("responsive inspector keeps an anchored shell and independently scrollable Choices and Plan content", () => {
  // Anchor at a standalone root selector: an earlier desktop-state rule also contains
  // `#pincard{...}` as the tail of `body.route-pinned ... #pincard`, which is not the shell rule.
  const cardRule = TSRC.match(/\n\s{2}#pincard\{[^}]*\}/)[0];
  assert.match(cardRule, /position:fixed/,
    "the pinned card must stay viewport-anchored when mobile controls change document scrollY");
  assert.ok(!/overflow-y:auto/.test(cardRule),
    "#pincard itself must not scroll — the absolutely-positioned × would scroll away with it");
  assert.match(TSRC, /#pincard\.open\{display:flex;flex-direction:column\}/);
  assert.match(TSRC, /#pinbody\{[^}]*overflow:hidden/,
    "the shell body remains non-scrolling so the close control never scrolls away");
  assert.match(TSRC, /\.route-choices-pane[^}]*overflow-y:auto/,
    "Choices retains its own scroll position when Plan opens or changes location");
  assert.match(TSRC, /\.route-plan-pane[^}]*overflow-y:auto/,
    "long Plan content scrolls internally rather than expanding over Choices");
  assert.match(TSRC, /body\.route-pinned #legend\{display:none\}/,
    "route inspection must not compete with the broad map legend");
  assert.ok(TSRC.includes('document.body.classList.add("route-pinned")'));
  assert.doesNotMatch(TSRC,/pin-map-view|pin-plan-view/,
    "the retired mutually-exclusive Plan/Map body-class model cannot reappear");
  assert.ok(TSRC.includes("resetPinScroll=true"),
    "a new pin must arm a reset through the async placeholder/final-card replacement");
  assert.match(TSRC, /requestAnimationFrame\(\(\)=>\{\s*if\(routePin===f\.properties\.id\)pinBody\.scrollTop=0;/,
    "the visible final card must reset again after browser scroll restoration/layout");
});

test("touch route inspection uses first-tap preview and same-cell second-tap inspect hooks", () => {
  const open = templateFn("openTouchPreview");
  const inspect = templateFn("inspectTouchPreview");
  assert.match(open, /touchFeature&&touchFeature\.properties\.id===id/,
    "same-cell detection is the second-tap boundary");
  assert.match(open, /inspectTouchPreview\(\);return;/,
    "a second tap on the previewed cell advances to inspection");
  assert.match(open, /showCellFocus\(f\)/,
    "touch preview owns the transient selected-cell focus treatment");
  assert.match(open, /if\(routePin!=null\)closePin\(\)/,
    "a new preview leaves the old pinned/adjusting state before collapsing controls");
  assert.match(open, /panelEl\.classList\.remove\("open"\)/,
    "a first-tap preview collapses expanded controls instead of stacking two mobile sheets");
  assert.match(inspect, /openPin\(f,l\)/,
    "Inspect routes must enter the full pinned inspector");
  assert.match(TSRC, /if\(touchInteraction\(ev\)\)openTouchPreview\(f,l\);else openPin\(f,l\);/,
    "a touch/pen cell interaction previews first while a mouse click pins directly");
  assert.match(TSRC, /id="peekinspect"/,
    "the preview exposes an explicit Inspect routes action");
  assert.match(TSRC,/Show all \$\{count\} routes on map/,
    "collapsed-map action names the exact routes it will reveal");
  assert.match(templateFn("compareHTML"),/mapRouteToggleHTML\(choices,recommended,selected\)/,
    "the map expansion control is omitted when compact mode already contains every route");
  assert.match(TSRC,/Show featured routes on map/,
    "expanded-map action names the compact route set it restores");
});

test("input modality only chooses preview affordances; viewport capability owns inspector layout", () => {
  const source=TSRC.slice(TSRC.indexOf("const TOUCH="),TSRC.indexOf("const REDUCE_MOTION"));
  assert.match(source,/any-pointer: coarse/);
  assert.match(source,/any-hover: hover/);
  assert.doesNotMatch(source,/ontouchstart|maxTouchPoints|max-width/,
    "viewport width and mere touch support cannot force a mouse user into two-tap inspection");
  const classify=defaultTouch=>new Function(`"use strict";const TOUCH=${defaultTouch};${templateFn("touchInteraction")};return touchInteraction;`)();
  const desktop=classify(false);
  assert.equal(desktop({originalEvent:{pointerType:"mouse"}}),false);
  assert.equal(desktop({originalEvent:{pointerType:"touch"}}),true,
    "a hybrid device still gets the touch preview for an actual finger tap");
  assert.equal(desktop({originalEvent:{pointerType:"pen"}}),true);
  assert.equal(desktop({originalEvent:{}}),false);
  assert.equal(classify(true)({originalEvent:{}}),true,
    "non-PointerEvent touch browsers retain the coarse/no-hover fallback");

  const inspectorSource=TSRC;
  assert.match(inspectorSource,/(?:wide-sidecar|single-card|bottom-sheet)/,
    "the controller recognizes viewport-owned wide, single-card, and bottom-sheet capabilities");
  assert.match(inspectorSource,/(?:innerWidth|visualViewport\.width|getBoundingClientRect\(\))/,
    "layout capability is computed from rendered/viewport space");
});

test("responsive inspector publishes one semantic state model instead of composing body view classes", () => {
  for(const attribute of [
    "layout-capability", "surface", "plan-open", "presentation",
    "sheet-content", "sheet-snap", "dragging",
  ]) {
    assert.match(TSRC,new RegExp(`data-${attribute}`),
      `#pincard must expose its ${attribute} state for one inspectable controller contract`);
  }
  assert.doesNotMatch(TSRC,/pin-map-view|pin-plan-view/,
    "the old Choices / Plan / Map class matrix must not remain as a second state authority");
  assert.doesNotMatch(TSRC,/@media \(min-width:1240px\)[\s\S]{0,900}grid-template-columns:minmax\(310px,\.9fr\) minmax\(370px,1\.1fr\)/,
    "a wide viewport must not force a permanent equal-weight two-pane workspace");
});

test("new pins start in Choices with Plan closed, while capability changes preserve requested Plan intent", () => {
  const open=templateFn("openPin");
  assert.match(open,/(?:planOpen\s*=\s*false|planOpen:false|planOpen\s*:\s*false)/,
    "each new pin resets transient presentation to a closed Route Plan");
  const controller=TSRC;
  assert.match(controller,/planOpen/,
    "Plan request state is explicit rather than inferred from a particular DOM placement");
  assert.match(controller,/(?:wide-sidecar|single-card|bottom-sheet)/,
    "the same request is rendered as sidecar, inline tray, or mobile sheet by capability");
  assert.doesNotMatch(controller,/(?:resize|visualViewport)[\s\S]{0,220}planOpen\s*=\s*false/,
    "resizing between sidecar and inline-tray capabilities never silently closes Plan");
});

test("map focus and Settings replace active route chrome while retaining route return state", () => {
  const controller=TSRC;
  assert.match(controller,/presentation[\s\S]{0,180}map-focus|map-focus[\s\S]{0,180}presentation/,
    "map focus is a presentation state, not an unpin operation");
  assert.match(controller,/settingsReturn/,
    "opening Settings records a return snapshot rather than destroying the inspector state");
  assert.match(controller,/surface[\s\S]{0,80}(?:routes|settings)|(?:routes|settings)[\s\S]{0,80}surface/,
    "exactly one primary surface is selected at a time");
  assert.doesNotMatch(controller,/(?:routePin\s*=\s*null|closePin\(\))[\s\S]{0,120}(?:map-focus|settingsReturn)/,
    "neither Map focus nor Settings may clear the selected pin");
});

test("compact inspector is one bottom-sheet controller with Choices, Plan, and Settings content", () => {
  const controller=TSRC;
  assert.match(controller,/sheetSnap[\s\S]{0,180}(?:peek|browse|expanded)|(?:peek|browse|expanded)[\s\S]{0,180}sheetSnap/,
    "compact routes have Peek, Browse, and Expanded snap states");
  assert.match(controller,/sheetContent[\s\S]{0,180}(?:choices|plan|settings)|(?:choices|plan|settings)[\s\S]{0,180}sheetContent/,
    "Choices, Plan, and Settings are content in the same sheet rather than peer pages");
  assert.match(TSRC,/(?:View choices|Show map|Expand)/,
    "every drag-only snap has a visible keyboard-operable alternative");
  assert.doesNotMatch(TSRC,/data-inspector-view="map"|data-view="map"/,
    "Show map collapses the sheet; it does not select an obsolete Map content screen");
});

test("sheet drag uses the handle only and handles threshold, velocity, cancellation, scroll boundaries, and reduced motion", () => {
  const controller=TSRC;
  assert.match(controller,/setPointerCapture\(/,
    "a handle drag owns its pointer until release");
  assert.match(controller,/pointercancel/,
    "a cancelled gesture restores a settled sheet state");
  assert.match(controller,/(?:dragThreshold|DRAG_THRESHOLD|clickSuppress|suppress(?:NextHandle)?Click)/,
    "a real drag suppresses the synthetic click that would otherwise toggle the sheet again");
  assert.match(controller,/(?:velocity|VELOCITY)/,
    "release snapping considers flick velocity as well as drag distance");
  assert.match(controller,/(?:clamp|Math\.min\(.*Math\.max)/,
    "live drag translation remains within valid Peek/Expanded bounds");
  assert.match(controller,/(?:scrollTop|closest\([^)]*(?:route-choices|route-plan|sheet))/,
    "ordinary Choices and Plan scrolling is not promoted into a sheet drag");
  assert.match(controller,/REDUCE_MOTION/,
    "reduced motion settles sheet transitions without continued animation");
  assert.match(controller,/(?:Enter|Space|key === " "|key===" ")/,
    "the handle has a keyboard path between Peek and Browse");
});

test("Route Plan and sheet controls retain accessible expanded/controlled relationships", () => {
  assert.match(TSRC,/aria-controls="route-plan-panel"/,
    "the Route Plan toggle names its controlled plan region");
  assert.match(TSRC,/aria-expanded="\$\{[^}]*planOpen[^}]*\}"/,
    "the Plan toggle exposes open/closed state to assistive technology");
  assert.match(TSRC,/(?:aria-hidden|inert)[\s\S]{0,180}(?:sheetSnap|peek)|(?:sheetSnap|peek)[\s\S]{0,180}(?:aria-hidden|inert)/,
    "Peek cannot leave off-screen route rows in the keyboard focus order");
});

test("last pointerdown modality keeps hybrid pointer interactions faithful", () => {
  assert.match(TSRC,/document\.addEventListener\("pointerdown",rememberPointerModality,true\)/);
  const H=new Function(
    '"use strict";let lastPointerType="";const TOUCH=false;\n'+
    templateFn("rememberPointerModality")+"\n"+templateFn("touchInteraction")+
    "\nreturn {rememberPointerModality,touchInteraction};"
  )();
  H.rememberPointerModality({pointerType:"touch"});
  assert.equal(H.touchInteraction({originalEvent:{}}),true,
    "a compatibility click after a finger down retains first-tap preview");
  H.rememberPointerModality({pointerType:"mouse"});
  assert.equal(H.touchInteraction({originalEvent:{}}),false,
    "a mouse click on the same hybrid device opens the inspector directly");
  H.rememberPointerModality({pointerType:"pen"});
  assert.equal(H.touchInteraction({originalEvent:{}}),true,
    "pen input shares the touch preview path");
});

test("mobile preview stays nonmodal while route and informational regions expose coherent accessibility contracts", () => {
  assert.match(TSRC,/<div id="busy" role="status" aria-live="polite" aria-atomic="true">/);
  assert.match(TSRC,/<div id="toast" role="status" aria-live="polite" aria-atomic="true"><\/div>/);
  assert.match(TSRC,/<aside id="touchpeek" role="region"[^>]*aria-live="polite"[^>]*aria-atomic="true"/);
  assert.match(TSRC,/<aside id="pincard" role="region" aria-label="Route inspector" tabindex="-1">/);
  assert.match(TSRC,/<div id="howmodal" class="modal" role="dialog" aria-modal="true" aria-labelledby="howtitle">/);
  assert.match(TSRC,/<div id="aboutmodal" class="modal" role="dialog" aria-modal="true" aria-labelledby="abouttitle">/);
  assert.doesNotMatch(templateFn("openTouchPreview"),/\.focus\(/,
    "the nonmodal first-tap preview announces itself without stealing map focus");
  assert.match(templateFn("inspectTouchPreview"),/pinCard\.focus\(\{preventScroll:true\}\)/,
    "the explicit Inspect transition enters the full route region intentionally");
  assert.match(TSRC,/function trapModalTab\(e\)/,
    "informational dialogs keep Tab within their own controls");
  assert.match(TSRC,/#touchpeek\{[^}]*overscroll-behavior:contain/);
  assert.match(TSRC,/\.modal \.card\{[^}]*overscroll-behavior:contain/);
  assert.match(TSRC,/<button class="info-link" id="howlink" type="button">How it works<\/button>/);
  assert.doesNotMatch(TSRC,/\.help:hover,\.help:focus-visible\{[^}]*outline:none/,
    "the compact help control keeps the global focus-visible ring");
});

test("document language, skip path, focus treatment, and touch handling are explicit", () => {
  assert.match(TSRC,/<html lang="en">/);
  assert.match(TSRC,/<a class="skip-link" href="#panel">Skip to commute controls<\/a>/);
  assert.match(TSRC,/<aside id="panel" aria-label="Commute controls" tabindex="-1">/);
  assert.match(TSRC,/\.skip-link:focus-visible\{transform:translateY\(0\)\}/);
  assert.match(TSRC,/function revealControlsForSkip\(\)[\s\S]*document\.body\.classList\.add\("route-adjusting"\)[\s\S]*document\.getElementById\("panel"\)\.focus\(\{preventScroll:true\}\)/);
  assert.match(TSRC,/document\.querySelector\("\.skip-link"\)\.addEventListener\("click",e=>\{e\.preventDefault\(\);revealControlsForSkip\(\);\}\)/);
  assert.match(TSRC,/button:focus-visible,a:focus-visible,input:focus-visible,summary:focus-visible,\[tabindex\]:focus-visible/);
  assert.match(TSRC,/button,a,input\[type=range\],summary\{touch-action:manipulation\}/);
});

test("dismissing the mobile preview never leaves focus in its hidden sheet", () => {
  const closePreview=templateFn("closeTouchPreview");
  assert.match(closePreview,/const restoreMapFocus=clearDraw&&routePin==null&&touchPeek\.contains\(document\.activeElement\)/,
    "only a true preview dismissal needs a focus return");
  assert.match(closePreview,/if\(restoreMapFocus\)requestAnimationFrame\(\(\)=>map\.getContainer\(\)\.focus\?\.\(\{preventScroll:true\}\)\)/,
    "a preview control that closes its own sheet returns focus to the map");
});

test("Escape is consumed by an open autocomplete before route-level Escape handling", () => {
  const consume=new Function(`"use strict";${templateFn("consumeAutocompleteEscape")};return consumeAutocompleteEscape;`)();
  let stopped=0,closed=0;
  const event={key:"Escape",stopPropagation(){stopped++;}};
  assert.equal(consume(event,true,()=>closed++),true);
  assert.deepEqual({stopped,closed},{stopped:1,closed:1},
    "the open menu closes and prevents the document handler from seeing the same Escape");
  assert.equal(consume({key:"Escape",stopPropagation(){stopped++;}},false,()=>closed++),false);
  assert.deepEqual({stopped,closed},{stopped:1,closed:1},
    "with no open menu, Escape remains available to close Preview or the route inspector");
  assert.equal((TSRC.match(/consumeAutocompleteEscape\(e,open,(?:closeAC|obCloseAC)\)/g)||[]).length,2,
    "both workplace autocomplete inputs use the same propagation boundary");
});

test("heatmap opacity is composited O(1), defaults to 65%, and round-trips through storage/hash", () => {
  const apply=templateFn("applyHeatOpacity");
  assert.match(TSRC,/--heat-opacity:\.65;--heat-pinned-opacity:\.195/);
  assert.match(TSRC,/let metric="r", ideal=25, thr=40, cmode="time", mapcolors="on", heatOpacity=\.65/);
  assert.match(TSRC,/id="opacity"[^>]*value="65"/);
  assert.match(TSRC,/\.leaflet-cells-pane canvas\{opacity:var\(--heat-opacity\)/);
  assert.match(apply,/setProperty\("--heat-opacity"/);
  assert.match(apply,/setProperty\("--heat-pinned-opacity"/);
  assert.doesNotMatch(apply,/\blayer\.setStyle/,
    "the slider must not restyle every heatmap feature");
  assert.doesNotMatch(apply,/cellFocus(?:Halo)?Layer\.setStyle/,
    "selection contrast is independent of the heatmap opacity control");
  assert.match(TSRC,/const cellOpacity=\(\)=>1/,
    "feature alpha stays constant while the pane owns user opacity");

  const input=TSRC.slice(TSRC.indexOf('document.getElementById("opacity").oninput='),
    TSRC.indexOf('// stamp the initial aria-pressed state'));
  assert.match(input,/applyHeatOpacity\(\)/);
  assert.match(input,/localStorage\.setItem\("map_opacity_v1"/);
  assert.match(input,/syncHash\(\)/);
  assert.doesNotMatch(input,/redraw\(|aggregate\(|layer\.setStyle/);

  assert.match(templateFn("syncHash"),/`op=\$\{Math\.round\(heatOpacity\*100\)\}`/);
  assert.match(templateFn("applyHash"),/o\.op[\s\S]*heatOpacity=Math\.min\(1,Math\.max\(\.2,\+o\.op\/100\)\);applyHeatOpacity\(\)/);
  const boot=TSRC.slice(TSRC.indexOf('(function bootOpacity()'),TSRC.indexOf('// First-paint theming'));
  assert.ok(boot.indexOf("parseHash()")<boot.indexOf('localStorage.getItem("map_opacity_v1")'),
    "hash opacity must win over the persisted preference");
});

test("hover keeps heatmap opacity unchanged while explicit selection gets an opacity-independent focus halo", () => {
  assert.doesNotMatch(TSRC,/tile-spotlight/,
    "the retired city-dimming hover path must not return");
  for (const pane of ["cellsPane", "cellFocusHaloPane", "cellFocusPane", "routePane"])
    assert.ok(TSRC.includes(`map.createPane("${pane}")`), `missing ${pane}`);
  assert.match(TSRC, /L\.geoJSON\(CELLS,\{pane:"cellsPane"/);
  assert.match(TSRC, /L\.geoJSON\(null,\{pane:"cellFocusHaloPane"/);
  assert.match(TSRC, /L\.geoJSON\(null,\{pane:"cellFocusPane"/);
  assert.match(TSRC, /const ROUTE_SVG=L\.svg\(\{pane:"routePane"\}\)/);
  const halo=templateFn("cellFocusHaloStyle"),focus=templateFn("cellFocusStyle");
  assert.match(halo,/fillOpacity:0[\s\S]*opacity:1[\s\S]*weight:7/);
  assert.match(focus,/fillOpacity:0[\s\S]*opacity:1[\s\S]*weight:3/);
  assert.doesNotMatch(halo,/heatOpacity/);
  assert.doesNotMatch(focus,/heatOpacity|fillColor/,
    "the inner stroke must not inherit the user-adjustable heatmap fill");
  assert.match(templateFn("openPin"),/showCellFocus\(f\)/,
    "a pinned cell gets the theme-contrasting double outline");
  assert.match(TSRC,/body\.route-pinned \.leaflet-cells-pane canvas\{opacity:var\(--heat-pinned-opacity\)\}/);

  const hover=TSRC.slice(TSRC.indexOf('l.on("mouseover"'),TSRC.indexOf('l.on("mouseout"'));
  assert.doesNotMatch(hover,/showCellFocus|tile-spotlight/,
    "desktop hover previews a route without dimming or outlining the city");
  assert.match(hover,/loadBreak\(/,"route preview remains wired on hover");
});

test("mobile pin settings are compact tokens that wrap instead of clipping the active mode", () => {
  assert.match(TSRC,/\.pin-settings\{display:grid;grid-template-columns:repeat\(3,minmax\(0,1fr\)\)/);
  assert.match(TSRC,/#pincard \.pin-context\{grid-template-columns:1fr;align-items:start;gap:9px\}/,
    "on mobile, settings take their own row instead of competing with the Adjust control");
  const pin=templateFn("pinHTML");
  assert.doesNotThrow(()=>new Function(pin),"the tokenized header remains valid browser JavaScript");
  assert.match(pin,/const settings=\[/);
  assert.match(pin,/"Scheduled"/);
  assert.match(pin,/"Medium"/);
  assert.match(pin,/class="pin-settings" aria-label="Route settings:/);
  assert.match(pin,/settings\.map\(\(\[label,value\]\)=>`<span/,
    "each setting is a separately wrappable, readable token");
});

test("route-row focus previews are desktop-only; touch cannot trigger focus redraws", () => {
  assert.match(TSRC,/document\.addEventListener\("focusin",e=>\{[\s\S]*?if\(row&&!TOUCH\)previewRow/);
  assert.match(TSRC,/document\.addEventListener\("focusout",e=>\{[\s\S]*?if\(!row\|\|TOUCH\)return/);
});

function _breakdownLRU(limit=3) {
  return new Function(`"use strict";${templateFn("createLRU")};return createLRU(${limit});`)();
}

test("route breakdown LRU is bounded, refreshes recency, protects selection, and clears", () => {
  const cache=_breakdownLRU(3);
  cache.set(1,"one");cache.set(2,"two");cache.set(3,"three");
  assert.equal(cache.get(1),"one");                  // order is now 2, 3, 1
  cache.set(4,"four");
  assert.equal(cache.get(2),undefined,"least-recently-used entry is evicted first");
  assert.equal(cache.get(1),"one","a read refreshes recency");
  assert.equal(cache.size,3);

  const selected=_breakdownLRU(2);
  const pending={_pinPending:true};
  selected.set(7,pending);selected.set(8,{plain:true});
  selected.set(9,{plain:true},7);
  assert.equal(selected.get(7),pending,"the selected pending response survives pressure");
  assert.equal(selected.get(8),undefined);
  const upgraded={_pin:true};selected.set(7,upgraded,7);
  assert.equal(selected.get(7),upgraded,"a pinned upgrade replaces the same selected entry");
  selected.clear();
  assert.equal(selected.size,0);
  assert.equal(selected.get(7),undefined);

  assert.match(TSRC,/const BREAKDOWN_CACHE_LIMIT=32/);
  assert.match(templateFn("setWorkplace"),/BDCACHE\.clear\(\)/,
    "a new destination clears the bounded route cache");
  assert.match(templateFn("applyRecomputeParam"),/BDCACHE\.clear\(\)/,
    "routing-setting changes clear stale route entries");
});

function _loadBreakScheduling(opts) {
  const harness=`
"use strict";
let GEN=1,maxxfers="any",walkspeed="med",routePin=null,EXACT=true,REAL={},VAR={};
let bdTimer=null,bdActive=null,bdToken=0,bdLightFlight=null;
const timers=[],fetches=[],html=[];
${templateFn("createLRU")}
const BDCACHE=createLRU(32),DESTLL=[37.7,-122.4];
function val(){return 22;} function bdHTML(){return "done";}
function maybeRoute(){} function upgradePinnedBreakdown(){} function escapeHTML(s){return String(s);}
function ridesParam(){return "";} function speedParam(){return "";} function retryAfter(){return 0;}
function sleep(){return Promise.resolve();}
function clearTimeout(){}
function setTimeout(fn,ms){timers.push({fn,ms});return timers.length;}
function fetch(url){fetches.push(url);return new Promise(()=>{});}
${templateFn("routeRequestKey")}
${templateFn("cancelLightBreakdown")}
${templateFn("lightBreakdownRequest")}
${templateFn("loadBreak")}
const f={properties:{id:11,n:"Test area"}};
loadBreak(f,h=>html.push(h),()=>{},${opts ? JSON.stringify(opts) : "undefined"});
return {timers,fetches,html};`;
  return new Function(harness)();
}

test("explicit route loads start now while true desktop hover alone keeps 150 ms debounce", () => {
  const explicit=_loadBreakScheduling();
  assert.equal(explicit.fetches.length,1,
    "click/tap/pin/refresh starts fetch in the current task");
  assert.equal(explicit.timers.length,0,"explicit loads have no artificial scheduling delay");
  assert.match(explicit.fetches[0],/^\/itinerary\?id=11&dlat=37\.7&dlon=-122\.4/,
    "the itinerary API contract is unchanged");

  const hover=_loadBreakScheduling({hoverDebounce:true});
  assert.equal(hover.fetches.length,0,"hover waits before requesting");
  assert.equal(hover.timers.length,1);
  assert.equal(hover.timers[0].ms,150);
  hover.timers[0].fn();
  assert.equal(hover.fetches.length,1,"the debounced task still performs the same lookup");

  const marks=TSRC.match(/\{hoverDebounce:true\}/g)||[];
  assert.equal(marks.length,1,"only one explicit callsite may opt into hover debounce");
  const mouseover=TSRC.slice(TSRC.indexOf('l.on("mouseover"'),TSRC.indexOf('l.on("mouseout"'));
  assert.match(mouseover,/loadBreak\([\s\S]*\{hoverDebounce:true\}\)/);
  for(const fn of ["openPin","openTouchPreview","refreshOpenInfo"])
    assert.doesNotMatch(templateFn(fn),/hoverDebounce/,
      `${fn} must remain an immediate explicit/refresh path`);
});

function _cachedPinHelpers() {
  const load=templateFn("loadBreak");
  const lru=templateFn("createLRU");
  const harness=`
"use strict";
let routePin=7,EXACT=true,REAL={},VAR={};
let bdTimer=null,bdToken=0;const calls=[];
${lru}
const BDCACHE=createLRU(32);BDCACHE.set(7,{geom:[{mode:"walk"}],_pin:false});
function val(){return 22;}
function bdHTML(d){return \`cached:\${d._pinPending}\`;}
function maybeRoute(id){calls.push(\`route:\${id}\`);}
function upgradePinnedBreakdown(f){calls.push(\`upgrade:\${f.properties.id}\`);}
function escapeHTML(s){return String(s);}
function ridesParam(){return "";} function speedParam(){return "";}
const DESTLL=[1,2];
${load}
return {calls,run(){const f={properties:{id:7,n:"Cached area"}};
  loadBreak(f,h=>calls.push(\`html:\${h}\`),d=>calls.push(\`ready:\${d._pinPending}\`));}};`;
  return new Function(harness)();
}

test("pinning a preview-cached cell renders immediately, then enriches with pin=1", () => {
  const H=_cachedPinHelpers();H.run();
  assert.deepEqual(H.calls,["html:cached:true","route:7","ready:true","upgrade:7"],
    "cached geometry/card renders synchronously before background enrichment starts");
  const upgrade=templateFn("upgradePinnedBreakdown");
  assert.match(upgrade,/&pin=1/);
  assert.match(upgrade,/d\._pin=true;d\._pinPending=false;BDCACHE\.set\(id,d,routePin\)/);
  assert.match(upgrade,/if\(!d\|\|d\.error\)[\s\S]*cached\._pinPending=false/,
    "a resolved HTTP/error response must clear the card's pending status");
  const light=templateFn("lightBreakdownRequest");
  assert.match(light,/\/itinerary\?id=/);
  assert.doesNotMatch(light,/pin=1/,
    "the coalesced initial Preview/Inspector request remains lightweight");
  const load=templateFn("loadBreak");
  assert.match(load,/if\(wantPin&&!d\._pin\)upgradePinnedBreakdown\(f\)/,
    "pin enrichment starts only after the cached response has been rendered");
});

function _previewInspectCoalesceHarness() {
  const harness=`
"use strict";
let GEN=1,maxxfers="any",walkspeed="med",routePin=null,EXACT=true,REAL={},VAR={};
let bdTimer=null,bdActive=null,bdToken=0,bdLightFlight=null,pinFeature=null;
let pinUpgradeSeq=0,pinUpgradeKey="",pinPrefetchKey="";
const fetches=[],deferreds=[],renders=[],ready=[];
const DESTLL=[37.7,-122.4],f={properties:{id:7,n:"Tapped area"}};
${templateFn("createLRU")}
const BDCACHE=createLRU(32);
function val(){return 22;} function bdHTML(d){return d._pin?"full":"light";}
function maybeRoute(){} function escapeHTML(s){return String(s);}
function ridesParam(){return "";} function speedParam(){return "";}
function retryAfter(){return 0;} function sleep(){return Promise.resolve();}
function clearTimeout(){} function varianceSettled(){return true;}
function schedulePinPrefetch(){throw new Error("no preview prefetch key should own this path");}
function fetch(url,opts){fetches.push({url,opts});return new Promise((resolve,reject)=>deferreds.push({resolve,reject}));}
function renderPin(_f,d){renders.push({pin:d._pin,pending:d._pinPending});}
const pinBody={scrollTop:0};function requestAnimationFrame(fn){fn();}
${templateFn("routeRequestKey")}
${templateFn("pinRequestKey")}
${templateFn("cancelLightBreakdown")}
${templateFn("lightBreakdownRequest")}
${templateFn("breakdownRequestPending")}
${templateFn("upgradePinnedBreakdown")}
${templateFn("loadBreak")}
return {
  preview(){loadBreak(f,()=>{},d=>ready.push({owner:"preview",pin:d&&d._pin}));},
  inspect(){bdToken++;routePin=7;pinFeature=f;
    loadBreak(f,()=>{},d=>ready.push({owner:"inspect",pin:d&&d._pin}));},
  resolveLight(){deferreds[0].resolve({status:200,ok:true,json:async()=>({geom:[{mode:"walk"}],alts:[]})});},
  resolvePin(){deferreds[1].resolve({status:200,ok:true,json:async()=>({geom:[{mode:"walk"}],alts:[{line:"planned"}]})});},
  upgradeAgain(){upgradePinnedBreakdown(f);},
  fetches,renders,ready,cached(){return BDCACHE.get(7);}
};`;
  return new Function(harness)();
}

test("Preview then immediate Inspect coalesces one light request and one final enrichment", async () => {
  const H=_previewInspectCoalesceHarness();
  H.preview();
  assert.equal(H.fetches.length,1);assert.doesNotMatch(H.fetches[0].url,/pin=1/);
  H.inspect();
  assert.equal(H.fetches.length,1,
    "Inspect subscribes to the in-flight generation/cell/params light request instead of refetching it");
  H.resolveLight();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(H.fetches.length,2,"the settled lightweight response starts one structural enrichment");
  assert.match(H.fetches[1].url,/&pin=1/);
  assert.deepEqual(H.ready,[{owner:"inspect",pin:false}],
    "only the current Inspector subscriber may publish the shared light response");
  H.upgradeAgain();
  assert.equal(H.fetches.length,2,"the in-flight pin upgrade key prevents a second enrichment");
  H.resolvePin();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(H.cached()._pin,true);assert.equal(H.cached()._pinPending,false);
  assert.deepEqual(H.renders.at(-1),{pin:true,pending:false},
    "the one enriched response becomes both the final card and cache source of truth");
});

function _closeDuringLightJsonHarness() {
  const harness=`
"use strict";
let GEN=1,maxxfers="any",walkspeed="med",routePin=7,EXACT=true,REAL={},VAR={};
let bdTimer=null,bdActive=null,bdToken=0,bdLightFlight=null,pinFeature=null;
let pinUpgradeSeq=0,pinUpgradeKey="",pendingFitId=null,resetPinScroll=false;
let compareList=[],selKey="primary",previewKey=null,mapChoiceKey=null,recommendedChoiceKey=null,recommendedChoiceKeys={},selectionLocked=false,allRoutesOpen=false,showAllRoutes=false,hoverCell=null;
const PRIMARY_KEY="primary",DESTLL=[37.7,-122.4],html=[],routes=[];
function resetRouteDisclosureState(){allRoutesOpen=false;}
const f={properties:{id:7,n:"Closing area"}};
let resolveHeaders,resolveJSON;
${templateFn("createLRU")}
const BDCACHE=createLRU(32);
function val(){return 22;} function bdHTML(){return "final";} function escapeHTML(s){return String(s);}
function maybeRoute(id){routes.push(id);} function upgradePinnedBreakdown(){routes.push("upgrade");}
function ridesParam(){return "";} function speedParam(){return "";} function retryAfter(){return 0;}
function sleep(){return Promise.resolve();} function clearTimeout(){}
function fetch(){return new Promise(resolve=>{resolveHeaders=resolve;});}
function cancelPinPrefetch(){} function clearCellFocus(){} function clearRoute(){routes.push("clear");}
function drawJourney(){routes.push("hover");}
const classNames=new Set(["open"]);
const pinCard={classList:{contains:n=>classNames.has(n),remove:(...ns)=>ns.forEach(n=>classNames.delete(n))}};
const pinBody={scrollTop:0};
const document={body:{classList:{remove(){}}}};
const map={invalidateSize(){},getContainer(){return {focus(){}};}};
function requestAnimationFrame(fn){fn();}
${templateFn("routeRequestKey")}
${templateFn("cancelLightBreakdown")}
${templateFn("lightBreakdownRequest")}
${templateFn("loadBreak")}
${templateFn("closePin")}
return {
  start(){loadBreak(f,h=>html.push(h),()=>html.push("ready"));},
  headers(){resolveHeaders({status:200,ok:true,json:()=>new Promise(resolve=>{resolveJSON=resolve;})});},
  close:closePin,
  json(){resolveJSON({geom:[{mode:"walk"}],alts:[]});},
  html,routes,cached(){return BDCACHE.get(7);},get token(){return bdToken;}
};`;
  return new Function(harness)();
}

test("closing after light response headers but before JSON prevents every late publication", async () => {
  const H=_closeDuringLightJsonHarness();H.start();
  H.headers();await new Promise(resolve=>setImmediate(resolve));
  const token=H.token;H.close();
  assert.equal(H.token,token+1,"close invalidates the active light-response subscriber before aborting");
  H.json();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(H.cached(),undefined,"late JSON cannot repopulate the closed cell's cache");
  assert.deepEqual(H.html,[],"late JSON cannot paint or announce the closed card");
  assert.deepEqual(H.routes,["clear"],"only close clears the route; late data cannot redraw or enrich it");
});

function _deferredPinRaceHarness() {
  const harness=`
"use strict";
let GEN=1,maxxfers="any",walkspeed="med",routePin=7,hoverCell=null,EXACT=true,REAL={7:{frag:2}},VAR={7:{frag:2}};
let bdTimer=null,bdActive=null,bdToken=0,bdLightFlight=null,pinFeature=null;
const fetches=[],html=[],ready=[],upgrades=[];let resolveLight;
${templateFn("createLRU")}
const BDCACHE=createLRU(32),DESTLL=[37.7,-122.4];
function val(){return 22;} function bdHTML(d){return d._pin?"full":"light";}
function maybeRoute(){} function upgradePinnedBreakdown(){upgrades.push("upgrade");}
function escapeHTML(s){return String(s);} function ridesParam(){return "";}
function speedParam(){return "";} function retryAfter(){return 0;}
function sleep(){return Promise.resolve();}
function fetch(url){fetches.push(url);return new Promise(resolve=>{resolveLight=resolve;});}
function setPinHTML(h){html.push(h);} function renderPin(_f,d){html.push(d._pin?"render-full":"render-light");}
const layer={eachLayer(){}};
${templateFn("routeRequestKey")}
${templateFn("cancelLightBreakdown")}
${templateFn("lightBreakdownRequest")}
${templateFn("breakdownRequestPending")}
${templateFn("loadBreak")}
${templateFn("refreshOpenInfo")}
${templateFn("refreshDrawnRoute")}
const f={properties:{id:7,n:"Pinned area"}};pinFeature=f;
loadBreak(f,h=>html.push(h),d=>ready.push(d&&d._pin));
return {
  fetches,html,ready,upgrades,
  refreshOpen:refreshOpenInfo,refreshDrawn:refreshDrawnRoute,
  seedFull(){BDCACHE.set(7,{_pin:true,geom:[{mode:"walk"}],alts:[{line:"full"}]},7);},
  resolveLight(){resolveLight({status:200,ok:true,json:async()=>({_pin:false,geom:[{mode:"walk"}],alts:[]})});},
  cached(){return BDCACHE.get(7);},pending(){return breakdownRequestPending(7);}
};`;
  return new Function(harness)();
}

test("an in-flight lightweight pin owns refresh and cannot overwrite a newer full response", async () => {
  const H=_deferredPinRaceHarness();
  assert.equal(H.fetches.length,1);
  assert.equal(H.pending(),true,"the uncached lightweight request is registered as the owner");

  H.refreshOpen();H.refreshDrawn();
  assert.equal(H.fetches.length,1,
    "a variance/open-card refresh must not duplicate the active lightweight request");
  assert.deepEqual(H.upgrades,[],"the drawn-route refresh also waits for the light owner");

  H.seedFull();                         // pin=1 wins the network race
  H.resolveLight();                     // stale lightweight response arrives afterward
  await new Promise(resolve=>setImmediate(resolve));

  assert.equal(H.cached()._pin,true,"late lightweight data cannot downgrade the full cache entry");
  assert.equal(H.html.at(-1),"full","the card stays on the richer response");
  assert.deepEqual(H.ready,[true]);
  assert.deepEqual(H.upgrades,[],"the stale response cannot launch a duplicate pin=1 request");
  assert.equal(H.pending(),false,"request ownership is released after settlement");
});

function _deferredHoverRefreshHarness() {
  const harness=`
"use strict";
let GEN=1,maxxfers="any",walkspeed="med",routePin=null,hoverCell=7,EXACT=true,REAL={},VAR={7:{frag:2}};
let bdTimer=null,bdActive=null,bdToken=0,bdLightFlight=null,pinFeature=null;
const fetches=[],upgrades=[];
${templateFn("createLRU")}
const BDCACHE=createLRU(32),DESTLL=[37.7,-122.4];
function val(){return 22;} function bdHTML(){return "light";} function maybeRoute(){}
function upgradePinnedBreakdown(){upgrades.push("upgrade");} function escapeHTML(s){return String(s);}
function ridesParam(){return "";} function speedParam(){return "";} function retryAfter(){return 0;}
function sleep(){return Promise.resolve();}
function fetch(url){fetches.push(url);return new Promise(()=>{});}
function setPinHTML(){} function renderPin(){}
const f={properties:{id:7,n:"Hovered area"}},tip={isOpen(){return true;}};
const hoverLayer={feature:f,getTooltip(){return tip;},setTooltipContent(){}};
const layer={eachLayer(fn){fn(hoverLayer);}};
function _featureById(){return f;}
${templateFn("routeRequestKey")}
${templateFn("cancelLightBreakdown")}
${templateFn("lightBreakdownRequest")}
${templateFn("breakdownRequestPending")}
${templateFn("loadBreak")}
${templateFn("refreshOpenInfo")}
${templateFn("refreshDrawnRoute")}
loadBreak(f,()=>{},()=>{});
return {fetches,upgrades,pending(){return breakdownRequestPending(7);},
  refreshOpen:refreshOpenInfo,refreshDrawn:refreshDrawnRoute};`;
  return new Function(harness)();
}

test("an in-flight lightweight hover also owns variance/open-route refreshes", () => {
  const H=_deferredHoverRefreshHarness();
  assert.equal(H.fetches.length,1);assert.equal(H.pending(),true);
  H.refreshOpen();H.refreshDrawn();
  assert.equal(H.fetches.length,1,
    "open-tooltip and drawn-route refreshes cannot cancel/duplicate an uncached hover lookup");
  assert.deepEqual(H.upgrades,[]);
});

test("keep-pin parameter recompute cancels old enrichment and owns the only final pin request", () => {
  const recompute=templateFn("applyRecomputeParam");
  assert.match(recompute,/pinUpgradeSeq\+\+;pinUpgradeKey="";[\s\S]*BDCACHE\.clear\(\)/,
    "changing speed/transfers invalidates an older progressive pin response before clearing cache");
  const setWorkplace=templateFn("setWorkplace");
  assert.match(setWorkplace,/await loadVariance\(\{skipActive:keepPinId!=null\}\)/,
    "atomic keep-pin recompute suppresses variance's competing active-card refresh");
  const variance=templateFn("loadVariance");
  assert.match(variance,/skipActive=!!\(opts&&opts\.skipActive\)/);
  assert.match(variance,/if\(!skipActive\)\{refreshOpenInfo\(\);[\s\S]*refreshDrawnRoute\(\)/);
  assert.match(setWorkplace,/refreshPinForParams\(keepPinId,myGen\)/,
    "the generation-guarded parameter refresh remains the sole final pin=1 owner");
});

test("settled bottom-sheet geometry is measured for map occlusion without fitting during live drag", () => {
  const insets=templateFn("viewInsets");
  assert.match(insets,/data-layout-capability[\s\S]{0,500}bottom=Math\.max\(bottom,sz\.y-qc\.t\)/,
    "a settled bottom sheet contributes its measured rectangle as a bottom occluder");
  const controller=TSRC;
  assert.match(controller,/data-dragging|dragging/,
    "the controller can distinguish a live drag from a settled snap");
  assert.doesNotMatch(controller,/dragging[\s\S]{0,180}fitToCompare\(/,
    "the map must not refit on every drag frame");
  assert.match(controller,/(?:transitionend|requestAnimationFrame)[\s\S]{0,260}fitToCompare\(/,
    "occlusion fitting occurs after a presentation transition settles");
});

test("the route inspector region is stable while dedicated status nodes announce updates", () => {
  assert.doesNotMatch(TSRC,/<aside id="pincard"[^>]*aria-live/,
    "replacing the whole inspector must not re-announce every route row");
  assert.match(TSRC,/class="cmploading" role="status"/);
  assert.match(TSRC,/note\.setAttribute\("role","status"\)/);
});

test("reduced-motion preference disables animated route fitting", () => {
  assert.match(TSRC, /const REDUCE_MOTION=.*prefers-reduced-motion: reduce/);
  const fit = templateFn("fitBoundsUnoccluded");
  assert.match(fit, /animate:!REDUCE_MOTION/);
  assert.match(fit, /duration:REDUCE_MOTION\?0:0\.5/);
  assert.match(TSRC, /@media \(prefers-reduced-motion:reduce\)/);
  const reduced=TSRC.match(/@media \(prefers-reduced-motion:reduce\)\{([\s\S]*?)\n  \}/)?.[1]||"";
  assert.doesNotMatch(reduced,/\*,\*:before,\*:after/,
    "reduced motion must not blanket-disable unrelated transitions");
  assert.match(reduced,/#pincard[\s\S]*#panel[\s\S]*\.leaflet-cells-pane canvas/,
    "route and map state feedback remains immediate in the scoped override");
});
