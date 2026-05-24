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
test("gmapsURL: builds the expected Google Maps transit directions link", () => {
  const url = gmapsURL(37.78, -122.41, 37.7955, -122.3937);
  assert.equal(
    url,
    "https://www.google.com/maps/dir/?api=1" +
      "&origin=37.78,-122.41" +
      "&destination=37.7955,-122.3937" +
      "&travelmode=transit"
  );
});

test("gmapsURL: always requests the transit travel mode", () => {
  const url = gmapsURL(1, 2, 3, 4);
  assert.match(url, /travelmode=transit$/);
  assert.match(url, /origin=1,2/);
  assert.match(url, /destination=3,4/);
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
