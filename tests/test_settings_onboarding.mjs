import { test } from "node:test";
import assert from "node:assert/strict";
import * as settings from "../scripts/static/settings-onboarding.mjs";

test("settings permalink round-trip preserves workplace and controls", () => {
  const hash = settings.buildHash({ destination: [37.77155, -122.40375], label: "650 Townsend St, Bank of America",
    ideal: 25, threshold: 40, metric: "r", colorMode: "time", mapColors: "on", opacity: 0.65,
    maxTransfers: "any", walkSpeed: "med", theme: "auto" });
  const parsed = settings.parseHash(hash);
  assert.deepEqual(settings.parseWorkplaceParam(parsed.wp), {
    lat: 37.77155, lon: -122.40375, label: "650 Townsend St, Bank of America",
  });
  assert.equal(parsed.op, "65");
  assert.equal(parsed.th, "auto");
});

test("startup classification rejects incomplete or nonnumeric destinations", () => {
  assert.equal(settings.classifyStartupState("#wp=37.77,,bad", null), "onboarding");
  assert.equal(settings.classifyStartupState("#wp=nope,-122.4,label", null), "onboarding");
  assert.equal(settings.classifyStartupState("", '{"lat":37.77,"lon":-122.4}'), "restoring");
  assert.equal(settings.classifyStartupState("#wp=37.77,-122.4,label", null), "restoring");
});

test("settings presentation helpers clamp opacity and escape autocomplete rows", () => {
  assert.equal(settings.clampHeatOpacity(undefined), 0.65);
  assert.equal(settings.clampHeatOpacity(0), 0.2);
  assert.equal(settings.clampHeatOpacity(2), 1);
  const html = settings.autocompleteHTML([{ label: "A&B <Market>" }], 0, "obac");
  assert.match(html, /id="obac-item-0"/);
  assert.match(html, /A&amp;B &lt;Market&gt;/);
  assert.match(html, /aria-selected="true"/);
});

test("neighborhood export helpers preserve ranking and CSV quoting", () => {
  const rows = settings.exportRows({ Alpha: 20, 'Beta, North': 15, Outside: 50 }, 25);
  assert.deepEqual(rows, [["Beta, North", 15], ["Alpha", 20]]);
  assert.equal(settings.neighborhoodCSV(rows), 'name,minutes\n"Beta, North",15\nAlpha,20');
  assert.equal(settings.exportFilename(25, "r"), "sf-neighborhoods-under-25min-realistic.csv");
});
