/* Pure settings, onboarding, permalink, and export primitives.
 *
 * The page module remains the DOM/network adapter.  These helpers keep the
 * durable contracts (URL state, saved workplace validation, autocomplete
 * rows, and CSV output) browser-independent and easy to exercise directly.
 */

export const STARTUP_STATES = Object.freeze(["restoring", "onboarding", "error", "ready"]);

export function classifyStartupState(hash, saved) {
  const parsed = parseHash(hash);
  const workplace = parsed && parseWorkplaceParam(parsed.wp);
  if (workplace) return "restoring";
  return readSavedWorkplace(saved) ? "restoring" : "onboarding";
}

export function parseHash(hash) {
  const raw = String(hash || "").replace(/^#/, "");
  if (!raw) return null;
  const result = {};
  raw.split("&").forEach((part) => {
    const at = part.indexOf("=");
    if (at >= 0) result[part.slice(0, at)] = part.slice(at + 1);
  });
  return result;
}

export function parseWorkplaceParam(raw) {
  if (raw == null) return null;
  const bits = String(raw).split(",");
  const lat = +bits[0];
  const lon = +bits[1];
  if (bits.length < 2 || bits[0] === "" || bits[1] === "" ||
      !Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  const encodedLabel = bits.slice(2).join(",");
  let label;
  try { label = decodeURIComponent(encodedLabel); } catch (_error) { label = encodedLabel; }
  return { lat, lon, label: label || "" };
}

export function readSavedWorkplace(value) {
  try {
    const saved = typeof value === "string" ? JSON.parse(value || "null") : value;
    return saved && Number.isFinite(+saved.lat) && Number.isFinite(+saved.lon) ? saved : null;
  } catch (_error) {
    return null;
  }
}

export function clampHeatOpacity(value, min = 0.2, max = 1) {
  const parsed = Number(value);
  const safe = Number.isFinite(parsed) ? parsed : 0.65;
  return Math.min(max, Math.max(min, safe));
}

export function resolveThemePreference(preference, prefersLight = false) {
  return preference === "light" || preference === "dark"
    ? preference : (prefersLight ? "light" : "dark");
}

export function buildHash({ destination, label = "", ideal, threshold, metric,
  colorMode, mapColors, opacity, maxTransfers, walkSpeed, theme } = {}) {
  const parts = [];
  if (Array.isArray(destination) && destination.length >= 2) {
    parts.push(`wp=${Number(destination[0]).toFixed(6)},${Number(destination[1]).toFixed(6)},${encodeURIComponent(label)}`);
  }
  parts.push(`ideal=${ideal}`, `thr=${threshold}`, `metric=${metric}`, `cmode=${colorMode}`,
    `colors=${mapColors}`, `op=${Math.round(Number(opacity) * 100)}`, `mt=${maxTransfers}`,
    `sp=${walkSpeed}`, `th=${theme}`);
  return `#${parts.join("&")}`;
}

export function escapeHTML(value) {
  return String(value || "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));
}

export function autocompleteHTML(items, activeIndex = -1, idPrefix = "ac") {
  return (items || []).map((item, index) =>
    `<div class="ac-item${index === activeIndex ? " hl" : ""}" id="${idPrefix}-item-${index}" role="option" aria-selected="${index === activeIndex}" data-i="${index}">${escapeHTML(item.label)}</div>`
  ).join("");
}

export function exportRows(neighborhoods, threshold) {
  return Object.entries(neighborhoods || {})
    .filter(([, value]) => value != null && value <= threshold)
    .sort((a, b) => a[1] - b[1]);
}

export function csvCell(value) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function neighborhoodCSV(rows) {
  return "name,minutes\n" + (rows || []).map(([name, minutes]) =>
    `${csvCell(name)},${minutes}`).join("\n");
}

export function exportFilename(threshold, metric) {
  return `sf-neighborhoods-under-${threshold}min-${metric === "r" ? "realistic" : "bestcase"}.csv`;
}
