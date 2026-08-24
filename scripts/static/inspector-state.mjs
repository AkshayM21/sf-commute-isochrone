/* Pure state machine for the responsive route inspector.
 *
 * The page owns DOM measurement, pointer capture, focus, and rendering.  This
 * module owns only decisions, so resize/drag/keyboard sequences can be tested
 * without a browser and without accidentally coupling a sheet transition to
 * map or pane scroll state.
 */

export const SHEET_SNAPS = Object.freeze(["peek", "browse", "expanded"]);
export const SHEET_DRAG_THRESHOLD = 7;
export const SHEET_VELOCITY_THRESHOLD = 0.55;

export function inspectorLayoutCapability(width) {
  const w = Number(width);
  const value = Number.isFinite(w) ? w : 0;
  return value < 720 ? "bottom-sheet" : value >= 1280 ? "wide-sidecar" : "single-card";
}

export function sheetMetrics(height) {
  const raw = Number(height);
  const viewport = Number.isFinite(raw) ? raw : 0;
  const available = Math.max(0, Math.floor(viewport));
  const expanded = Math.max(0, available - 8);
  const peekFloor = available < 260 ? 96 : 112;
  const peek = Math.min(expanded, Math.max(peekFloor,
    Math.min(132, Math.round(available * 0.18))));
  const browseTarget = Math.round(available * (available < 480 ? 0.70 : 0.58));
  const browse = Math.min(expanded, Math.max(peek, browseTarget));
  return {
    height: available,
    visible: { peek, browse, expanded },
    snaps: { peek: available - peek, browse: available - browse, expanded: available - expanded },
  };
}

export function clampSheetOffset(value, metrics) {
  const snaps = (metrics && metrics.snaps) || {};
  const expanded = Number.isFinite(Number(snaps.expanded)) ? Number(snaps.expanded) : 0;
  const peek = Number.isFinite(Number(snaps.peek)) ? Number(snaps.peek) : expanded;
  return Math.max(expanded, Math.min(peek, Number(value) || 0));
}

export function sheetSnapForRelease(offset, velocity, metrics, {
  velocityThreshold = SHEET_VELOCITY_THRESHOLD,
} = {}) {
  const snaps = (metrics && metrics.snaps) || {};
  const point = Number(offset) || 0;
  const vy = Number(velocity) || 0;
  if (vy > velocityThreshold) return "peek";
  if (vy < -velocityThreshold) return "expanded";
  const keys = SHEET_SNAPS.filter((key) => Number.isFinite(Number(snaps[key])));
  if (!keys.length) return "browse";
  return keys.reduce((best, key) =>
    Math.abs(point - snaps[key]) < Math.abs(point - snaps[best]) ? key : best, keys.includes("browse") ? "browse" : keys[0]);
}

export function beginSheetGesture({ pointerId, startY, startOffset, now = 0, metrics } = {}) {
  return {
    pointerId, startY: Number(startY) || 0, startOffset: Number(startOffset) || 0,
    lastY: Number(startY) || 0, lastAt: Number(now) || 0, velocity: 0,
    metrics, moved: false,
  };
}

export function updateSheetGesture(drag, { clientY, now = 0 } = {}) {
  if (!drag) return { drag: null, offset: null };
  const y = Number(clientY) || 0;
  const time = Number(now) || 0;
  const dy = y - drag.startY;
  const dt = Math.max(1, time - drag.lastAt);
  const next = { ...drag, lastY: y, lastAt: time,
    velocity: (y - drag.lastY) / dt,
    moved: drag.moved || Math.abs(dy) > SHEET_DRAG_THRESHOLD };
  return { drag: next, offset: clampSheetOffset(next.startOffset + dy, next.metrics) };
}

export function finishSheetGesture(drag, { clientY, cancelled = false } = {}) {
  if (!drag) return { snap: null, moved: false, suppressClick: false, offset: null };
  if (cancelled) return { snap: null, moved: false, suppressClick: false, offset: null };
  const current = clampSheetOffset(drag.startOffset + ((Number(clientY) || 0) - drag.startY), drag.metrics);
  return {
    snap: sheetSnapForRelease(current, drag.velocity, drag.metrics),
    moved: drag.moved,
    suppressClick: drag.moved,
    offset: current,
  };
}

export function sheetKeyboardSnap(current, key) {
  const at = Math.max(0, SHEET_SNAPS.indexOf(current));
  if (key === "Enter" || key === " ") return current === "peek" ? "browse" : "peek";
  if (key === "ArrowUp") return SHEET_SNAPS[Math.min(SHEET_SNAPS.length - 1, at + 1)];
  if (key === "ArrowDown") return SHEET_SNAPS[Math.max(0, at - 1)];
  if (key === "Home") return "peek";
  if (key === "End") return "expanded";
  return null;
}

export function createInspectorState({ capability = "single-card", sheetSnap } = {}) {
  return {
    surface: "routes", planOpen: false, presentation: "expanded", sheetContent: "choices",
    sheetSnap: sheetSnap || (capability === "bottom-sheet" ? "peek" : "browse"),
    settingsReturn: null, returnFocus: null, returnFocusSelector: "", dragging: false,
  };
}

export function inspectorSnapshot(state) {
  return {
    surface: state.surface, planOpen: !!state.planOpen, presentation: state.presentation,
    sheetContent: state.sheetContent, sheetSnap: state.sheetSnap,
  };
}

export function restoreInspectorSnapshot(state, snapshot) {
  if (!snapshot) return { ...state, settingsReturn: null, dragging: false };
  return { ...state, ...snapshot, settingsReturn: null, dragging: false };
}

export function normalizeInspectorState(state, capability) {
  const next = { ...state };
  if (capability === "bottom-sheet" && next.surface === "routes") {
    next.sheetContent = next.planOpen ? "plan" : "choices";
    if (next.planOpen && next.sheetSnap === "peek") next.sheetSnap = "browse";
  }
  if (capability !== "bottom-sheet" && next.presentation === "map-focus" && next.surface === "settings") {
    next.presentation = "expanded";
  }
  return next;
}

/* Return a decision rather than mutating a scrollTop.  In particular, a resize
 * or drag captures the live pane position first; it never substitutes the
 * selected route's initial offset for a user who scrolled elsewhere. */
export function inspectorScrollDecision({ phase = "capture", visible = true,
  current = 0, saved = 0, pane = "choices" } = {}) {
  const currentOffset = Number(current) || 0;
  const savedOffset = Number(saved) || 0;
  if (phase === "capture") return { pane, save: !!visible, restore: false, value: currentOffset };
  if (phase === "restore") return {
    pane, save: false, restore: !!visible && currentOffset === 0 && savedOffset > 0, value: savedOffset,
  };
  if (phase === "resize" || phase === "drag-start" || phase === "render") return {
    pane, save: !!visible, restore: !!visible && currentOffset === 0 && savedOffset > 0,
    value: currentOffset || savedOffset,
  };
  return { pane, save: false, restore: false, value: currentOffset };
}

export function transitionInspectorState(state, action, {
  capability = "single-card", origin = "programmatic", snapshot = null,
  sourceHidden = false, returnFocusSelector = "[data-settings-toggle]", snap = null,
} = {}) {
  let next = { ...state };
  const effects = { focus: null, enteringMapFocus: false };
  if (action === "plan-open" || action === "plan") {
    const opening = action === "plan-open" ? true : !next.planOpen;
    next.planOpen = opening;
    if (capability === "bottom-sheet") {
      next.sheetContent = opening ? "plan" : "choices";
      next.sheetSnap = "browse";
    }
    if (opening && (origin === "keyboard" || sourceHidden)) effects.focus = "plan";
    if (!opening && origin === "keyboard") effects.focus = "route-plan-control";
  } else if (action === "plan-close" || action === "choices") {
    next.sheetContent = "choices"; next.planOpen = false;
    if (capability === "bottom-sheet") next.sheetSnap = "browse";
    if (origin === "keyboard") effects.focus = "route-plan-control";
  } else if (action === "settings") {
    if (next.surface !== "settings") {
      next.settingsReturn = snapshot || inspectorSnapshot(next);
      next.returnFocusSelector = returnFocusSelector;
      next.surface = "settings"; next.sheetContent = "settings";
      if (capability === "bottom-sheet") next.sheetSnap = "browse";
    }
  } else if (action === "settings-return") {
    next = restoreInspectorSnapshot(next, next.settingsReturn);
    effects.focus = "settings-return";
  } else if (action === "map-focus") {
    if (capability === "bottom-sheet") next.sheetSnap = "peek";
    else next.presentation = next.presentation === "map-focus" ? "expanded" : "map-focus";
  } else if (action === "show-choices") {
    next.sheetContent = "choices";
    if (capability === "bottom-sheet") next.sheetSnap = "browse";
    next.presentation = "expanded";
  } else if (action === "snap") {
    next.sheetSnap = SHEET_SNAPS.includes(snap) ? snap : "browse";
  }
  const normalized = normalizeInspectorState(next, capability);
  if (action === "map-focus" && capability !== "bottom-sheet" && normalized.presentation === "map-focus") {
    effects.enteringMapFocus = true;
  }
  return { state: normalized, effects };
}

export const classifyInspectorLayout = inspectorLayoutCapability;
export const sheetSnapForDrag = sheetSnapForRelease;
