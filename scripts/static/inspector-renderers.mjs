/* Route-choice and Route-plan view-model/DOM renderers.
 *
 * This module deliberately knows nothing about the map, requests, or sheet state.  The
 * controller supplies live values as callbacks so a re-render can preserve the historical
 * selection, disclosure, and responsive presentation semantics without making the renderers
 * stateful.  The returned strings are the existing inspector markup; class names and copy are
 * part of the public DOM contract.
 */
import * as routeModel from "./route-choice-model.mjs";

const defaultEscape = (value) => String(value || "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));

export function createInspectorRenderers(options = {}) {
  const read = (value, fallback) => typeof value === "function" ? value() : (value ?? fallback);
  const escapeHTML = options.escapeHTML || defaultEscape;
  const getMetric = () => read(options.metric, "r");
  const getCompareList = () => read(options.compareList, []);
  const getFamilies = () => read(options.families, null);
  const getSelectedKey = () => read(options.selectedKey, null);
  const getPlanOpen = () => !!read(options.planOpen, false);
  const getRecommendedKey = () => read(options.recommendedChoiceKey, null);
  const getMapKey = () => read(options.mapChoiceKey, null);
  const getShowAll = () => !!read(options.showAllRoutes, false);
  const primaryCasing = () => read(options.primaryCasing, "#fff");
  const getGmaps = options.gmaps || ((lat, lon) => `https://www.google.com/maps/dir/?api=1&destination=${lat},${lon}`);
  const optionMetric = (o) => getMetric();
  const fmt = (v) => routeModel.formatMinutes(v);
  const optLegs = (o) => routeModel.optLegs(o, optionMetric());
  const optTotal = (o) => routeModel.optTotal(o, optionMetric());
  const optWalk = (o) => routeModel.optionWalk(o, optionMetric());
  const optTransfers = (o) => routeModel.optionTransfers(o, optionMetric());
  const routeServices = (raw, fallback) => routeModel.routeServices(raw, fallback);
  const serviceNames = (meta) => routeModel.serviceNames(meta);
  const lname = (g) => routeModel.lname(g);
  const naturalList = (items) => routeModel.naturalList(items);
  const displayStopName = (name) => routeModel.displayStopName(name);
  const legPhysicalMin = (g) => routeModel.legPhysicalMin(g);
  const legScheduleAllowanceMin = (g) => routeModel.legScheduleAllowanceMin(g);

  function buildRouteChoices(compareList = getCompareList()) {
    const out = [];
    (getFamilies() || routeModel.buildFamilies(compareList, optionMetric())).forEach((f) => f.branches.forEach((b) => {
      (b.opts || []).forEach((x) => { if (x) out.push({ family: f, branch: b, o: x.o, r: x.r }); });
    }));
    return out.sort((a, b) => {
      const ar = choiceMatchesKey(a, getRecommendedKey()), br = choiceMatchesKey(b, getRecommendedKey());
      if (ar !== br) return ar ? -1 : 1;
      return (a.r.head - b.r.head) || (optWalk(a.o) - optWalk(b.o)) ||
        (optTransfers(a.o) - optTransfers(b.o)) ||
        ((a.r.fragKnown ? a.r.frag : Number.MAX_SAFE_INTEGER) -
         (b.r.fragKnown ? b.r.frag : Number.MAX_SAFE_INTEGER)) ||
        (Number(optTotal(a.o) || 0) - Number(optTotal(b.o) || 0)) ||
        String(a.o.key).localeCompare(String(b.o.key));
    });
  }

  function choiceMatchesKey(choice, key) {
    return !!(choice && key != null && (String(choice.o.key) === String(key) ||
      String(choice.o.choiceKey || "") === String(key)));
  }
  function routeChoices() { return buildRouteChoices(); }
  function recommendedChoice(choices = routeChoices()) {
    return choices.find((c) => choiceMatchesKey(c, getRecommendedKey())) ||
      choices.find((c) => choiceMatchesKey(c, getMapKey())) ||
      choices.find((c) => c.o.isPrimary) || choices[0] || null;
  }
  function mapChoice(choices = routeChoices()) {
    return choices.find((c) => choiceMatchesKey(c, getMapKey())) ||
      choices.find((c) => c.o.isPrimary) || choices[0] || null;
  }
  function representativeChoices(choices = routeChoices()) {
    const grouped = new Map();
    choices.forEach((choice) => {
      const key = `${choice.family.key}\u0000${choice.branch.key}`;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(choice);
    });
    return [...grouped.values()].map((group) => group.find((choice) => choice.o.isPrimary) || group[0]);
  }
  function branchServiceRows(choice) {
    const b = (choice && choice.branch && choice.branch.meta) || {};
    let services = routeServices(b.services, b.lines);
    if (!services.length) {
      const key = choice && choice.branch && choice.branch.key;
      services = routeServices(choice && choice.family && choice.family.meta.services, [])
        .filter((s) => (s.branchKeys || []).map(String).includes(String(key)));
    }
    return services;
  }
  function routeTitle(choice) {
    const fam = choice.family.meta.name || "Route", branch = choice.branch.meta.name || "";
    const walkPrefix = "walk after ", compact = (s) => String(s || "").replace(/\s+/g, " ").trim().toLowerCase();
    if (compact(branch).startsWith(walkPrefix) && compact(branch).slice(walkPrefix.length) === compact(fam))
      return `${fam} → walk to destination`;
    return branch && branch !== fam ? `${fam} → ${branch}` : fam;
  }
  function routeReasonParts(choice, recommended) {
    if (!choice || !recommended) return [];
    const baseline = choice === recommended ? (mapChoice() || recommended) : recommended;
    if (choice === recommended && baseline === recommended) return ["Best overall balance"];
    const parts = [], delta = choice.r.head - baseline.r.head;
    parts.push(delta > .05 ? `${fmt(delta)} slower` : delta < -.05 ? `${fmt(-delta)} faster` : "Same trip time");
    const walkDelta = optWalk(baseline.o) - optWalk(choice.o), transfers = optTransfers(choice.o), baseTransfers = optTransfers(baseline.o);
    if (walkDelta > .45) parts.push(`${fmt(walkDelta)} less walking`);
    else if (walkDelta < -.45) parts.push(`${fmt(-walkDelta)} more walking`);
    else if (transfers < baseTransfers) parts.push(`${baseTransfers - transfers} fewer transfer${baseTransfers - transfers === 1 ? "" : "s"}`);
    else if (transfers > baseTransfers) parts.push(`${transfers - baseTransfers} more transfer${transfers - baseTransfers === 1 ? "" : "s"}`);
    else if (choice.r.fragKnown && baseline.r.fragKnown && choice.r.frag < baseline.r.frag) parts.push("Lower bad-day delay");
    else if (choice.r.fragKnown && baseline.r.fragKnown && choice.r.frag > baseline.r.frag) parts.push("Higher bad-day delay");
    else if (choice === recommended) return ["Best overall balance"];
    else if (choice.family.key !== baseline.family.key) parts.push("Different boarding corridor");
    else if (choice.branch.key !== baseline.branch.key) parts.push("Different destination approach");
    else parts.push("Distinct itinerary");
    return parts;
  }
  function routeBadDay(choice) {
    return choice && choice.r.fragKnown ? Number(choice.r.badDayBase || choice.r.head || 0) + Number(choice.r.frag || 0) : null;
  }
  function routeFactsVM(choice, includeTime = false) {
    const facts = [], badDay = routeBadDay(choice);
    if (includeTime) facts.push(["time", "Trip time", fmt(choice.r.head)]);
    facts.push(["walk", "Walking", fmt(optWalk(choice.o))]);
    facts.push(["transfers", "Transfers", String(optTransfers(choice.o))]);
    facts.push(["bad-day", "Bad day", badDay == null ? "Not available" : fmt(badDay)]);
    return facts;
  }
  function routeFactsHTML(choice, includeTime) {
    return routeFactsVM(choice, includeTime).map(([key, label, value]) => `<span class="route-fact" data-fact="${key}">` +
      `<span class="route-fact-label">${label}</span><span class="route-fact-value">${escapeHTML(value)}</span></span>`).join("");
  }
  function routeTradeoffsHTML(choice, recommended) {
    return `<span class="route-tradeoffs" data-route-reason>${routeReasonParts(choice, recommended).map((p) => `<span>${escapeHTML(p)}</span>`).join("")}</span>`;
  }
  function routeRowHTML(choice, recommended, where) {
    const o = choice.o, r = choice.r, selected = getSelectedKey() === o.key;
    const classes = ["route-row", "route-choice", choice === recommended ? "recommended" : "", where || ""].filter(Boolean).join(" ");
    return `<article class="route-choice-card" data-choice-card-key="${escapeHTML(o.key)}" data-selected="${selected}">` +
      `<button type="button" class="${classes}" data-key="${escapeHTML(o.key)}" data-choice-key="${escapeHTML(o.key)}" ` +
      `data-family="${escapeHTML(choice.family.key)}" data-branch="${escapeHTML(choice.branch.key)}" ` +
      `data-family-name="${escapeHTML(choice.family.meta.name || "Route")}" data-branch-name="${escapeHTML(choice.branch.meta.name || "")}" ` +
      `aria-pressed="${selected}"><span class="route-dot" aria-hidden="true" style="background:${o.identityColor || primaryCasing()}"></span>` +
      `<span class="route-main"><span class="route-name">${escapeHTML(routeTitle(choice))}</span>${selected ? `<span class="route-selected-label">Selected</span>` : ""}</span>` +
      `<span class="route-time">${fmt(r.head)}</span><span class="route-facts">${routeFactsHTML(choice, false)}</span>` +
      `${routeTradeoffsHTML(choice, recommended)}</button>` +
      `<button type="button" class="route-plan-entry" data-route-plan-for="${escapeHTML(o.key)}" aria-controls="route-plan-panel" ` +
      `aria-expanded="${selected && getPlanOpen()}" aria-label="Open route plan for ${escapeHTML(routeTitle(choice))}">Route plan</button></article>`;
  }
  function practicalChoices(choices, recommended) {
    const candidates = representativeChoices(choices).filter((c) => c !== recommended).sort((a, b) => (a.r.head - b.r.head) || (a.r.frag - b.r.frag));
    const picked = [], shownTitles = new Set(recommended ? [routeTitle(recommended)] : []);
    for (const c of candidates) {
      const title = routeTitle(c); if (shownTitles.has(title)) continue;
      shownTitles.add(title); picked.push(c); if (picked.length === 3) break;
    }
    return picked;
  }
  function featuredChoices(choices, recommended) { return [recommended, ...practicalChoices(choices, recommended)].filter(Boolean); }
  const COMPACT_MAP_ROUTE_LIMIT = 4;
  function compactMapChoices(choices, recommended, selected) {
    const compact = [], seen = new Set();
    const add = (choice) => { const key = choice && choice.o && choice.o.key;
      if (choice && key != null && !seen.has(key) && compact.length < COMPACT_MAP_ROUTE_LIMIT) { seen.add(key); compact.push(choice); } };
    add(recommended); add(selected); featuredChoices(choices, recommended).forEach(add); return compact;
  }
  function mapRouteToggleLabel(routeCount) {
    const count = routeCount == null ? routeChoices().length : routeCount;
    return getShowAll() ? "Show featured routes on map" : `Show all ${count} routes on map`;
  }
  function mapRouteToggleHTML(choices, recommended, selected) {
    if (compactMapChoices(choices, recommended, selected).length >= choices.length) return "";
    return `<div class="pin-links"><button class="show-routes" type="button" aria-pressed="${getShowAll()}">${mapRouteToggleLabel(choices.length)}</button></div>`;
  }
  function moreRouteChoices(choices, recommended) {
    const featured = new Set(featuredChoices(choices, recommended).map((c) => c.o.key)); return choices.filter((c) => !featured.has(c.o.key));
  }
  function choiceBoardingParts(choice) {
    const first = (optLegs(choice && choice.o) || []).find((g) => g && g.mode === "transit") || {};
    return { board: first.board && first.board.name ? displayStopName(first.board.name) : "", toward: first.toward ? String(first.toward) : "" };
  }
  function choiceBoardingContext(choice) {
    const parts = choiceBoardingParts(choice); return [parts.board ? `Board at ${parts.board}` : "", parts.toward ? `toward ${parts.toward}` : ""].filter(Boolean).join(", ");
  }
  function boardingGroupLabel(family, members, duplicateName) {
    const name = (family && family.meta && family.meta.name) || "Additional boarding option", context = choiceBoardingContext((members || [])[0]);
    if (context) return `${context}, ${name}`;
    if (duplicateName && family && family.meta && family.meta.sub) return `${family.meta.sub}, ${name}`;
    return name;
  }
  function boardingHeadingHTML(family, members, duplicateName) {
    const name = (family && family.meta && family.meta.name) || "Additional boarding option", parts = choiceBoardingParts((members || [])[0]);
    if (parts.board) {
      const detail = [parts.toward ? `Toward ${parts.toward}` : "", `Services: ${name}`].filter(Boolean).join(", ");
      return `<span class="boarding-place">${escapeHTML(`Board at ${parts.board}`)}</span><span class="boarding-detail">${escapeHTML(detail)}</span>`;
    }
    return `<span class="boarding-place">${escapeHTML(boardingGroupLabel(family, members, duplicateName))}</span>`;
  }
  function additionalChoicesHTML(choices, recommended) {
    const groups = new Map();
    (choices || []).forEach((choice) => { const context = choiceBoardingContext(choice), key = `${String(choice.family.key)}\u0000${context}`;
      if (!groups.has(key)) groups.set(key, { family: choice.family, context, members: [] }); groups.get(key).members.push(choice); });
    const names = {}; groups.forEach((group) => { const name = (group.family.meta && group.family.meta.name) || "Additional boarding option"; names[name] = (names[name] || 0) + 1; });
    return [...groups.values()].map((group) => { const name = (group.family.meta && group.family.meta.name) || "Additional boarding option";
      return `<section class="boarding-group" data-family="${escapeHTML(group.family.key)}" data-boarding-context="${escapeHTML(group.context)}"><h3 class="boarding-heading">${boardingHeadingHTML(group.family, group.members, names[name] > 1)}</h3><div class="choice-list">${group.members.map((c) => routeRowHTML(c, recommended, "additional-route")).join("")}</div></section>`; }).join("");
  }
  function routeActionsHTML(choice) {
    const legs = optLegs(choice.o) || [], transit = legs.filter((g) => g && g.mode === "transit"), firstServices = serviceNames({ services: branchServiceRows(choice) });
    let rideIndex = 0, step = 0, lastService = "";
    return legs.map((g, index) => { if (!g) return ""; let copy = "", detail = "";
      if (g.mode === "transit") {
        const service = rideIndex === 0 && firstServices.length ? naturalList(firstServices) : lname(g);
        const board = g.board && g.board.name ? ` at ${displayStopName(g.board.name)}` : "";
        const toward = g.toward && !(rideIndex === 0 && firstServices.length > 1) ? ` toward ${g.toward}` : "";
        const verb = rideIndex === 0 ? `Board ${service}` : (service === lastService ? `Reboard ${service}` : `Transfer to ${service}`);
        copy = `${verb}${board}${toward}`; const ride = Number.isFinite(+g.min) ? `Ride ${fmt(g.min)}` : "";
        const alight = g.alight && g.alight.name ? `get off at ${displayStopName(g.alight.name)}` : "";
        detail = [ride, alight].filter(Boolean).join(", "); lastService = service; rideIndex++;
      } else if (g.mode === "walk") {
        const next = legs.slice(index + 1).find((leg) => leg && leg.mode === "transit");
        const destination = next && next.board && next.board.name ? ` to ${displayStopName(next.board.name)}` : (rideIndex >= transit.length && transit.length ? " to your workplace" : "");
        const physical = legPhysicalMin(g), allowance = legScheduleAllowanceMin(g);
        if (physical > 0) copy = `Walk ${fmt(physical)}${destination}`; else if (allowance > 0) copy = `Allow ${fmt(allowance)} before boarding`;
        if (allowance > 0 && physical > 0) detail = `Allow ${fmt(allowance)} before boarding`;
      }
      if (!copy) return ""; step++;
      return `<li><span class="route-step" aria-hidden="true">${step}</span><span class="route-copy"><span>${escapeHTML(copy)}</span>${detail ? `<small class="route-detail">${escapeHTML(detail)}</small>` : ""}</span></li>`;
    }).join("");
  }
  function routeSequence(choice) {
    const sequence = [], legs = optLegs(choice && choice.o) || [], firstServices = serviceNames({ services: branchServiceRows(choice) }); let rideIndex = 0;
    legs.forEach((g) => { if (!g) return; let label = ""; if (g.mode === "transit") { label = rideIndex === 0 && firstServices.length ? firstServices.join(" / ") : lname(g); rideIndex++; } else if (g.mode === "walk" && legPhysicalMin(g) > .05) label = "Walk"; if (label && sequence[sequence.length - 1] !== label) sequence.push(label); });
    return sequence.join(" → ") || "Route details";
  }
  function selectedRouteHTML(choice, recommended, d) {
    if (!choice) return ""; const o = choice.o, r = choice.r, directions = routeActionsHTML(choice), stepCount = (directions.match(/<li\b/g) || []).length;
    return `<section class="route-plan-pane selected-route" id="route-plan-panel" data-selected-choice-key="${escapeHTML(o.key)}" aria-labelledby="selected-route-title"><header class="plan-head"><div class="plan-heading"><h2 class="plan-title" id="selected-route-title" tabindex="-1">Route plan</h2><div class="plan-route-context"><span class="plan-route-name">${escapeHTML(routeTitle(choice))}</span><span class="plan-trip-time">${fmt(r.head)}</span></div></div><button type="button" class="plan-close" data-route-plan-close aria-controls="route-plan-panel"><span class="plan-close-desktop">Close directions</span><span class="plan-close-mobile">Back to choices</span></button></header><div class="plan-directions-head"><h3>Step-by-step directions</h3><span class="plan-step-count">${stepCount} step${stepCount === 1 ? "" : "s"}</span></div><ol class="route-directions">${directions || "<li>No detailed steps are available for this route.</li>"}</ol><footer class="plan-footer"><a class="plan-google" href="${getGmaps(d.olat, d.olon)}" target="_blank" rel="noopener"><span>Open in Google Maps</span><span aria-hidden="true">↗</span></a></footer></section>`;
  }
  function compareHTML(d, settings = {}) {
    const choices = routeChoices(), recommended = recommendedChoice(choices), practical = practicalChoices(choices, recommended), more = moreRouteChoices(choices, recommended);
    const noAlts = d && (d.alts || []).length === 0, varianceDown = !!read(settings.varianceFailed, false), typsPending = !varianceDown && !read(settings.departAfter, false) && getMetric() === "r" && (d.alts || []).length > 0 && !(d.alts || []).some((a) => a.real != null);
    const loadMsg = d && d._pinPending ? "loading more route choices…" : (noAlts ? (read(settings.varianceSettled, false) ? "" : "loading alternatives…") : (typsPending ? "loading typical times…" : ""));
    const loading = loadMsg ? `<div class="cmploading" role="status">${loadMsg}</div>` : "";
    const degraded = varianceDown ? `<div class="cmploading variance-degraded" role="status">Bad-day estimates are unavailable. Route choices are still shown.</div>` : "";
    const alternatives = practical.length ? `<section class="route-section alternatives practical-alternatives" aria-labelledby="alt-routes-title"><h2 class="route-section-title" id="alt-routes-title">Good alternatives <span>(${practical.length})</span></h2><div class="choice-list">${practical.map((c) => routeRowHTML(c, recommended, "practical-route")).join("")}</div></section>` : "";
    const moreDetails = more.length ? `<details class="expert" id="allroutes"${read(settings.allRoutesOpen, false) ? " open" : ""}><summary id="all-routes-toggle" aria-expanded="${read(settings.allRoutesOpen, false)}" aria-controls="all-routes-panel">See ${more.length} additional route choice${more.length === 1 ? "" : "s"}</summary><div id="all-routes-panel">${additionalChoicesHTML(more, recommended)}</div></details>` : "";
    const selected = choices.find((c) => c.o.key === getSelectedKey()) || recommended;
    return `<section class="route-choices-pane" id="route-choices-panel" aria-labelledby="routes-title"><section class="route-section recommended route-recommendations"><h2 class="route-section-title" id="routes-title">Recommended route</h2><div class="choice-list">${recommended ? routeRowHTML(recommended, recommended, "recommended") : ""}</div></section>${alternatives}${loading}${degraded}${moreDetails}${mapRouteToggleHTML(choices, recommended, selected)}<div id="route-selection-status" class="sr-only" role="status" aria-live="polite" aria-atomic="true"></div></section>`;
  }
  return { buildRouteChoices, routeChoices, choiceMatchesKey, recommendedChoice, mapChoice, representativeChoices,
    branchServiceRows, routeTitle, routeReasonParts, routeBadDay, routeFactsVM, routeFactsHTML, routeTradeoffsHTML,
    routeRowHTML, practicalChoices, featuredChoices, compactMapChoices, mapRouteToggleLabel, mapRouteToggleHTML,
    moreRouteChoices, choiceBoardingParts, choiceBoardingContext, boardingGroupLabel, boardingHeadingHTML,
    additionalChoicesHTML, routeActionsHTML, routeSequence, selectedRouteHTML, compareHTML };
}
