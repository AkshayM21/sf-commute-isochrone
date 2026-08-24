/* Pure request and response lifecycle primitives shared by the browser adapters.
 *
 * Keep endpoint spelling and query ordering here: the server and the permalink/cache
 * contracts intentionally depend on these strings remaining byte-stable.
 */

export const OUTSIDE_AREA_CODE = "outside_supported_area";
export const OUTSIDE_AREA_MESSAGE =
  "That workplace is outside the supported San Francisco walking area. Choose a workplace in San Francisco.";

export function ridesQuery(maxxfers) {
  if (maxxfers === "any") return "";
  const n = +maxxfers;
  return Number.isFinite(n) ? `&maxrides=${n + 1}` : "";
}

export function speedQuery(walkspeed, speedtoggle = true) {
  return walkspeed === "med" || !speedtoggle ? "" : `&speed=${walkspeed}`;
}

export function routingQuery({ maxxfers = "any", walkspeed = "med", speedtoggle = true } = {}) {
  return ridesQuery(maxxfers) + speedQuery(walkspeed, speedtoggle);
}

export function computeURL({ lat, lon, maxxfers = "any", walkspeed = "med", speedtoggle = true } = {}) {
  return `/compute?lat=${lat}&lon=${lon}${routingQuery({ maxxfers, walkspeed, speedtoggle })}`;
}

export function varianceURL({ dlat, dlon, maxxfers = "any", walkspeed = "med", speedtoggle = true } = {}) {
  return `/variance?dlat=${dlat}&dlon=${dlon}${routingQuery({ maxxfers, walkspeed, speedtoggle })}`;
}

export function attributionURL({ dlat, dlon, maxxfers = "any", walkspeed = "med", speedtoggle = true } = {}) {
  return `/attribution?dlat=${dlat}&dlon=${dlon}${routingQuery({ maxxfers, walkspeed, speedtoggle })}`;
}

export function itineraryURL({ id, dlat, dlon, maxxfers = "any", walkspeed = "med", speedtoggle = true, pin = false } = {}) {
  return `/itinerary?id=${id}&dlat=${dlat}&dlon=${dlon}${routingQuery({ maxxfers, walkspeed, speedtoggle })}${pin ? "&pin=1" : ""}`;
}

export function geocodeURL(query) {
  return `/geocode?q=${encodeURIComponent(query)}`;
}

export function routeRequestKey({ generation, id, destination, maxxfers = "any", walkspeed = "med" } = {}) {
  return `${generation}|${id}|${destination && destination[0]}|${destination && destination[1]}|${maxxfers}|${walkspeed}`;
}

export function isOutsideSupportedArea(status, body) {
  return status === 422 && !!body && body.error === OUTSIDE_AREA_CODE;
}

export function classifyHTTPError(status, body, kind = "request") {
  if (isOutsideSupportedArea(status, body)) {
    return { code: OUTSIDE_AREA_CODE, message: OUTSIDE_AREA_MESSAGE };
  }
  const label = kind === "compute" ? "compute request" : kind === "itinerary" ? "Route lookup" : kind;
  return { code: "http_error", message: `${label} failed (${status}).` };
}

export async function parseJSONResponse(response) {
  try {
    return await response.json();
  } catch (_error) {
    return null;
  }
}

export function normalizeVariancePayload(payload) {
  const realistic = (payload && payload.realistic) || {};
  const variance = (payload && payload.variance) || {};
  return {
    realistic,
    variance,
    hasData: Object.keys(realistic).length > 0 || Object.keys(variance).length > 0,
  };
}

export function retryDelayMs(retryAfterHeader, fallbackSeconds = 4) {
  const seconds = retryAfterHeader == null ? NaN : parseFloat(retryAfterHeader);
  return (Number.isFinite(seconds) && seconds > 0 ? seconds : fallbackSeconds) * 1000;
}

export function shouldRetry(status, attempt = 0, maxRetries = 1) {
  return (status === 429 || status === 503) && attempt < maxRetries;
}

export function isCurrentGeneration(requestGeneration, currentGeneration) {
  return requestGeneration === currentGeneration;
}

export function shouldAbortStaleRequest({ requestGeneration, currentGeneration, signal } = {}) {
  return !!(signal && signal.aborted) || !isCurrentGeneration(requestGeneration, currentGeneration);
}

export function isCurrentRequest({ token, currentToken, requestGeneration, currentGeneration, signal } = {}) {
  return !shouldAbortStaleRequest({ requestGeneration, currentGeneration, signal }) && token === currentToken;
}
