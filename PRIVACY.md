# Privacy

Last updated: 2026-08-23

SF Commute Isochrone has no accounts, sign-in flow, advertising audience, or
user profile system. It does not sell personal information.

## Information used to provide the map

When you enter a workplace address, the browser sends the text to this site's
`/geocode` or `/autocomplete` endpoint. The server resolves the lookup through
the configured geocoding provider when it is not already in its process-memory
cache:

- **Geoapify** when the operator configures it with a key;
- otherwise **Photon**; or
- **Nominatim** only when the operator explicitly configures that provider.

The server keeps a bounded, in-memory lookup cache for repeated geocoding and
autocomplete queries. It also keeps bounded, in-memory commute-result caches
keyed by destination coordinates while the process is running. These caches are
operational caches, not user accounts, and are cleared when the process restarts.
A self-hosted/offline `destination.py` workflow can use a local, gitignored
destination cache; the public application does not inject that operator value
into the page.

Provider requests are subject to the provider's own terms and privacy
practices. Do not enter an address you would not want sent to the configured
geocoding provider.

## Browser storage and shared links

The application uses `localStorage` in your browser to restore:

- your most recently entered workplace (coordinates and label, under `wp_v1`);
- map color strength (`map_opacity_v1`); and
- the appearance preference (`theme_v1`).

The browser also writes the current commute state into the URL fragment (the
part after `#`). That fragment can include the workplace coordinates and label,
along with map and display settings. Fragments are useful for sharing a commute
view, but treat a copied URL as potentially sensitive. The fragment is normally
not sent in the HTTP request to this site, although it can be exposed if you
share or paste the URL into another service.

You can remove this data by clearing this site's browser storage and deleting
the URL fragment. Choosing another workplace replaces the saved workplace.

## External network services

The hosted service is delivered through **Cloudflare**. Map tiles are requested
from **CARTO**, and the basemap includes **OpenStreetMap** data and attribution.
Those services receive ordinary web requests necessary to deliver the site or
map tiles, and have their own privacy policies. Leaflet is loaded from the
unpkg CDN.

## Logs and retention

The current production deployment sends Caddy access logs to systemd-journald.
Before encoding each record, Caddy replaces address text, route-cell identifiers,
and origin/destination coordinate query parameters; masks client IP addresses to
a network prefix; and removes forwarded-IP headers. Access records can still
include the request time, path, response status, user agent, referrer, and
non-location settings such as walk pace or transfer count. Application diagnostic
messages do not include workplace or origin coordinates. Cloudflare may also keep
edge request logs and analytics under its own policies. This project does not
currently publish a fixed retention schedule; operators of self-hosted copies
control their own logs and retention.

## Changes

This document describes the repository's current hosted configuration, not a
promise that every fork or deployment uses the same providers or logging setup.
Material changes should be reflected here.
