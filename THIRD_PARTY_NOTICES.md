# Third-party notices and data attribution

SF Commute Isochrone combines public transportation, geographic, elevation,
and geocoding data. Source data changes over time; users should verify service
conditions with the original provider before relying on a commute estimate.

## Transit schedules

- **Muni and Caltrain GTFS** are obtained from [511 SF Bay Open
  Data](https://511.org/open-data). They remain subject to 511's terms and the
  applicable transit-agency data terms. Schedules, routes, and service alerts
  can change without notice.
- **BART GTFS** is obtained from the [BART developer
  feed](https://www.bart.gov/dev/schedules/google_transit.zip), subject to
  BART's published terms.

GTFS data is used to model scheduled service; it is not a guarantee of actual
arrival, transfer, accessibility, or service availability.

## Map and routing data

- © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
  OpenStreetMap data is available under the [Open Database License
  (ODbL)](https://opendatacommons.org/licenses/odbl/). The project downloads a
  Geofabrik extract and clips it for local routing. If you distribute a
  derivative database, review the ODbL's attribution, notice, and share-alike
  obligations.
- Basemap tiles are provided by [CARTO](https://carto.com/attributions), using
  OpenStreetMap data. Use of CARTO tiles is subject to CARTO's terms and
  attribution requirements.
- Elevation inputs come from the [USGS 3D Elevation Program
  (3DEP)](https://www.usgs.gov/3d-elevation-program). USGS data is generally
  public domain, but names and logos are not an endorsement and may have
  separate use rules.
- Neighborhood boundaries are built using [DataSF](https://data.sfgov.org/)
  datasets, including SF Find Neighborhoods and Realtor Neighborhoods. Follow
  the individual dataset pages for current metadata, license, and attribution
  requirements.

## Geocoding and autocomplete

The configured provider may be Geoapify, Photon, or Nominatim. Their results,
coverage, usage limits, attribution, caching permissions, and terms differ:

- [Geoapify terms and attribution](https://www.geoapify.com/terms/)
- [Photon](https://photon.komoot.io/) and its OpenStreetMap-derived data;
- [Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/)
  and OpenStreetMap attribution.

Forks and deployments must choose a provider and comply with its current terms,
including rate limits and any display, caching, or attribution requirements.
This project sends a descriptive User-Agent and throttles upstream requests,
but that does not replace a provider's policy obligations.

## Software dependencies

This repository also uses open-source software, including Python libraries,
Numba, Flask, Leaflet, and their transitive dependencies. Their license notices
are supplied by their respective distributions. See dependency metadata in the
environment you install for the complete, version-specific notices.
