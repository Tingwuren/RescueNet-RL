const EARTH_RADIUS_KM = 6371.0088;

function toRadians(value) {
  return (Number(value) * Math.PI) / 180;
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  const lat1Rad = toRadians(lat1);
  const lat2Rad = toRadians(lat2);

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1Rad) * Math.cos(lat2Rad) * Math.sin(dLon / 2) ** 2;

  return 2 * EARTH_RADIUS_KM * Math.asin(Math.sqrt(a));
}

export function formatDistance(distanceKm) {
  if (!Number.isFinite(distanceKm) || distanceKm < 0) {
    return "--";
  }
  if (distanceKm < 1) {
    return `${Math.round(distanceKm * 1000)} m`;
  }
  return `${distanceKm.toFixed(distanceKm >= 10 ? 1 : 2)} km`;
}

export function buildRegionMetrics(regionGrid) {
  if (!regionGrid?.geo_bounds || !regionGrid?.rows || !regionGrid?.cols) {
    return null;
  }

  const { lat_min: latMin, lat_max: latMax, lon_min: lonMin, lon_max: lonMax } = regionGrid.geo_bounds;
  const rows = Number(regionGrid.rows);
  const cols = Number(regionGrid.cols);

  if (![latMin, latMax, lonMin, lonMax, rows, cols].every(Number.isFinite) || rows <= 0 || cols <= 0) {
    return null;
  }

  const midLat = (latMin + latMax) / 2;
  const midLon = (lonMin + lonMax) / 2;
  const widthKm = haversineKm(midLat, lonMin, midLat, lonMax);
  const heightKm = haversineKm(latMin, midLon, latMax, midLon);

  return {
    widthKm,
    heightKm,
    cellWidthKm: widthKm / cols,
    cellHeightKm: heightKm / rows,
  };
}
