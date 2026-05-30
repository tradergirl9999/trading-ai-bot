"""
News map — geocode location mentions and show an interactive map.
Tries folium+geopy first, falls back to a self-contained Leaflet HTML.
"""
import os
import json
import webbrowser
import tempfile

_MAP_FILE = os.path.join(os.path.dirname(__file__), "..", "news_map.html")


def _geocode(locations: list[str]) -> list[dict]:
    """Return [{name, lat, lon}, ...] for each successfully geocoded location."""
    results = []
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        geo = Nominatim(user_agent="jarvis-news-map", timeout=6)
        for loc in locations:
            try:
                result = geo.geocode(loc)
                if result:
                    results.append({"name": loc, "lat": result.latitude, "lon": result.longitude})
            except GeocoderTimedOut:
                pass
    except ImportError:
        # Rough country/city bbox lookup fallback (common places only)
        _COORDS = {
            "usa": (38.9, -77.0), "united states": (38.9, -77.0), "us": (38.9, -77.0),
            "uk": (51.5, -0.1), "united kingdom": (51.5, -0.1), "london": (51.5, -0.1),
            "china": (39.9, 116.4), "beijing": (39.9, 116.4), "shanghai": (31.2, 121.5),
            "russia": (55.7, 37.6), "moscow": (55.7, 37.6),
            "germany": (52.5, 13.4), "berlin": (52.5, 13.4),
            "france": (48.8, 2.3), "paris": (48.8, 2.3),
            "japan": (35.7, 139.7), "tokyo": (35.7, 139.7),
            "india": (28.6, 77.2), "new delhi": (28.6, 77.2), "mumbai": (19.1, 72.9),
            "ukraine": (50.4, 30.5), "kyiv": (50.4, 30.5),
            "israel": (31.8, 35.2), "tel aviv": (32.1, 34.8), "gaza": (31.5, 34.5),
            "iran": (35.7, 51.4), "tehran": (35.7, 51.4),
            "north korea": (39.0, 125.7), "south korea": (37.6, 127.0), "seoul": (37.6, 127.0),
            "brazil": (-15.8, -47.9), "new york": (40.7, -74.0),
            "australia": (-33.9, 151.2), "sydney": (-33.9, 151.2),
            "canada": (45.4, -75.7), "ottawa": (45.4, -75.7),
            "middle east": (30.0, 40.0), "europe": (50.0, 10.0), "africa": (0.0, 20.0),
            "asia": (35.0, 100.0), "south america": (-15.0, -60.0),
        }
        for loc in locations:
            key = loc.lower().strip()
            if key in _COORDS:
                results.append({"name": loc, "lat": _COORDS[key][0], "lon": _COORDS[key][1]})
    return results


def _build_leaflet_html(points: list[dict], title: str = "Jarvis News Map") -> str:
    markers_js = ""
    if points:
        lats = [p["lat"] for p in points]
        lons = [p["lon"] for p in points]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        # zoom: tighter if single point, wider if spread
        spread = max(
            max(lats) - min(lats),
            max(lons) - min(lons),
        ) if len(points) > 1 else 0
        zoom = 5 if spread < 10 else 3 if spread < 60 else 2
    else:
        center_lat, center_lon, zoom = 20.0, 0.0, 2

    for p in points:
        markers_js += (
            f'L.circleMarker([{p["lat"]}, {p["lon"]}], '
            f'{{radius: 10, color: "#00e5c8", fillColor: "#00e5c8", '
            f'fillOpacity: 0.7, weight: 2}}).bindPopup('
            f'\'<b style="color:#00e5c8">{p["name"]}</b>\').addTo(map);\n'
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin: 0; background: #030c0e; font-family: Consolas, monospace; }}
  #map {{ height: 100vh; width: 100%; filter: hue-rotate(160deg) brightness(0.85) saturate(1.4); }}
  #title {{ position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
            z-index: 9999; background: rgba(3,12,14,0.85); color: #00e5c8;
            padding: 8px 24px; border: 1px solid #00e5c8; letter-spacing: 3px;
            font-size: 13px; }}
  #count {{ position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
            z-index: 9999; background: rgba(3,12,14,0.7); color: #009988;
            padding: 4px 16px; font-size: 11px; letter-spacing: 2px; }}
</style>
</head>
<body>
<div id="title">◈ JARVIS — NEWS MAP</div>
<div id="count">{len(points)} LOCATION{"S" if len(points) != 1 else ""} DETECTED</div>
<div id="map"></div>
<script>
var map = L.map('map', {{zoomControl: true}}).setView([{center_lat}, {center_lon}], {zoom});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© OpenStreetMap contributors',
  maxZoom: 18
}}).addTo(map);
{markers_js}
</script>
</body>
</html>"""


def show_news_map(locations: list[str]) -> str:
    """Geocode locations from news and open an interactive Leaflet map."""
    if not locations:
        return "No locations provided to map."

    points = _geocode(locations)

    if not points:
        # Last resort: open Google Maps search for the first location
        query = locations[0].replace(" ", "+")
        webbrowser.open(f"https://www.google.com/maps/search/{query}")
        return f"Could not geocode locations. Opened Google Maps for '{locations[0]}'."

    # Try folium first for a nicer map
    try:
        import folium
        m = folium.Map(
            location=[points[0]["lat"], points[0]["lon"]],
            zoom_start=4,
            tiles="CartoDB dark_matter",
        )
        for p in points:
            folium.CircleMarker(
                location=[p["lat"], p["lon"]],
                radius=10,
                color="#00e5c8",
                fill=True,
                fill_color="#00e5c8",
                fill_opacity=0.7,
                popup=folium.Popup(p["name"], max_width=200),
                tooltip=p["name"],
            ).add_to(m)
        if len(points) > 1:
            lats = [p["lat"] for p in points]
            lons = [p["lon"] for p in points]
            m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        m.save(_MAP_FILE)
    except ImportError:
        html = _build_leaflet_html(points)
        with open(_MAP_FILE, "w", encoding="utf-8") as f:
            f.write(html)

    abs_path = os.path.abspath(_MAP_FILE)
    webbrowser.open(f"file:///{abs_path}")
    names = ", ".join(p["name"] for p in points)
    return f"Map opened with {len(points)} location(s): {names}."
