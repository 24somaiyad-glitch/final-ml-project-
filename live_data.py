"""Real-time data sources. Every function fails safely (returns None) so the app never crashes offline.
1) Clock          -> hour and day (no API needed)
2) Open-Meteo     -> live weather, FREE, no API key
3) TomTom Traffic -> live road speed, free tier, needs an API key (optional)
4) Google Routes  -> live traffic LEVEL + delay for the chosen area (needs a Google key, Routes API enabled)
5) Google Maps    -> live traffic map for the chosen area (JavaScript TrafficLayer needs GOOGLE_MAPS_KEY;
                     without a key a keyless embed + an "open in Google Maps" link are used)"""
import os, json, datetime as dt
import requests
from zoneinfo import ZoneInfo

TZ = "Asia/Kolkata"                       # EDIT: your time zone
CITY_LAT, CITY_LON = 21.1458, 79.0882     # EDIT: your city centre
# EDIT: one sample point (lat, lon) on a main road of each area, used only for TomTom live speed
AREA_POINTS = {"Downtown": (21.1498, 79.0806), "Highway": (21.1700, 79.0500),
               "Residential": (21.1200, 79.1000), "Industrial": (21.0900, 79.0600)}

def now_local():
    return dt.datetime.now(ZoneInfo(TZ))

def weather_label(code, precip):
    if precip and precip > 0.1: return "Rain"
    if code in (45, 48): return "Fog"
    if code in range(51, 100): return "Rain"      # drizzle, rain, showers, thunderstorm (snow n/a)
    if code in (2, 3): return "Cloudy"
    return "Clear"

def get_live_weather(lat=CITY_LAT, lon=CITY_LON, timeout=5):
    """Returns {'weather': 'Rain', 'temp_c': 27.1} or None if offline."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=timeout,
                         params={"latitude": lat, "longitude": lon, "current": "temperature_2m,weather_code,precipitation"})
        c = r.json()["current"]
        return {"weather": weather_label(c["weather_code"], c.get("precipitation", 0)), "temp_c": c["temperature_2m"]}
    except Exception:
        return None

def get_live_traffic_level(area, key=None, timeout=5):
    """Current congestion from TomTom Flow Segment Data. Returns (level, speed_ratio) or None.
    ratio = current speed / free-flow speed:  >0.75 Low | 0.5-0.75 Medium | <0.5 High"""
    key = key or os.getenv("TOMTOM_KEY")
    if not key or area not in AREA_POINTS: return None
    lat, lon = AREA_POINTS[area]
    try:
        r = requests.get("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json", timeout=timeout,
                         params={"key": key, "point": f"{lat},{lon}"})
        f = r.json()["flowSegmentData"]; ratio = f["currentSpeed"] / f["freeFlowSpeed"]
        return ("Low" if ratio > .75 else "Medium" if ratio > .5 else "High"), round(ratio, 2)
    except Exception:
        return None


# ---------------------------------------------------------------- Google Maps (display only)
def area_point(area):
    return AREA_POINTS.get(area, (CITY_LAT, CITY_LON))

def google_maps_link(area, zoom=14):
    """Plain link that opens Google Maps with the live traffic layer switched on (no key needed)."""
    lat, lon = area_point(area)
    return f"https://www.google.com/maps/@{lat},{lon},{zoom}z/data=!5m1!1e1"      # !5m1!1e1 = traffic layer ON

_MAP_HTML = """<div id="map" style="height:__H__px;width:100%;border-radius:10px;font-family:sans-serif"></div>
<script>
function initMap(){
  var c={lat:__LAT__,lng:__LON__};
  var map=new google.maps.Map(document.getElementById('map'),{center:c,zoom:__Z__,mapTypeControl:false,streetViewControl:false});
  new google.maps.TrafficLayer().setMap(map);                       // live traffic colours
  new google.maps.Marker({position:c,map:map,title:__TITLE__});
}
window.gm_authFailure=function(){document.getElementById('map').innerHTML=
  '<p style="padding:16px">Google rejected the API key. Enable "Maps JavaScript API", turn on billing and check the key restrictions.</p>';};
</script>
<script async src="https://maps.googleapis.com/maps/api/js?key=__KEY__&callback=initMap"></script>"""

def google_traffic_map_html(area, key=None, zoom=14, height=420):
    """Official Google Maps JavaScript map with the live TrafficLayer. Returns None when there is no key."""
    key = key or os.getenv("GOOGLE_MAPS_KEY")
    if not key: return None
    lat, lon = area_point(area)
    return (_MAP_HTML.replace("__H__", str(height)).replace("__LAT__", str(lat)).replace("__LON__", str(lon))
            .replace("__Z__", str(zoom)).replace("__TITLE__", json.dumps(area)).replace("__KEY__", str(key)))

def google_embed_fallback_html(area, zoom=14, height=420):
    """Keyless best-effort embed (unofficial; the traffic overlay may not always show)."""
    lat, lon = area_point(area)
    src = f"https://maps.google.com/maps?ll={lat},{lon}&z={zoom}&t=m&layer=t&output=embed"
    return f'<iframe src="{src}" width="100%" height="{height}" style="border:0;border-radius:10px" loading="lazy"></iframe>'


# ---------------------------------------------------------------- Google live traffic reading (Routes API)
# EDIT (optional): a real short road trip per area as ((lat, lon), (lat, lon)). If an area is missing here,
# a ~3 km trip starting at AREA_POINTS[area] is used.
AREA_ROUTES = {}

def area_route(area):
    if area in AREA_ROUTES: return AREA_ROUTES[area]
    lat, lon = area_point(area)
    return (lat, lon), (lat + 0.02, lon + 0.02)

def _secs(s):
    return float(str(s).rstrip("s"))

def get_google_traffic(area, key=None, timeout=8):
    """Live reading from the Google Routes API (TRAFFIC_AWARE) for a short trip in the area.
    Returns {'level','ratio','trip_min','free_min','distance_km','delay_per_10km'} or {'error': '...'} or None (no key).
    ratio = free-flow time / current time (1.0 = no slowdown): >0.75 Low | 0.5-0.75 Medium | <0.5 High"""
    key = key or os.getenv("GOOGLE_ROUTES_KEY") or os.getenv("GOOGLE_MAPS_KEY")
    if not key: return None
    (o_lat, o_lon), (d_lat, d_lon) = area_route(area)
    body = {"origin": {"location": {"latLng": {"latitude": o_lat, "longitude": o_lon}}},
            "destination": {"location": {"latLng": {"latitude": d_lat, "longitude": d_lon}}},
            "travelMode": "DRIVE", "routingPreference": "TRAFFIC_AWARE"}
    try:
        r = requests.post("https://routes.googleapis.com/directions/v2:computeRoutes", json=body, timeout=timeout,
                          headers={"X-Goog-Api-Key": key, "Content-Type": "application/json",
                                   "X-Goog-FieldMask": "routes.duration,routes.staticDuration,routes.distanceMeters"})
        data = r.json()
        if r.status_code != 200:
            return {"error": str(data.get("error", {}).get("message", "HTTP " + str(r.status_code)))[:160]}
        rt = data["routes"][0]
        cur, free, km = _secs(rt["duration"]), _secs(rt["staticDuration"]), rt["distanceMeters"] / 1000
        ratio = min(1.0, free / cur) if cur > 0 else 1.0
        delay = max(0.0, cur - free) / 60
        return {"level": "Low" if ratio > .75 else "Medium" if ratio > .5 else "High", "ratio": round(ratio, 2),
                "trip_min": round(cur / 60, 1), "free_min": round(free / 60, 1), "distance_km": round(km, 1),
                "delay_per_10km": round(delay / km * 10, 1) if km > 0 else 0.0}
    except Exception as e:
        return {"error": "request failed: " + str(e)[:120]}
