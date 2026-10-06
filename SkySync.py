import math
import os
import time
import logging
from importlib import import_module

try:
    skyfield_api = import_module("skyfield.api")
    Loader = skyfield_api.Loader
    wgs84 = skyfield_api.wgs84
except ImportError as exc:
    raise SystemExit(
        "Skyfield is not installed. Run: python -m pip install skyfield"
    ) from exc

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("satellite_tracker")

TARGETS = {
    "ISS": {
        "url": (
            "https://celestrak.org/NORAD/elements/gp.php"
            "?GROUP=stations&FORMAT=tle"
        ),
      
        "filename": "stations.tle",
        "match": lambda name: "ISS" in name.upper(),
    },
    "Hubble": {
        "url": (
            "https://celestrak.org/NORAD/elements/gp.php"
            "?CATNR=20580&FORMAT=tle"
        ),
        "filename": "hst.tle",
        "match": lambda name: "HST" in name.upper() or "HUBBLE" in name.upper(),
    },
    "CSS": {
        "url": (
            "https://celestrak.org/NORAD/elements/gp.php"
            "?GROUP=active&FORMAT=tle"
        ),
        "filename": "css.tle",
        "match": lambda name: "CSS" in name.upper(),
    },
    "Envisat": {
        "url": (
            "https://celestrak.org/NORAD/elements/gp.php"
            "?CATNR=27386&FORMAT=tle"
        ),
        "filename": "envisat.tle",
        "match": lambda name: "ENVISAT" in name.upper(),
    },
    "Terra": {
        "url": (
            "https://celestrak.org/NORAD/elements/gp.php"
            "?CATNR=25994&FORMAT=tle"
        ),
        "filename": "terra.tle",
        "match": lambda name: "TERRA" in name.upper(),
    },
    "Aqua": {
        "url": (
            "https://celestrak.org/NORAD/elements/gp.php"
            "?CATNR=27424&FORMAT=tle"
        ),
        "filename": "aqua.tle",
        "match": lambda name: "AQUA" in name.upper(),
    },
    "NOAA 15": {
        "url": (    
            "https://celestrak.org/NORAD/elements/gp.php"
            "?CATNR=25338&FORMAT=tle"
        ),
        "filename": "noaa15.tle",
        "match": lambda name: "NOAA 15" in name.upper(),
    },
    "NOAA 18": {
        "url": (    
            "https://celestrak.org/NORAD/elements/gp.php"
            "?CATNR=28654&FORMAT=tle"
        ),
        "filename": "noaa18.tle",
        "match": lambda name: "NOAA 18" in name.upper(),
    },
    "NOAA 19": {
        "url": (    
            "https://celestrak.org/NORAD/elements/gp.php"
            "?CATNR=33591&FORMAT=tle"
        ),
        "filename": "noaa19.tle",
        "match": lambda name: "NOAA 19" in name.upper(),
    },
    "Sentinel-1A": {
        "url": (    
            "https://celestrak.org/NORAD/elements/gp.php"
            "?CATNR=39634&FORMAT=tle"
        ),
        "filename": "sentinel1a.tle",
        "match": lambda name: "SENTINEL-1A" in name.upper(),

    },
    "Sentinel-2A": {
        "url": (    
            "https://celestrak.org/NORAD/elements/gp.php"
            "?CATNR=40697&FORMAT=tle"
        ),
        "filename": "sentinel2a.tle",
        "match": lambda name: "SENTINEL-2A" in name.upper(),
    }
}


def load_satellite(custom_load, label, url, filename, match_fn):
    """Fetch a TLE feed and return the first satellite whose name matches."""
    log.debug(f"[{label}] TLE source URL = {url} -> cached as {filename}")

    try:
        satellites = custom_load.tle_file(url, filename=filename)
        log.debug(f"[{label}] tle_file() returned {len(satellites)} records")
    except Exception as e:
        log.exception(f"[{label}] Network or parsing error downloading TLE data")
        print(f"[{label}] Network error downloading TLE data: {e}")
        return None

    if not satellites:
        log.error(f"[{label}] tle_file() returned an empty list")
        print(f"[{label}] No satellite records were returned.")
        return None

    by_name = {sat.name: sat for sat in satellites}
    log.debug(f"[{label}] Loaded {len(by_name)} named satellites, "
              f"first few: {list(by_name.keys())[:5]}")

    match_name = next((name for name in by_name if match_fn(name)), None)

    if not match_name:
        log.error(f"[{label}] No matching satellite name found in the dataset")
        print(f"[{label}] Data not found in the Celestrak stream.")
        return None

    sat = by_name[match_name]
    log.debug(f"[{label}] Selected satellite object: {sat}")
    print(f"[{label}] Successfully loaded orbital elements for: {sat.name}")
    return sat


def track_satellites():
    base_dir = os.path.expanduser('~')
    data_dir = os.path.join(base_dir, 'skyfield_data')
    log.debug(f"Resolved base_dir = {base_dir}")

    os.makedirs(data_dir, exist_ok=True)
    data_dir = data_dir.rstrip('\\/')
    log.debug(f"Skyfield cache directory = {data_dir}")

    custom_load = Loader(data_dir)
    ts = custom_load.timescale()

    print(f"Fetching live satellite data (caching to: {data_dir})...")
    satellites = {}
    for label, target in TARGETS.items():
        sat = load_satellite(
            custom_load, label, target["url"], target["filename"], target["match"]
        )
        if sat is not None:
            satellites[label] = sat

    if not satellites:
        print("No satellites could be loaded. Aborting.")
        return

    print("-" * 50)

    try:
        while True:
            now = ts.now()
            print(f"Time (UTC): {now.utc_strftime('%Y-%m-%d %H:%M:%S')}")

            for label, sat in satellites.items():
                geocentric = sat.at(now)
                subpoint = wgs84.subpoint(geocentric)

                lat = subpoint.latitude.degrees
                lon = subpoint.longitude.degrees
                alt_km = subpoint.elevation.km

                vx, vy, vz = geocentric.velocity.km_per_s
                speed_km_s = math.sqrt(vx**2 + vy**2 + vz**2)
                speed_kmh = speed_km_s * 3600

                log.debug(
                    f"[{label}] raw geocentric={geocentric.position.km}, "
                    f"lat={lat}, lon={lon}, alt_km={alt_km}, "
                    f"speed_km_s={speed_km_s}"
                )

                print(f"  {label:8s} | Lat {lat:8.4f}\u00b0 | "
                      f"Lon {lon:8.4f}\u00b0 | Alt {alt_km:8.2f} km | "
                      f"Speed {speed_km_s:6.2f} km/s ({speed_kmh:8.0f} km/h)")

            print("-" * 50)
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nTracking stopped by user.")
        log.info("Tracking loop exited via KeyboardInterrupt")


if __name__ == "__main__":
    track_satellites()
