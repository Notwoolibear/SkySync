import math
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from skyfield.api import EarthSatellite, load, wgs84

st.set_page_config(page_title="SkySync", layout="wide")

REFRESH_SECONDS = 5
TIMEOUT = 8  # seconds per download attempt

# label -> NORAD catalog number
TARGETS = {
    "ISS": 25544,
    "Hubble": 20580,
    "CSS (Tianhe)": 48274,
    "Envisat": 27386,
    "Terra": 25994,
    "Aqua": 27424,
    "NOAA 15": 25338,
    "NOAA 18": 28654,
    "NOAA 19": 33591,
    "Sentinel-1A": 39634,
    "Sentinel-2A": 40697,
}

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = os.path.join(HERE, "tles.txt")  # optional file you can add to the repo
CACHE_DIR = os.path.join(os.path.expanduser("~"), "skysync_data")
os.makedirs(CACHE_DIR, exist_ok=True)

ts = load.timescale()


def parse_tle_text(text):
    """Return {catnr: (name, line1, line2)} from 2-line or 3-line TLE text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    found = {}
    for i in range(len(lines) - 1):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            name = ""
            if i > 0 and not lines[i - 1][:2] in ("1 ", "2 "):
                name = lines[i - 1]
            try:
                found[int(lines[i][2:7])] = (name, lines[i], lines[i + 1])
            except ValueError:
                pass
    return found


def from_celestrak(catnr):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=tle"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return parse_tle_text(r.text)[catnr]


def from_backup_api(catnr):
    r = requests.get(f"https://tle.ivanstanojevic.me/api/tle/{catnr}", timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return (j.get("name", ""), j["line1"], j["line2"])


def fetch_one(label, catnr):
    """Try live sources, then a saved copy, then tles.txt. Never raises."""
    errors = []
    cache_path = os.path.join(CACHE_DIR, f"{catnr}.tle")

    for source, nice in ((from_celestrak, "Celestrak"), (from_backup_api, "backup API")):
        try:
            tle = source(catnr)
            with open(cache_path, "w") as f:
                f.write("\n".join(tle) + "\n")
            return label, tle, f"live ({nice})", None
        except Exception as e:
            errors.append(f"{nice}: {type(e).__name__}")

    for path, nice in ((cache_path, "saved copy"), (BUNDLED, "tles.txt")):
        try:
            with open(path) as f:
                return label, parse_tle_text(f.read())[catnr], nice, None
        except Exception:
            pass

    return label, None, None, "; ".join(errors)


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def load_all():
    with ThreadPoolExecutor(max_workers=len(TARGETS)) as pool:
        results = list(pool.map(lambda kv: fetch_one(*kv), TARGETS.items()))

    sats, sources, errors = {}, {}, {}
    for label, tle, source, err in results:
        if tle is None:
            errors[label] = err
            continue
        name, l1, l2 = tle
        sats[label] = EarthSatellite(l1, l2, name or label, ts)
        sources[label] = source
    return sats, sources, errors


def positions(sats, sources):
    now = ts.now()
    rows = []
    for label, sat in sats.items():
        try:
            geo = sat.at(now)
        except Exception:
            continue  # e.g. decayed satellite with an outdated TLE
        sp = wgs84.subpoint(geo)
        vx, vy, vz = geo.velocity.km_per_s
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        rows.append(
            {
                "Satellite": label,
                "Lat (°)": round(sp.latitude.degrees, 4),
                "Lon (°)": round(sp.longitude.degrees, 4),
                "Alt (km)": round(sp.elevation.km, 1),
                "Speed (km/s)": round(speed, 2),
                "Speed (km/h)": round(speed * 3600),
                "TLE age (days)": round(now.tt - sat.epoch.tt, 1),
                "Data": sources[label],
            }
        )
    return now, pd.DataFrame(rows)


st.title("🛰️ SkySync")
st.caption("version 3 · plotly map")

with st.spinner("Loading orbital data (up to ~20 seconds)..."):
    satellites, sources, errors = load_all()

if errors:
    st.warning(
        "Could not load: "
        + ", ".join(f"{k} ({v})" for k, v in errors.items())
    )
    if st.button("Retry failed downloads"):
        load_all.clear()
        st.rerun()

if not satellites:
    st.error(
        "No satellites could be loaded. The server may be blocked from "
        "downloading orbital data. See the note about tles.txt."
    )
    st.stop()

if any(s in ("saved copy", "tles.txt") for s in sources.values()):
    st.info(
        "Some satellites are using saved orbital data instead of a live download. "
        "Positions get less accurate as the 'TLE age' grows."
    )


@st.fragment(run_every=REFRESH_SECONDS)
def live_view():
    now, df = positions(satellites, sources)
    if df.empty:
        st.error("No positions could be calculated.")
        return

    st.caption(
        f"Last updated: {now.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC · "
        f"refreshes every {REFRESH_SECONDS}s · tracking {len(df)} satellites"
    )

    fig = go.Figure(
        go.Scattergeo(
            lat=df["Lat (°)"],
            lon=df["Lon (°)"],
            text=df["Satellite"],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=12),
            marker=dict(size=9, color="red", line=dict(width=1, color="white")),
            customdata=df["Alt (km)"],
            hovertemplate="%{text}<br>Alt: %{customdata} km<extra></extra>",
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#2b3a42",
        showocean=True,
        oceancolor="#0e1a22",
        showcountries=True,
        countrycolor="#46565e",
        showframe=False,
    )
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df, hide_index=True, width="stretch")


live_view()
