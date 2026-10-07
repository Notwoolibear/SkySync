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
ZOOM_SCALE = 5  # how far the map zooms in on a chosen satellite

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
    "Pixxel firefly 1": 62701,
    "Pixxel firefly 2": 62704,
    "Pixxel firefly 3": 62710,
    "Sentinel 3B": 43437,
}

# ---------------------------------------------------------------------------
# YOUR DESCRIPTIONS: replace the text between the quotes for each satellite.
# The name on the left must match the name in TARGETS above exactly.
# Keep the comma at the end of every line.
# ---------------------------------------------------------------------------
DESCRIPTIONS = {
    "ISS": "Description: The ISS is a large space station in low Earth orbit that serves as a laboratory for scientific research and technology experiments. It is operated through international cooperation and has been continuously inhabited since 2000.",
    "HUBBLE": "Description: will add soon",
    "CSS (Tianhe)": "Description: will add soon",
    "Envisat": "Description: will add soon",
    "TERRA": "Description: will add soon",
    "Aqua": "Description: will add soon",
    "NOAA 15": "Description: will add soon",
    "NOAA 18": "Description: will add soon",
    "NOAA 19": "Description: will add soon",
    "Sentinel-1A": "Description: will add soon",
    "Sentinel-2A": "Description: will add soon",
    "Pixxel firefly 1": "Description: will add soon",
    "Pixxel firefly 2": "Description: will add soon",
    "Pixxel firefly 3": "Description: will add soon",
    "Sentinel 3B": "Description: will add soon",
}

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = os.path.join(HERE, "tles.txt")  # optional file you can add to the repo
CACHE_DIR = os.path.join(os.path.expanduser("~"), "skysync_data")
os.makedirs(CACHE_DIR, exist_ok=True)

ts = load.timescale()


# ------------------------------ data loading -------------------------------
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


# ------------------------------ calculations -------------------------------
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
                "NORAD ID": TARGETS[label],
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


# --------------------------------- display ---------------------------------
def build_map(df, row):
    """World map with one dot per satellite. Zooms to `row` if one is chosen."""
    names = df["Satellite"].tolist()
    chosen = row["Satellite"] if row is not None else None
    colors = ["#ffd166" if n == chosen else "#ff4d4d" for n in names]
    sizes = [16 if n == chosen else 11 for n in names]

    fig = go.Figure(
        go.Scattergeo(
            lat=df["Lat (°)"],
            lon=df["Lon (°)"],
            text=names,
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=12, color="#e8eef2"),
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
            customdata=df[["Satellite", "Alt (km)"]].values.tolist(),
            hovertemplate="%{customdata[0]}<br>Alt: %{customdata[1]} km<extra></extra>",
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
    if row is not None:
        fig.update_geos(
            center=dict(lat=float(row["Lat (°)"]), lon=float(row["Lon (°)"])),
            projection_scale=ZOOM_SCALE,
        )
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def clicked_name(event, df):
    """Name of the satellite whose dot was clicked, or None."""
    try:
        points = event["selection"]["points"]
    except Exception:
        return None
    if not points:
        return None
    pt = points[0]
    name = None
    cd = pt.get("customdata")
    if cd:
        name = cd[0] if isinstance(cd, (list, tuple)) else cd
    if name is None and pt.get("point_index") is not None:
        idx = pt["point_index"]
        if 0 <= idx < len(df):
            name = df.iloc[idx]["Satellite"]
    if name is None:
        name = pt.get("text")
    return name if name in TARGETS else None


def show_info(row):
    label = row["Satellite"]
    with st.container(border=True):
        st.subheader(label)
        a, b = st.columns(2)
        a.metric("NORAD ID", str(int(row["NORAD ID"])))
        b.metric("Altitude", f"{row['Alt (km)']:,.1f} km")
        a, b = st.columns(2)
        a.metric("Speed", f"{row['Speed (km/s)']:.2f} km/s")
        b.metric("Speed", f"{int(row['Speed (km/h)']):,} km/h")
        a, b = st.columns(2)
        a.metric("Longitude", f"{row['Lon (°)']:.4f}°")
        b.metric("Latitude", f"{row['Lat (°)']:.4f}°")
        st.markdown(DESCRIPTIONS.get(label) or "No description yet.")


# ----------------------------------- page ----------------------------------
DEBUG = "debug" in st.query_params  # open your link with ?debug=1 to see data status

st.title("🛰️ SkySync")

with st.spinner("Loading orbital data (up to ~20 seconds)..."):
    satellites, sources, errors = load_all()

if not satellites:
    st.error(
        "No satellites could be loaded. The server may be blocked from "
        "downloading orbital data. Add a tles.txt file to the repository."
    )
    st.stop()

if DEBUG:
    st.caption("version 4 · search + zoom")
    if errors:
        st.warning(
            "Could not load: " + ", ".join(f"{k} ({v})" for k, v in errors.items())
        )

# A click on a dot sets this, and it is applied to the search box before it is drawn.
if "pending_search" in st.session_state:
    st.session_state["search"] = st.session_state.pop("pending_search")
st.session_state.setdefault("map_version", 0)

st.selectbox(
    "Search satellites",
    options=sorted(satellites, key=str.lower),
    index=None,
    placeholder="Search satellites…",
    key="search",
    label_visibility="collapsed",
)


@st.fragment(run_every=REFRESH_SECONDS)
def live_view():
    now, df = positions(satellites, sources)
    if df.empty:
        st.error("No positions could be calculated.")
        return

    row = None
    chosen = st.session_state.get("search")
    if chosen:
        match = df[df["Satellite"] == chosen]
        if not match.empty:
            row = match.iloc[0]

    if row is not None:
        map_col, info_col = st.columns([3, 2])
    else:
        map_col, info_col = st.container(), None

    with map_col:
        event = st.plotly_chart(
            build_map(df, row),
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key=f"map_{st.session_state.map_version}",
        )

    clicked = clicked_name(event, df)
    if clicked:
        st.session_state["pending_search"] = clicked
        st.session_state.map_version += 1  # fresh chart, so the click isn't reused
        st.rerun()

    if info_col is not None:
        with info_col:
            show_info(row)

    if DEBUG:
        st.caption(f"Last updated: {now.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC")
        st.dataframe(df, hide_index=True, width="stretch")


live_view()
