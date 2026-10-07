import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from skyfield.api import EarthSatellite, load, wgs84

st.set_page_config(page_title="SkySync", layout="wide")

REFRESH_SECONDS = 5
TIMEOUT = 8  # seconds per download attempt
ZOOM_SCALE = 5  # how far the map zooms in on a chosen satellite

# Free AI descriptions (Google Gemini). The key is read from Streamlit Secrets,
# never from this file. If the default model name stops working, put the current
# free model name in Secrets as GEMINI_MODEL.
GEMINI_MODEL = "gemini-flash-latest"
WIKI_HEADERS = {"User-Agent": "SkySync/1.0 (satellite tracker; Streamlit app)"}

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
    "Landsat 9": 49260,
    "Landsat 8": 39084,
    "Vanguard 1": 5,  # written as 5, because Python does not allow 00005
    "GOES-18": 51850,
    "Cartosat 3": 44804,
    "Aura": 28376,
}

# ---------------------------------------------------------------------------
# MANUAL DESCRIPTIONS. A satellite listed here always uses this text.
# A satellite NOT listed here (for example one you just added to TARGETS) gets
# an AI-written description automatically the first time someone selects it.
# Keep each description on one line, avoid double quotes inside, keep the comma.
# ---------------------------------------------------------------------------
DESCRIPTIONS = {
    "ISS": "The International Space Station is a research laboratory built by five space agencies. People have lived aboard it continuously since November 2000, and it circles Earth about every 90 minutes at roughly 400 km up.",
    "Hubble": "Launched in 1990, the Hubble Space Telescope is a joint NASA and ESA mission. Orbiting above the blurring effect of the atmosphere, it has captured some of the sharpest images of distant galaxies, nebulae and stars ever taken.",
    "CSS (Tianhe)": "Tianhe is the core module of China's Tiangong space station, launched in April 2021. It provides living quarters and control systems for the crews who stay aboard and connects to the station's laboratory modules.",
    "Envisat": "Launched by ESA in 2002, Envisat was the largest civilian Earth-observation satellite of its time, studying land, oceans, ice and the atmosphere. Contact was lost in 2012, and it now drifts in orbit as a defunct satellite.",
    "Terra": "NASA's Terra, launched in December 1999, was the flagship of the Earth Observing System. Its instruments, including MODIS, track changes in land, oceans, clouds and air quality across the whole planet.",
    "Aqua": "Launched by NASA in 2002, Aqua studies Earth's water cycle, including evaporation, clouds, rainfall, sea ice and snow cover. It flies in a close formation with other Earth-observing satellites known as the A-Train.",
    "NOAA 15": "NOAA 15 is a polar-orbiting weather satellite launched in 1998. It passes over each region a few times a day, collecting cloud and temperature data, and its weather images have been picked up by amateur radio hobbyists.",
    "NOAA 18": "NOAA 18 is a polar-orbiting weather satellite launched in 2005. Like its siblings, it scans the whole globe as Earth turns beneath it, supporting forecasting, storm tracking and climate records.",
    "NOAA 19": "NOAA 19, launched in 2009, was the last of NOAA's older POES series of polar-orbiting weather satellites. It images clouds, measures temperatures and moisture, and supports search-and-rescue relays.",
    "Sentinel-1A": "Sentinel-1A, launched in 2014 for the European Copernicus programme, carries a radar that can see the ground through clouds and in darkness. It is used to monitor floods, sea ice, ship traffic and ground movement.",
    "Sentinel-2A": "Sentinel-2A, launched in 2015 for the Copernicus programme, takes high-resolution colour and infrared images of land. It is used to track crops, forests, coastlines and the effects of disasters.",
    "Pixxel firefly 1": "Pixxel Firefly 1 is a hyperspectral imaging satellite built by Bengaluru-based Pixxel and launched in early 2025. Instead of ordinary colour photos, it records many narrow bands of light, which helps reveal crop health, minerals, pollution and water quality.",
    "Pixxel firefly 2": "Pixxel Firefly 2 is one of three sister satellites in Pixxel's Firefly constellation, launched together in early 2025. Its hyperspectral camera splits sunlight reflected from the ground into many colours to spot details that normal cameras miss.",
    "Pixxel firefly 3": "Pixxel Firefly 3 is part of the Firefly hyperspectral constellation built by Pixxel, an Indian space-data company. Data from satellites like this supports farming, mining, forestry and environmental monitoring.",
    "Sentinel 3B": "Sentinel-3B, launched in 2018 for the European Copernicus programme, is the twin of Sentinel-3A. It measures sea-surface temperature, sea level, ocean colour and land conditions to monitor the health of oceans and coasts.",
    "Landsat 9": "Landsat 9, launched in 2021 by NASA and the US Geological Survey, continues the longest continuous record of Earth's land seen from space, which began in 1972. Together with Landsat 8 it images the whole planet every eight days or so.",
    "Landsat 8": "Landsat 8, launched in 2013 by NASA and the US Geological Survey, photographs Earth's land in visible, infrared and thermal light. Its images are free to use and help track forests, farmland, cities, glaciers and water supplies.",
    "Vanguard 1": "Vanguard 1, a grapefruit-sized sphere launched by the United States in March 1958, is the oldest human-made object still in orbit. Measuring its path helped scientists learn that Earth is slightly pear-shaped.",
    "GOES-18": "GOES-18, launched in 2022, is a NOAA weather satellite in geostationary orbit about 35,800 km up, so it stays over the same spot on Earth. It was placed to watch the western Americas and the Pacific, tracking storms, wildfires and lightning.",
    "Cartosat 3": "Cartosat-3 is an Indian Earth-observation satellite launched by ISRO in November 2019. It takes very sharp images, with a reported ground detail of about 25 centimetres, used for city planning, mapping and infrastructure.",
    "Aura": "NASA's Aura, launched in 2004, studies the chemistry of Earth's atmosphere, including the ozone layer, air quality and climate. It flies in the A-Train, a line of Earth-observing satellites that follow nearly the same track.",
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
                "Name in data": sat.name,
                "Data": sources[label],
            }
        )
    return now, pd.DataFrame(rows)


# ------------------------ automatic AI descriptions -------------------------
GENERATED_FILE = os.path.join(CACHE_DIR, "generated_descriptions.json")


def get_secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


def launch_year(sat):
    """Launch year from the international designator in the TLE (e.g. 98067A)."""
    try:
        yy = int(sat.model.intldesg.strip()[:2])
        return 1900 + yy if yy >= 57 else 2000 + yy
    except Exception:
        return None


def wiki_extract(label):
    params = {
        "action": "query", "generator": "search", "gsrsearch": f"{label} satellite",
        "gsrlimit": 1, "prop": "extracts", "exintro": 1, "explaintext": 1,
        "exsentences": 6, "redirects": 1, "format": "json",
    }
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params=params, headers=WIKI_HEADERS, timeout=TIMEOUT,
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    if not pages:
        return None, None
    page = sorted(pages.values(), key=lambda p: p.get("index", 0))[0]
    return page.get("title"), (page.get("extract") or "").strip()


def first_sentences(text, n=2):
    return " ".join(re.split(r"(?<=[.!?])\s+", text.strip())[:n])


def gemini_write(label, catnr, year, wiki_title, wiki_text, key):
    prompt = (
        "Write a short description (2 sentences, at most 55 words) of the satellite "
        "below for a public satellite-tracking website.\n"
        "Rules: use only facts from the notes below; do not guess; do not say whether "
        "the satellite is still operating unless the notes say so; plain text only, "
        "no markdown and no quotation marks. If the Wikipedia text is not about this "
        "satellite, ignore it and state only the basics.\n\n"
        f"Satellite: {label}\n"
        f"NORAD ID: {catnr}\n"
        f"Launch year (from its international designator): {year or 'unknown'}\n"
        f"Wikipedia page: {wiki_title or 'none'}\n"
        f"Wikipedia text: {wiki_text or 'none'}"
    )
    model = get_secret("GEMINI_MODEL") or GEMINI_MODEL
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800},
        },
        timeout=20,
    )
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return " ".join(p.get("text", "") for p in parts).strip().replace('"', "")


@st.cache_data(ttl=300, show_spinner=False)  # failures are retried after 5 minutes
def _generate(label, catnr, year):
    """Return (text, kind, note). kind is 'ai', 'wiki' or 'basic'."""
    notes = []
    wiki_title = wiki_text = None
    try:
        wiki_title, wiki_text = wiki_extract(label)
    except Exception as e:
        notes.append(f"wikipedia: {type(e).__name__}")

    key = get_secret("GEMINI_API_KEY")
    if key:
        try:
            out = gemini_write(label, catnr, year, wiki_title, wiki_text, key)
            if out:
                return out, "ai", ""
            notes.append("gemini: empty answer")
        except requests.HTTPError as e:
            notes.append(f"gemini: HTTP {e.response.status_code}")
        except Exception as e:
            notes.append(f"gemini: {type(e).__name__}")
    else:
        notes.append("no GEMINI_API_KEY in Secrets")

    if wiki_text:
        return first_sentences(wiki_text, 2), "wiki", "; ".join(notes)
    basic = f"{label} (NORAD ID {catnr})" + (f", launched in {year}." if year else ".")
    return basic, "basic", "; ".join(notes)


def load_generated():
    try:
        with open(GENERATED_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def get_description(label, catnr, year):
    """Return (text, kind, note). Manual text wins, then saved AI text, then new AI text."""
    manual = DESCRIPTIONS.get(label)
    if manual and manual.strip():
        return manual, "manual", ""
    saved = load_generated().get(str(catnr))
    if saved:
        return saved, "ai", ""
    text, kind, note = _generate(label, catnr, year)
    if kind == "ai":  # only keep real AI text, so adding a key later upgrades the rest
        store = load_generated()
        store[str(catnr)] = text
        try:
            with open(GENERATED_FILE, "w") as f:
                json.dump(store, f)
        except Exception:
            pass
    return text, kind, note


# --------------------------------- display ---------------------------------
def is_dark():
    """True when the page theme is dark (change it in the menu: Settings > Theme)."""
    try:
        return st.context.theme.type != "light"
    except Exception:
        return True


def split_at_dateline(lats, lons):
    """Insert gaps where a path wraps from +180 to -180 so no line crosses the map."""
    out_lat, out_lon = [], []
    for i in range(len(lats)):
        if i > 0 and abs(lons[i] - lons[i - 1]) > 180:
            out_lat.append(None)
            out_lon.append(None)
        out_lat.append(float(lats[i]))
        out_lon.append(float(lons[i]))
    return out_lat, out_lon


def ground_track(sat, now, steps=240):
    """Path on the ground (lat, lon lists) over the satellite's next full orbit."""
    try:
        minutes = 2 * math.pi / sat.model.no_kozai  # orbital period in minutes
    except Exception:
        minutes = 100.0
    minutes = min(max(minutes, 60.0), 1500.0)
    t = ts.tt_jd(now.tt + np.linspace(0, minutes / 1440.0, steps))
    sp = wgs84.subpoint(sat.at(t))
    return split_at_dateline(sp.latitude.degrees, sp.longitude.degrees)


PALETTES = {
    "dark": dict(land="#2b3a42", ocean="#0e1a22", borders="#46565e", text="#e8eef2",
                 dot="#ff4d4d", chosen="#ffd166", outline="#ffffff"),
    "light": dict(land="#e6e2d6", ocean="#cfe3f2", borders="#b5b0a2", text="#1b2733",
                  dot="#d62828", chosen="#f77f00", outline="#1b2733"),
}


def build_map(df, row, track, dark):
    """World map with one dot per satellite. Zooms to `row` and draws its path."""
    p = PALETTES["dark" if dark else "light"]
    names = df["Satellite"].tolist()
    chosen = row["Satellite"] if row is not None else None
    colors = [p["chosen"] if n == chosen else p["dot"] for n in names]
    sizes = [16 if n == chosen else 11 for n in names]

    fig = go.Figure()
    # trace 0: dotted path (empty unless a satellite is chosen)
    fig.add_trace(
        go.Scattergeo(
            lat=track[0], lon=track[1], mode="lines",
            line=dict(width=2, dash="dot", color=p["chosen"]),
            hoverinfo="skip", showlegend=False,
        )
    )
    # trace 1: the satellites themselves (drawn on top of the path)
    fig.add_trace(
        go.Scattergeo(
            lat=df["Lat (°)"],
            lon=df["Lon (°)"],
            text=names,
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=12, color=p["text"]),
            marker=dict(size=sizes, color=colors, line=dict(width=1, color=p["outline"])),
            customdata=df[["Satellite", "Alt (km)"]].values.tolist(),
            hovertemplate="%{customdata[0]}<br>Alt: %{customdata[1]} km<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor=p["land"],
        showocean=True, oceancolor=p["ocean"],
        showcountries=True, countrycolor=p["borders"],
        showframe=False, bgcolor="rgba(0,0,0,0)",
    )
    if row is not None:
        fig.update_geos(
            center=dict(lat=float(row["Lat (°)"]), lon=float(row["Lon (°)"])),
            projection_scale=ZOOM_SCALE,
        )
    # Keeps the view the visitor rotated or zoomed to across refreshes. The value
    # changes when a different satellite is chosen, which resets the view on purpose.
    revision = chosen or "world"
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", uirevision=revision,
    )
    fig.update_geos(uirevision=revision)
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
    if pt.get("curve_number", 1) != 1:  # trace 0 is the dotted path, not a satellite
        return None
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
        with st.spinner("Writing description..."):
            text, kind, note = get_description(
                label, int(row["NORAD ID"]), launch_year(satellites[label])
            )
        st.markdown(text)
        if kind == "ai":
            st.caption("✨ AI-generated description. It may contain errors.")
        elif kind == "wiki":
            st.caption("Summary from Wikipedia.")
        if DEBUG and note:
            st.caption(f"Description notes: {note}")


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
    st.caption("version 8 · path + theme + steady map")
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

    track = ([], [])
    if row is not None:
        try:
            track = ground_track(satellites[chosen], now)
        except Exception:
            pass  # no path is better than a broken page

    if row is not None:
        map_col, info_col = st.columns([3, 2])
    else:
        map_col, info_col = st.container(), None

    with map_col:
        event = st.plotly_chart(
            build_map(df, row, track, is_dark()),
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            config={"scrollZoom": True},
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
