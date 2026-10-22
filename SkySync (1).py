import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from skyfield.api import EarthSatellite, load, wgs84

st.set_page_config(page_title="SkySync", layout="wide")

REFRESH_SECONDS = 5
TIMEOUT = 8  # seconds per download attempt

# Satellites are added AUTOMATICALLY from these NORAD/CelesTrak groups, on top of
# the featured list (TARGETS) below. Group names: celestrak.org/NORAD/elements
# "active" alone already contains Starlink, OneWeb and the geostationary satellites,
# so do not add "starlink", "oneweb" or "geo" as well (they would only duplicate dots).
# Keep this list the same as GROUPS in update_tles.py.
GROUPS = [
    "stations", "visual", "weather", "goes", "resource", "science",
    "gps-ops", "galileo", "glo-ops", "beidou",   # navigation (GPS, Galileo, GLONASS, BeiDou)
    "sarsat", "dmc", "geodetic", "engineering", "military",
    "planet", "spire", "cubesat", "amateur", "education", "radar",   # imaging, small satellites, radio
    "iridium-NEXT", "globalstar", "orbcomm", "intelsat", "ses",      # communications
    "sbas", "musson",                                                # navigation aids
    "active",   # ALL active satellites (~15,000, includes Starlink, OneWeb and geostationary)
]
MAX_AUTO = 30000  # most automatically added satellites to show

# Major / famous satellites: these get the largest dots. Use the names from TARGETS.
FAMOUS = ["ISS", "Hubble", "CSS (Tianhe)", "Vanguard 1", "Envisat", "Landsat 9", "Terra", "Aqua"]

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


def fetch_group(group):
    """Every satellite in a CelesTrak group: live, else a saved copy. {catnr: tle}"""
    path = os.path.join(CACHE_DIR, f"group_{group}.tle")
    try:
        r = requests.get(
            f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle",
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        found = parse_tle_text(r.text)
        if found:
            with open(path, "w") as f:
                f.write(r.text)
            return found
    except Exception:
        pass
    try:
        with open(path) as f:
            return parse_tle_text(f.read())
    except Exception:
        return {}


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def load_all():
    with ThreadPoolExecutor(max_workers=len(TARGETS) + len(GROUPS)) as pool:
        feat_jobs = [pool.submit(fetch_one, l, c) for l, c in TARGETS.items()]
        grp_jobs = [pool.submit(fetch_group, g) for g in GROUPS]
        featured = [j.result() for j in feat_jobs]
        groups = [j.result() for j in grp_jobs]

    found = {}
    for g in groups:
        found.update(g)
    try:  # tles.txt from update_tles.py fills in whatever could not be downloaded
        with open(BUNDLED) as f:
            for k, v in parse_tle_text(f.read()).items():
                found.setdefault(k, v)
    except Exception:
        pass

    sats, sources, errors, tles, meta = {}, {}, {}, {}, {}
    for label, tle, source, err in featured:
        if tle is None:
            errors[label] = err
            continue
        name, l1, l2 = tle
        sats[label] = EarthSatellite(l1, l2, name or label, ts)
        sources[label], tles[label] = source, tle
        meta[label] = {"id": TARGETS[label], "top": True}

    have = {m["id"] for m in meta.values()}
    auto = []
    for catnr, (name, l1, l2) in found.items():
        if catnr in have:
            continue
        label = (name or f"SAT {catnr}").strip()
        auto.append((label, catnr, name, l1, l2))
    auto.sort(key=lambda x: x[0].lower())
    used = set(sats)
    for label, catnr, name, l1, l2 in auto[:MAX_AUTO]:
        if label in used:
            label = f"{label} ({catnr})"
        used.add(label)
        sources[label], tles[label] = "auto", (name, l1, l2)
        meta[label] = {"id": catnr, "top": False}
    return sats, sources, errors, tles, meta


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


# ----------------------------------- page ----------------------------------
# The map, search, details card and light/dark switch all run in the visitor's
# browser (WIDGET_HTML below). Streamlit only hands it the data once, so nothing
# refreshes the map and the view the visitor rotated to is never reset.
WIDGET_HTML = r'''<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0e1117;--fg:#e8eef2;--card:#1a1f2b;--line:#2f3645;--acc:#ffd166}
body.light{--bg:#f6f4ee;--fg:#1b2733;--card:#fff;--line:#d8d3c4;--acc:#f77f00}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif}
#top{display:flex;gap:8px;padding:2px 2px 8px;flex-wrap:wrap}
#sw{position:relative;flex:1 1 220px}
#q{width:100%;padding:10px 34px 10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);font-size:15px}
#clr{position:absolute;right:8px;top:6px;border:0;background:none;color:var(--fg);font-size:20px;cursor:pointer;display:none}
#res{position:absolute;z-index:10;left:0;right:0;top:44px;background:var(--card);border:1px solid var(--line);border-radius:8px;max-height:240px;overflow:auto;display:none}
#res div{padding:8px 12px;cursor:pointer}#res div:hover{background:var(--line)}
.tg{display:flex}
.tg button{padding:10px 12px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;font-size:14px}
.tg button:first-child{border-radius:8px 0 0 8px}.tg button:last-child{border-radius:0 8px 8px 0;border-left:0}
.tg button.on{background:var(--acc);color:#111;border-color:var(--acc)}
#main{display:flex;gap:10px;flex-wrap:wrap}
#mapwrap{flex:3 1 480px;height:560px;min-width:0;position:relative;overflow:hidden;border-radius:10px}
#cv{position:absolute;left:0;top:0;width:100%;height:100%;touch-action:none;cursor:grab}
#tip{position:absolute;pointer-events:none;display:none;padding:4px 8px;border-radius:6px;background:var(--card);border:1px solid var(--line);font-size:12px;white-space:nowrap;z-index:3}
#status{position:absolute;left:8px;bottom:6px;font-size:12px;opacity:.7;pointer-events:none}
#zc{position:absolute;right:8px;top:8px;display:flex;flex-direction:column;gap:4px}
#zc button{width:34px;height:34px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;cursor:pointer;font-size:18px;line-height:1}
#info{flex:2 1 280px;position:relative;z-index:2;display:none;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;align-self:flex-start}
#info h2{margin:0 0 10px;font-size:20px}
.g{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;margin-bottom:12px}
.g small{display:block;opacity:.65;font-size:12px}.g b{font-size:17px}
#desc{line-height:1.45;font-size:14px}#kind{opacity:.6;font-size:12px;margin-top:8px}
</style>
<script src="https://cdn.jsdelivr.net/npm/satellite.js@4.1.3/dist/satellite.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js"></script>
</head><body>
<div id="top"><div id="sw"><input id="q" placeholder="Search satellites…" autocomplete="off"><button id="clr" title="Clear">×</button><div id="res"></div></div>
<div class="tg"><button id="bg" class="on">🌍 Globe</button><button id="bf">🗺️ Flat</button></div>
<div class="tg"><button id="bl">☀️ Light</button><button id="bd" class="on">🌙 Dark</button></div></div>
<div id="main"><div id="mapwrap"><canvas id="cv"></canvas><div id="tip"></div><div id="status">Loading…</div>
<div id="zc"><button id="zi" title="Zoom in">+</button><button id="zo" title="Zoom out">−</button><button id="zr" title="Reset view">⟲</button></div></div>
<div id="info"></div></div>
<script>
const D=__DATA__;
const SAT_URL='https://cdn.jsdelivr.net/npm/satellite.js@4.1.3/dist/satellite.min.js';
const WORLD_URL='https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json';
const RAD=Math.PI/180,DEG=180/Math.PI;
const $=id=>document.getElementById(id);
window.addEventListener('error',e=>{const s=$('status');if(s)s.textContent='Error: '+e.message;});

/* ---------- satellites ---------- */
const S=[];
D.feat.forEach(f=>S.push({n:f.name,id:f.id,top:true,big:!!f.big,l1:f.l1,l2:f.l2,desc:f.desc,kind:f.kind}));
const NF=S.length;
D.auto.split('\n').forEach(line=>{if(!line)return;const p=line.split('\t');S.push({n:p[0],id:+p[1],top:false,big:false,l1:p[2],l2:p[3],desc:'',kind:'none'});});
const N=S.length;
const NAMES=S.map(s=>s.n.toLowerCase());
const LAT=new Float32Array(N),LON=new Float32Array(N),ALT=new Float32Array(N);
const SP=new Float32Array(N),CP=new Float32Array(N),SL=new Float32Array(N),CL=new Float32Array(N);
const SX=new Float32Array(N),SY=new Float32Array(N),OK=new Uint8Array(N),VIS=new Uint8Array(N);

/* ---------- state ---------- */
const P={dark:{land:'#2b3a42',ocean:'#0e1a22',bord:'#46565e',text:'#e8eef2',dot:'#ff4d4d',auto:'rgba(255,140,140,.85)',sel:'#ffd166',out:'#ffffff'},
light:{land:'#e6e2d6',ocean:'#cfe3f2',bord:'#b5b0a2',text:'#1b2733',dot:'#d62828',auto:'rgba(200,40,40,.75)',sel:'#f77f00',out:'#1b2733'}};
let theme='dark',mode='globe',rot=[0,-20],Z=1,ty=0,kG=1,kF=1,W=600,H=500,dpr=1;
let dirty=true,anim=null,sel=-1,hov=-1,LAND=null,BORD=null,TRK=null,CUR=null,lastCompute=0;
const cv=$('cv'),ctx=cv.getContext('2d'),wrap=$('mapwrap'),q=$('q'),res=$('res'),clr=$('clr'),info=$('info'),tip=$('tip');
const PG=d3.geoOrthographic().clipAngle(90).precision(0.6),PF=d3.geoNaturalEarth1().precision(0.6);
const GR=d3.geoGraticule10();

function resize(){const r=wrap.getBoundingClientRect();W=Math.max(200,r.width);H=Math.max(200,r.height);
 dpr=Math.min(2,window.devicePixelRatio||1);cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);
 kG=Math.min(W,H)/2*0.96;const t=d3.geoNaturalEarth1();t.fitSize([W,H],{type:'Sphere'});kF=t.scale();dirty=true;}
new ResizeObserver(resize).observe(wrap);resize();

function proj(){const g=mode==='globe',p=g?PG:PF;
 p.scale((g?kG:kF)*Z).translate([W/2,H/2+(g?0:ty)]).rotate(g?[rot[0],rot[1]]:[rot[0],0]);return p;}

/* ---------- positions (worker, with main-thread fallback) ---------- */
function core(R,t){const d=new Date(t),gm=satellite.gstime(d),n=R.length,o=new Float32Array(3*n);
 for(let i=0;i<n;i++){const pv=satellite.propagate(R[i],d),p=pv.position;
  if(!p||p===true){o[3*i]=NaN;continue;}
  const g=satellite.eciToGeodetic(p,gm);
  o[3*i]=satellite.degreesLat(g.latitude);o[3*i+1]=satellite.degreesLong(g.longitude);o[3*i+2]=g.height;}
 return o;}
const L=new Array(2*N);for(let i=0;i<N;i++){L[2*i]=S[i].l1;L[2*i+1]=S[i].l2;}
let worker=null,busy=false,MR=null,ticker=null;
function setTrig(i){const a=LAT[i]*RAD,b=LON[i]*RAD;SP[i]=Math.sin(a);CP[i]=Math.cos(a);SL[i]=Math.sin(b);CL[i]=Math.cos(b);}
function apply(o){for(let i=0;i<N;i++){const la=o[3*i];if(la!==la){OK[i]=0;continue;}
  OK[i]=1;LAT[i]=la;LON[i]=o[3*i+1];ALT[i]=o[3*i+2];setTrig(i);}
 lastCompute=Date.now();$('status').textContent=N.toLocaleString()+' satellites · drag to rotate, scroll to zoom';
 refreshSel();dirty=true;}
function tick(){if(busy)return;busy=true;const t=Date.now();
 if(worker){worker.postMessage({t:t});return;}
 setTimeout(()=>{if(!MR){MR=new Array(N);for(let i=0;i<N;i++)MR[i]=satellite.twoline2satrec(L[2*i],L[2*i+1]);}
  apply(core(MR,t));busy=false;},0);}
function startTicker(){if(ticker)return;tick();ticker=setInterval(tick,2000);}
function useMain(){worker=null;busy=false;if(!ticker)startTicker();else tick();}
function startWorker(){
 try{const src='importScripts('+JSON.stringify(SAT_URL)+');const core='+core.toString()+';let R=null;onmessage=function(e){const m=e.data;if(m.l){R=new Array(m.l.length/2);for(let i=0;i<R.length;i++)R[i]=satellite.twoline2satrec(m.l[2*i],m.l[2*i+1]);postMessage({ready:1});return;}const o=core(R,m.t);postMessage({o:o},[o.buffer]);};';
  const w=new Worker(URL.createObjectURL(new Blob([src],{type:'text/javascript'})));
  w.onmessage=e=>{if(e.data.ready){worker=w;startTicker();return;}busy=false;apply(e.data.o);};
  w.onerror=()=>{useMain();};
  w.postMessage({l:L});
  setTimeout(()=>{if(!worker&&!ticker)useMain();},15000);
 }catch(err){useMain();}}
$('status').textContent='Calculating positions…';
startWorker();

/* ---------- single-satellite helpers ---------- */
const RC={};
function satrec(i){return RC[i]||(RC[i]=satellite.twoline2satrec(S[i].l1,S[i].l2));}
function st(i,d){const pv=satellite.propagate(satrec(i),d);if(!pv.position||pv.position===true)return null;
 const g=satellite.eciToGeodetic(pv.position,satellite.gstime(d)),v=pv.velocity;
 return{lat:satellite.degreesLat(g.latitude),lon:satellite.degreesLong(g.longitude),alt:g.height,v:Math.hypot(v.x,v.y,v.z)};}
function track(i){let per=2*Math.PI/satrec(i).no;if(!(per>0))per=100;per=Math.min(Math.max(per,60),1500);
 const n=240,step=per/n*60000,now=Date.now(),base=Math.floor(now/step)*step,c=[];
 const add=t=>{const p=st(i,new Date(t));if(p)c.push([p.lon,p.lat]);};
 for(let k=-Math.round(n/4);k<=n;k++){const t=base+k*step;if(t<now)add(t);}
 add(now);
 for(let k=0;k<=n;k++){const t=base+k*step;if(t>now)add(t);}
 return{type:'LineString',coordinates:c};}
function refreshTrack(){TRK=sel>=0?track(sel):null;dirty=true;}
function refreshSel(){if(sel<0)return;const c=st(sel,new Date());if(!c)return;
 LAT[sel]=c.lat;LON[sel]=c.lon;ALT[sel]=c.alt;OK[sel]=1;setTrig(sel);CUR=c;nums();dirty=true;}
setInterval(refreshSel,300);
setInterval(()=>{if(sel>=0)refreshTrack();},5000);

/* ---------- drawing ---------- */
function project(p){const k=p.scale(),t=p.translate();
 if(mode==='globe'){const lc=-rot[0]*RAD,pc=-rot[1]*RAD,sc=Math.sin(pc),cc=Math.cos(pc),slc=Math.sin(lc),clc=Math.cos(lc);
  for(let i=0;i<N;i++){if(!OK[i]){VIS[i]=0;continue;}
   const cd=CL[i]*clc+SL[i]*slc,sd=SL[i]*clc-CL[i]*slc,z=SP[i]*sc+CP[i]*cc*cd;
   if(z<=0){VIS[i]=0;continue;}
   SX[i]=t[0]+k*CP[i]*sd;SY[i]=t[1]-k*(cc*SP[i]-sc*CP[i]*cd);VIS[i]=1;}
 }else{
  for(let i=0;i<N;i++){if(!OK[i]){VIS[i]=0;continue;}
   const c=p([LON[i],LAT[i]]);if(!c){VIS[i]=0;continue;}SX[i]=c[0];SY[i]=c[1];VIS[i]=1;}
 }}
function draw(){dirty=false;
 const pal=P[theme],p=proj(),pc=d3.geoPath(p,ctx);
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
 ctx.beginPath();pc({type:'Sphere'});ctx.fillStyle=pal.ocean;ctx.fill();
 ctx.globalAlpha=.25;ctx.beginPath();pc(GR);ctx.strokeStyle=pal.bord;ctx.lineWidth=.5;ctx.stroke();ctx.globalAlpha=1;
 if(LAND){ctx.beginPath();pc(LAND);ctx.fillStyle=pal.land;ctx.fill();}
 if(BORD){ctx.beginPath();pc(BORD);ctx.strokeStyle=pal.bord;ctx.lineWidth=.6;ctx.stroke();}
 ctx.beginPath();pc({type:'Sphere'});ctx.strokeStyle=pal.bord;ctx.lineWidth=1;ctx.stroke();
 if(sel>=0&&TRK){ctx.beginPath();pc(TRK);ctx.setLineDash([2,6]);ctx.lineCap='round';ctx.strokeStyle=pal.sel;ctx.lineWidth=2;ctx.stroke();ctx.setLineDash([]);}
 project(p);
 const as=Math.min(3.4,1.7+0.35*Math.log2(Z));
 ctx.fillStyle=pal.auto;
 for(let i=NF;i<N;i++){if(!VIS[i]||i===sel)continue;const x=SX[i],y=SY[i];
  if(x<-4||y<-4||x>W+4||y>H+4)continue;ctx.fillRect(x-as/2,y-as/2,as,as);}
 ctx.lineWidth=1.2;ctx.strokeStyle=pal.out;ctx.fillStyle=pal.dot;
 for(let i=0;i<NF;i++){if(!VIS[i]||i===sel)continue;ctx.beginPath();ctx.arc(SX[i],SY[i],S[i].big?8:5,0,6.2832);ctx.fill();ctx.stroke();}
 if(sel>=0&&VIS[sel]){const x=SX[sel],y=SY[sel];
  ctx.beginPath();ctx.arc(x,y,11,0,6.2832);ctx.fillStyle=pal.sel;ctx.fill();ctx.strokeStyle=pal.out;ctx.lineWidth=1.5;ctx.stroke();
  ctx.font='600 13px system-ui,sans-serif';ctx.textAlign='center';ctx.lineJoin='round';ctx.lineWidth=3;
  ctx.strokeStyle=pal.ocean;ctx.strokeText(S[sel].n,x,y-17);ctx.fillStyle=pal.text;ctx.fillText(S[sel].n,x,y-17);}
}
function step(now){if(!anim)return;const a=anim,u=Math.min(1,(now-a.t0)/a.ms),e=u<.5?2*u*u:1-Math.pow(-2*u+2,2)/2;
 rot[0]=a.s.r0+a.dr*e;rot[1]=a.s.r1+(a.t.r1-a.s.r1)*e;
 Z=Math.exp(Math.log(a.s.Z)+(Math.log(a.t.Z)-Math.log(a.s.Z))*e);ty=a.s.ty+(a.t.ty-a.s.ty)*e;
 if(u>=1){anim=null;rot[0]=((rot[0]+540)%360)-180;}dirty=true;}
function frame(now){step(now);if(dirty)draw();requestAnimationFrame(frame);}
requestAnimationFrame(frame);
function animTo(t,ms){let dr=t.r0-rot[0];dr=((dr+540)%360)-180;
 anim={s:{r0:rot[0],r1:rot[1],Z:Z,ty:ty},t:t,dr:dr,t0:performance.now(),ms:ms};dirty=true;}
function goTo(lat,lon,z){let t={r0:-lon,r1:mode==='globe'?-lat:0,Z:z,ty:0};
 if(mode==='flat'){const tp=d3.geoNaturalEarth1().scale(kF*z).translate([W/2,H/2]).rotate([-lon,0]);const c=tp([lon,lat]);if(c)t.ty=H/2-c[1];}
 animTo(t,900);}

/* ---------- interaction ---------- */
function hit(x,y){let b=-1,bd=1e9;
 for(let i=0;i<N;i++){if(!VIS[i])continue;const dx=SX[i]-x,dy=SY[i]-y,d=dx*dx+dy*dy,f=i<NF,lim=f?(S[i].big?144:100):49;
  if(d<lim){const w=f?d*.4:d;if(w<bd){bd=w;b=i;}}}
 return b;}
function hover(x,y){const i=hit(x,y);cv.style.cursor=i>=0?'pointer':'grab';
 if(i<0){hov=-1;tip.style.display='none';return;}
 hov=i;tip.textContent=S[i].n+' · '+Math.round(ALT[i]).toLocaleString()+' km';
 tip.style.left=Math.min(x+12,W-150)+'px';tip.style.top=(y+14)+'px';tip.style.display='block';}
function dragBy(dx,dy){anim=null;
 if(mode==='globe'){const k=kG*Z;rot[0]+=dx*DEG/k;rot[1]=Math.max(-90,Math.min(90,rot[1]-dy*DEG/k));}
 else{const k=kF*Z*0.87;rot[0]+=dx*DEG/k;ty=Math.max(-H*Z/2,Math.min(H*Z/2,ty+dy));}
 rot[0]=((rot[0]+540)%360)-180;dirty=true;}
function zoomBy(f){anim=null;Z=Math.min(80,Math.max(1,Z*f));if(Z===1)ty=0;dirty=true;}
const ptrs=new Map();let pinch=0,moved=0,downAt=null;
const d2=()=>{const a=[...ptrs.values()];return Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]);};
cv.addEventListener('pointerdown',e=>{try{cv.setPointerCapture(e.pointerId);}catch(x){}
 ptrs.set(e.pointerId,[e.offsetX,e.offsetY]);moved=0;downAt=[e.offsetX,e.offsetY];anim=null;if(ptrs.size===2)pinch=d2();});
cv.addEventListener('pointermove',e=>{
 if(!ptrs.has(e.pointerId)){if(e.pointerType==='mouse')hover(e.offsetX,e.offsetY);return;}
 const o=ptrs.get(e.pointerId),x=e.offsetX,y=e.offsetY;
 if(ptrs.size===1){const dx=x-o[0],dy=y-o[1];moved+=Math.abs(dx)+Math.abs(dy);if(moved>4)tip.style.display='none';dragBy(dx,dy);}
 ptrs.set(e.pointerId,[x,y]);
 if(ptrs.size===2){const d=d2();if(pinch)zoomBy(d/pinch);pinch=d;moved+=10;}});
const up=e=>{if(ptrs.size===1&&moved<5&&downAt){const i=hit(downAt[0],downAt[1]);if(i>=0)pick(i);}ptrs.delete(e.pointerId);pinch=0;};
cv.addEventListener('pointerup',up);cv.addEventListener('pointercancel',e=>{ptrs.delete(e.pointerId);pinch=0;});
cv.addEventListener('pointerleave',()=>{tip.style.display='none';});
cv.addEventListener('wheel',e=>{e.preventDefault();zoomBy(Math.exp(-e.deltaY*0.0015));},{passive:false});
cv.addEventListener('dblclick',()=>resetView());
function resetView(){animTo({r0:rot[0],r1:mode==='globe'?-20:0,Z:1,ty:0},600);}
$('zi').onclick=()=>zoomBy(1.6);$('zo').onclick=()=>zoomBy(1/1.6);$('zr').onclick=resetView;

/* ---------- info card ---------- */
function nums(){const c=CUR;if(!c||!$('vA'))return;
 $('vA').textContent=c.alt.toFixed(1)+' km';$('vS').textContent=c.v.toFixed(2)+' km/s';
 $('vH').textContent=Math.round(c.v*3600).toLocaleString()+' km/h';
 $('vO').textContent=c.lon.toFixed(4)+'°';$('vL').textContent=c.lat.toFixed(4)+'°';}
function show(){const s=S[sel];info.style.display='block';
 info.innerHTML='<h2 id="nm"></h2><div class="g"><div><small>NORAD ID</small><b>'+s.id+'</b></div><div><small>Altitude</small><b id="vA"></b></div><div><small>Speed</small><b id="vS"></b></div><div><small>Speed</small><b id="vH"></b></div><div><small>Longitude</small><b id="vO"></b></div><div><small>Latitude</small><b id="vL"></b></div></div><div id="desc"></div><div id="kind"></div>';
 $('nm').textContent=s.n;$('desc').textContent=s.desc;
 $('kind').textContent=s.kind==='ai'?'✨ AI-generated description. It may contain errors.':s.kind==='wiki'?'Summary from Wikipedia'+(s.wt?' ('+s.wt+')':'')+'. It may not match this exact satellite.':s.kind==='loading'?'Looking for a Wikipedia summary…':'';
 nums();if(s.kind==='none')wiki(s,sel);}
function yr(s){const y=parseInt(s.l1.substring(9,17).trim().substring(0,2));return isNaN(y)?null:(y>=57?1900+y:2000+y);}
function wiki(s,i){const y=yr(s);s.desc=s.n+' (NORAD ID '+s.id+')'+(y?', launched in '+y+'.':'.');s.kind='loading';
 $('desc').textContent=s.desc;$('kind').textContent='Looking for a Wikipedia summary…';
 fetch('https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrlimit=1&prop=extracts&exintro=1&explaintext=1&exsentences=3&redirects=1&format=json&origin=*&gsrsearch='+encodeURIComponent(s.n.replace(/\(.*?\)/g,'')+' satellite'))
 .then(r=>r.json()).then(j=>{const pg=Object.values((j.query||{}).pages||{})[0];
  if(pg&&pg.extract){s.desc=pg.extract;s.kind='wiki';s.wt=pg.title;}else s.kind='basic';
  if(sel===i)show();}).catch(()=>{s.kind='basic';if(sel===i)show();});}
function pick(i){sel=i;q.value=i>=0?S[i].n:'';clr.style.display=i>=0?'block':'none';res.style.display='none';hov=-1;tip.style.display='none';
 if(i>=0){CUR=null;const c=st(i,new Date());if(c){CUR=c;LAT[i]=c.lat;LON[i]=c.lon;ALT[i]=c.alt;OK[i]=1;setTrig(i);goTo(c.lat,c.lon,mode==='globe'?3.2:3.5);}
  refreshTrack();show();info.scrollIntoView({block:'nearest'});}
 else{info.style.display='none';TRK=null;resetView();}
 dirty=true;}

/* ---------- search ---------- */
function addRow(i){const d=document.createElement('div');d.textContent=S[i].n;d.onmousedown=e=>{e.preventDefault();pick(i);};res.appendChild(d);}
function list(){const t=q.value.trim().toLowerCase();res.innerHTML='';let n=0;
 if(!t){for(let i=0;i<NF&&n<30;i++){addRow(i);n++;}}
 else{const num=/^\d+$/.test(t);for(let i=0;i<N&&n<60;i++){if(NAMES[i].startsWith(t)||(num&&String(S[i].id).startsWith(t))){addRow(i);n++;}}}
 res.style.display=n?'block':'none';}
q.oninput=list;q.onfocus=list;q.onblur=()=>setTimeout(()=>{res.style.display='none';},150);
q.onkeydown=e=>{if(e.key==='Enter'){const t=q.value.trim().toLowerCase();if(!t)return;let i=NAMES.findIndex(n=>n.startsWith(t));if(i<0&&/^\d+$/.test(t))i=S.findIndex(s=>String(s.id)===t);if(i>=0)pick(i);}};
clr.onclick=()=>pick(-1);

/* ---------- toggles ---------- */
function setTheme(m){theme=m;document.body.className=m==='light'?'light':'';$('bl').className=m==='light'?'on':'';$('bd').className=m==='dark'?'on':'';dirty=true;}
function setProj(m){mode=m;$('bg').className=m==='globe'?'on':'';$('bf').className=m==='flat'?'on':'';
 anim=null;Z=1;ty=0;rot[1]=m==='globe'?-20:0;dirty=true;
 if(sel>=0&&OK[sel])goTo(LAT[sel],LON[sel],m==='globe'?3.2:3.5);}
$('bl').onclick=()=>setTheme('light');$('bd').onclick=()=>setTheme('dark');
$('bg').onclick=()=>setProj('globe');$('bf').onclick=()=>setProj('flat');

/* ---------- map outlines ---------- */
fetch(WORLD_URL).then(r=>r.json()).then(w=>{LAND=topojson.feature(w,w.objects.land);
 BORD=topojson.mesh(w,w.objects.countries,(a,b)=>a!==b);dirty=true;}).catch(()=>{});
</script></body></html>
'''

DEBUG = "debug" in st.query_params  # open your link with ?debug=1 to see data status

st.title("🛰️ SkySync")

with st.spinner("Loading orbital data (up to ~20 seconds)..."):
    satellites, sources, errors, tles, meta = load_all()

if not meta:
    st.error(
        "No satellites could be loaded. The server may be blocked from "
        "downloading orbital data. Add a tles.txt file to the repository."
    )
    st.stop()

if DEBUG:
    st.caption(f"version 14 · fast map · {len(meta):,} satellites ({len(satellites)} featured)")
    if errors:
        st.warning("Could not load: " + ", ".join(f"{k} ({v})" for k, v in errors.items()))
    st.dataframe(
        pd.DataFrame(
            [{"Satellite": k, "Name in data": tles[k][0], "Data": sources[k]} for k in satellites]
        ),
        hide_index=True,
    )

feat, auto_lines = [], []
with st.spinner("Preparing descriptions..."):
    for label, m in meta.items():
        name, l1, l2 = tles[label]
        if m["top"]:  # your featured satellites: written text, else AI / Wikipedia
            text, kind, note = get_description(label, m["id"], launch_year(satellites[label]))
            feat.append(
                {"name": label, "id": m["id"], "big": label in FAMOUS,
                 "l1": l1, "l2": l2, "desc": text, "kind": kind}
            )
        else:  # everything added automatically: the browser looks up a Wikipedia summary
            auto_lines.append(f"{label}\t{m['id']}\t{l1}\t{l2}")

payload = {"feat": feat, "auto": "\n".join(auto_lines)}
data_json = json.dumps(payload).replace("</", "<\\/")
components.html(WIDGET_HTML.replace("__DATA__", data_json), height=800, scrolling=True)
