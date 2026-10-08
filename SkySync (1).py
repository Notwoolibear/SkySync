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
ZOOM_SCALE = 5  # (unused now, the browser map sets its own zoom)

# Satellites are added AUTOMATICALLY from these NORAD/CelesTrak groups, on top of
# the featured list (TARGETS) below. Group names: celestrak.org/NORAD/elements
# Avoid huge groups such as "starlink" or "active": thousands of dots make the map slow.
# Keep this list the same as GROUPS in update_tles.py.
GROUPS = ["stations", "visual", "weather", "noaa", "goes", "resource", "science"]
MAX_AUTO = 400  # most automatically added satellites to show

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
    for label, catnr, name, l1, l2 in auto[:MAX_AUTO]:
        if label in sats:
            label = f"{label} ({catnr})"
        try:
            sats[label] = EarthSatellite(l1, l2, name or label, ts)
        except Exception:
            continue
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
#top{display:flex;gap:8px;padding:2px 2px 8px}
#sw{position:relative;flex:1}
#q{width:100%;padding:10px 34px 10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);font-size:15px}
#clr{position:absolute;right:8px;top:6px;border:0;background:none;color:var(--fg);font-size:20px;cursor:pointer;display:none}
#res{position:absolute;z-index:10;left:0;right:0;top:44px;background:var(--card);border:1px solid var(--line);border-radius:8px;max-height:240px;overflow:auto;display:none}
#res div{padding:8px 12px;cursor:pointer}#res div:hover{background:var(--line)}
.tg button{padding:10px 12px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;font-size:14px}
.tg button:first-child{border-radius:8px 0 0 8px}.tg button:last-child{border-radius:0 8px 8px 0;border-left:0}
.tg button.on{background:var(--acc);color:#111;border-color:var(--acc)}
#main{display:flex;gap:10px;flex-wrap:wrap}
#map{flex:3 1 480px;height:540px;min-width:0}
#info{flex:2 1 280px;display:none;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;align-self:flex-start}
#info h2{margin:0 0 10px;font-size:20px}
.g{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;margin-bottom:12px}
.g small{display:block;opacity:.65;font-size:12px}.g b{font-size:17px}
#desc{line-height:1.45;font-size:14px}#kind{opacity:.6;font-size:12px;margin-top:8px}
</style></head><body>
<div id="top"><div id="sw"><input id="q" placeholder="Search satellites…" autocomplete="off"><button id="clr" title="Clear">×</button><div id="res"></div></div>
<div class="tg"><button id="bl">☀️ Light</button><button id="bd" class="on">🌙 Dark</button></div></div>
<div id="main"><div id="map"></div><div id="info"></div></div>
<script src="https://cdn.jsdelivr.net/npm/satellite.js@4.1.3/dist/satellite.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
const SATS=__DATA__;
const P={dark:{land:'#2b3a42',ocean:'#0e1a22',bord:'#46565e',text:'#e8eef2',dot:'#ff4d4d',sel:'#ffd166',out:'#fff'},
light:{land:'#e6e2d6',ocean:'#cfe3f2',bord:'#b5b0a2',text:'#1b2733',dot:'#d62828',sel:'#f77f00',out:'#1b2733'}};
let mode='dark',sel=-1,cur=[];
const R=SATS.map(s=>satellite.twoline2satrec(s.l1,s.l2));
const $=id=>document.getElementById(id),q=$('q'),res=$('res'),clr=$('clr'),info=$('info'),map=$('map');
function st(i,d){const pv=satellite.propagate(R[i],d);if(!pv.position)return null;
 const g=satellite.eciToGeodetic(pv.position,satellite.gstime(d)),v=pv.velocity;
 return{lat:satellite.degreesLat(g.latitude),lon:satellite.degreesLong(g.longitude),alt:g.height,v:Math.hypot(v.x,v.y,v.z)};}
function track(i){let per=2*Math.PI/R[i].no;if(!(per>0))per=100;per=Math.min(Math.max(per,60),1500);
 const n=240,step=per/n*60000,now=Date.now(),base=Math.floor(now/step)*step,la=[],lo=[];
 const add=t=>{const c=st(i,new Date(t));if(c){la.push(c.lat);lo.push(c.lon);}};
 for(let k=-Math.round(n/4);k<=n;k++){const t=base+k*step;if(t<now)add(t);}
 add(now);
 for(let k=0;k<=n;k++){const t=base+k*step;if(t>now)add(t);}
 return{la,lo};}
const marks=()=>{const p=P[mode];return{'marker.color':[SATS.map((_,i)=>i===sel?p.sel:p.dot)],'marker.size':[SATS.map((x,i)=>i===sel?16:(x.top?11:7))],'text':[SATS.map((x,i)=>(x.top||i===sel)?x.name:'')]};};
const geo=()=>{const p=P[mode];return{projection:{type:'natural earth'},showland:true,landcolor:p.land,showocean:true,oceancolor:p.ocean,showcountries:true,countrycolor:p.bord,showframe:false,bgcolor:'rgba(0,0,0,0)'};};
cur=SATS.map((_,i)=>st(i,new Date()));
Plotly.newPlot(map,[
 {type:'scattergeo',mode:'lines',lat:[],lon:[],line:{width:2,dash:'dot',color:P.dark.sel},hoverinfo:'skip'},
 {type:'scattergeo',mode:'markers+text',lat:cur.map(c=>c&&c.lat),lon:cur.map(c=>c&&c.lon),text:SATS.map(s=>s.top?s.name:''),textposition:'top center',
  textfont:{size:12,color:P.dark.text},marker:{size:SATS.map(s=>s.top?11:7),color:P.dark.dot,line:{width:1,color:P.dark.out}},
  customdata:cur.map((c,i)=>[SATS[i].name,c?Math.round(c.alt):null]),hovertemplate:'%{customdata[0]}<br>Alt: %{customdata[1]} km<extra></extra>'}],
 {geo:geo(),margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'rgba(0,0,0,0)',showlegend:false},
 {scrollZoom:true,displaylogo:false,responsive:true});
map.on('plotly_click',e=>{const p=e.points&&e.points[0];if(p&&p.curveNumber===1)pick(p.pointIndex);});
function nums(){const c=cur[sel];if(!c||!$('vA'))return;
 $('vA').textContent=c.alt.toFixed(1)+' km';$('vS').textContent=c.v.toFixed(2)+' km/s';
 $('vH').textContent=Math.round(c.v*3600).toLocaleString()+' km/h';
 $('vO').textContent=c.lon.toFixed(4)+'°';$('vL').textContent=c.lat.toFixed(4)+'°';}
function show(){const s=SATS[sel];info.style.display='block';
 info.innerHTML='<h2 id="nm"></h2><div class="g"><div><small>NORAD ID</small><b>'+s.id+'</b></div><div><small>Altitude</small><b id="vA"></b></div><div><small>Speed</small><b id="vS"></b></div><div><small>Speed</small><b id="vH"></b></div><div><small>Longitude</small><b id="vO"></b></div><div><small>Latitude</small><b id="vL"></b></div></div><div id="desc"></div><div id="kind"></div>';
 $('nm').textContent=s.name;$('desc').textContent=s.desc;
 $('kind').textContent=s.kind==='ai'?'✨ AI-generated description. It may contain errors.':s.kind==='wiki'?'Summary from Wikipedia'+(s.wt?' ('+s.wt+')':'')+'. It may not match this exact satellite.':'';
 nums();if(s.kind==='none')wiki(s,sel);}
function yr(s){const y=parseInt(s.l1.substring(9,17).trim().substring(0,2));return isNaN(y)?null:(y>=57?1900+y:2000+y);}
function wiki(s,i){const y=yr(s);s.desc=s.name+' (NORAD ID '+s.id+')'+(y?', launched in '+y+'.':'.');
 $('desc').textContent=s.desc;$('kind').textContent='Looking for a Wikipedia summary…';
 fetch('https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrlimit=1&prop=extracts&exintro=1&explaintext=1&exsentences=3&redirects=1&format=json&origin=*&gsrsearch='+encodeURIComponent(s.name.replace(/\(.*?\)/g,'')+' satellite'))
 .then(r=>r.json()).then(j=>{const pg=Object.values((j.query||{}).pages||{})[0];
  if(pg&&pg.extract){s.desc=pg.extract;s.kind='wiki';s.wt=pg.title;}else s.kind='basic';
  if(sel===i)show();}).catch(()=>{s.kind='basic';if(sel===i)show();});}
function drawTrack(){const t=track(sel);Plotly.restyle(map,{lat:[t.la],lon:[t.lo]},[0]);}
function pick(i){sel=i;q.value=i>=0?SATS[i].name:'';clr.style.display=i>=0?'block':'none';res.style.display='none';
 Plotly.restyle(map,marks(),[1]);
 const rot={'geo.projection.rotation.lon':0,'geo.projection.rotation.lat':0,'geo.projection.rotation.roll':0};
 if(i>=0&&cur[i]){drawTrack();Plotly.relayout(map,Object.assign({'geo.center.lat':cur[i].lat,'geo.center.lon':cur[i].lon,'geo.projection.scale':5},rot));show();}
 else{Plotly.restyle(map,{lat:[[]],lon:[[]]},[0]);Plotly.relayout(map,Object.assign({'geo.center.lat':0,'geo.center.lon':0,'geo.projection.scale':1},rot));info.style.display='none';}}
function list(){const t=q.value.trim().toLowerCase();res.innerHTML='';let n=0;
 SATS.forEach((s,i)=>{if(n>=60||!s.name.toLowerCase().startsWith(t))return;n++;const d=document.createElement('div');d.textContent=s.name;
  d.onmousedown=e=>{e.preventDefault();pick(i);};res.appendChild(d);});
 res.style.display=n?'block':'none';}
q.oninput=list;q.onfocus=list;q.onblur=()=>setTimeout(()=>{res.style.display='none';},150);
q.onkeydown=e=>{if(e.key==='Enter'){const t=q.value.trim().toLowerCase(),i=SATS.findIndex(s=>s.name.toLowerCase().startsWith(t));if(i>=0)pick(i);}};
clr.onclick=()=>pick(-1);
function setMode(m){mode=m;document.body.className=m==='light'?'light':'';$('bl').className=m==='light'?'on':'';$('bd').className=m==='dark'?'on':'';
 const p=P[m];Plotly.relayout(map,{'geo.landcolor':p.land,'geo.oceancolor':p.ocean,'geo.countrycolor':p.bord});
 Plotly.restyle(map,Object.assign({'textfont.color':[p.text],'marker.line.color':[p.out]},marks()),[1]);
 Plotly.restyle(map,{'line.color':[p.sel]},[0]);}
$('bl').onclick=()=>setMode('light');$('bd').onclick=()=>setMode('dark');
setInterval(()=>{const d=new Date();cur=SATS.map((_,i)=>st(i,d));
 Plotly.restyle(map,{lat:[cur.map(c=>c&&c.lat)],lon:[cur.map(c=>c&&c.lon)],customdata:[cur.map((c,i)=>[SATS[i].name,c?Math.round(c.alt):null])]},[1]);nums();},1000);
setInterval(()=>{if(sel>=0)drawTrack();},5000);
</script></body></html>
'''

DEBUG = "debug" in st.query_params  # open your link with ?debug=1 to see data status

st.title("🛰️ SkySync")

with st.spinner("Loading orbital data (up to ~20 seconds)..."):
    satellites, sources, errors, tles, meta = load_all()

if not satellites:
    st.error(
        "No satellites could be loaded. The server may be blocked from "
        "downloading orbital data. Add a tles.txt file to the repository."
    )
    st.stop()

if DEBUG:
    st.caption(f"version 11 · {len(satellites)} satellites")
    if errors:
        st.warning("Could not load: " + ", ".join(f"{k} ({v})" for k, v in errors.items()))
    st.dataframe(
        pd.DataFrame(
            [{"Satellite": k, "Name in data": tles[k][0], "Data": sources[k]} for k in satellites]
        ),
        hide_index=True,
    )

payload = []
with st.spinner("Preparing descriptions..."):
    for label, sat in satellites.items():
        m = meta[label]
        if m["top"]:  # your featured satellites: written text, else AI / Wikipedia
            text, kind, note = get_description(label, m["id"], launch_year(sat))
        else:  # automatically added ones: the browser looks up a Wikipedia summary
            text, kind = "", "none"
        payload.append(
            {"name": label, "id": m["id"], "top": m["top"], "l1": tles[label][1],
             "l2": tles[label][2], "desc": text, "kind": kind}
        )

data_json = json.dumps(payload).replace("</", "<\\/")
components.html(WIDGET_HTML.replace("__DATA__", data_json), height=780, scrolling=True)
