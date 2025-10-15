from __future__ import annotations
import streamlit as st
import json, csv, io, zipfile, sys, math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Tuple
from pathlib import Path
import pandas as pd
import altair as alt

# ------------------------------------------------------------------------------
# Import des bindings protobuf locaux
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gtfs_realtime_pb2 as gtfs_local
except Exception:
    gtfs_local = None

# ------------------------------------------------------------------------------
# Modèles
# ------------------------------------------------------------------------------
@dataclass
class StopSelector:
    stop_sequence: Optional[int] = None
    stop_id: Optional[str] = None

@dataclass
class ReplacementStop:
    stop_id: str
    travel_time_to_stop: int = 0

@dataclass
class Modification:
    start_stop_selector: Optional[StopSelector] = None
    end_stop_selector: Optional[StopSelector] = None
    replacement_stops: List[ReplacementStop] = field(default_factory=list)

@dataclass
class SelectedTrips:
    trip_ids: List[str] = field(default_factory=list)
    shape_id: Optional[str] = None

@dataclass
class TripModEntity:
    entity_id: str
    selected_trips: List[SelectedTrips]
    modifications: List[Modification] = field(default_factory=list)

@dataclass
class RtShapesAndStops:
    shapes: Dict[str, List[Tuple[float, float]]]  # shape_id -> [(lat, lon)]
    rt_stops: Dict[str, Tuple[float, float]]      # stop_id -> (lat, lon)

# ------------------------------------------------------------------------------
# Décodage polyline
# ------------------------------------------------------------------------------
def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    coords = []
    index, lat, lon = 0, 0, 0
    while index < len(encoded):
        result, shift = 0, 0
        while True:
            b = ord(encoded[index]) - 63; index += 1
            result |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        result, shift = 0, 0
        while True:
            b = ord(encoded[index]) - 63; index += 1
            result |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlon = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += dlon
        coords.append((lat / 1e5, lon / 1e5))
    return coords

# ------------------------------------------------------------------------------
# Parsing JSON TripModifications
# ------------------------------------------------------------------------------
def parse_tripmods_json(feed: Dict[str, Any]) -> Tuple[List[TripModEntity], RtShapesAndStops]:
    ents: List[TripModEntity] = []
    shapes, rt_stops = {}, {}
    for e in feed.get("entity", []):
        if e.get("trip_modifications"):
            tm = e["trip_modifications"]
            selected = []
            for s in tm.get("selected_trips", []):
                trips = s.get("trip_ids", [])
                if isinstance(trips, str): trips = [trips]
                selected.append(SelectedTrips(trip_ids=trips, shape_id=s.get("shape_id")))
            mods = []
            for m in tm.get("modifications", []):
                start_sel = StopSelector(**(m.get("start_stop_selector") or {}))
                end_sel = StopSelector(**(m.get("end_stop_selector") or {}))
                repl = [ReplacementStop(**rs) for rs in m.get("replacement_stops", [])]
                mods.append(Modification(start_stop_selector=start_sel, end_stop_selector=end_sel, replacement_stops=repl))
            ents.append(TripModEntity(entity_id=str(e.get("id")), selected_trips=selected, modifications=mods))
        if e.get("shape"):
            sid = e["shape"].get("shape_id")
            enc = e["shape"].get("encoded_polyline")
            if sid and enc:
                shapes[sid] = decode_polyline(enc)
        if e.get("stop"):
            stp = e["stop"]
            sid = stp.get("stop_id")
            lat, lon = stp.get("stop_lat"), stp.get("stop_lon")
            if sid and lat and lon:
                rt_stops[sid] = (float(lat), float(lon))
    return ents, RtShapesAndStops(shapes=shapes, rt_stops=rt_stops)

def load_tripmods(file_bytes: bytes) -> Tuple[List[TripModEntity], RtShapesAndStops]:
    feed = json.loads(file_bytes.decode("utf-8"))
    return parse_tripmods_json(feed)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _nearest_index(poly: List[Tuple[float, float]], lat: float, lon: float) -> Optional[int]:
    if not poly: return None
    best, idx = float("inf"), None
    for i, (la, lo) in enumerate(poly):
        d = (la - lat)**2 + (lo - lon)**2
        if d < best: best, idx = d, i
    return idx

def build_chart(ent: TripModEntity, rt: RtShapesAndStops) -> Optional[alt.Chart]:
    # shape_id depuis selected_trips
    shape_id = next((s.shape_id for s in ent.selected_trips if s.shape_id), None)
    if not shape_id or shape_id not in rt.shapes: return None
    poly = rt.shapes[shape_id]
    df_route = pd.DataFrame([{"lat": la, "lon": lo} for la, lo in poly])
    layers = [alt.Chart(df_route).mark_line(color="#9aa0a6", strokeWidth=2).encode(x="lon:Q", y="lat:Q")]
    if ent.modifications:
        m = ent.modifications[0]
        start_c = rt.rt_stops.get(m.start_stop_selector.stop_id)
        end_c = rt.rt_stops.get(m.end_stop_selector.stop_id)
        if start_c and end_c:
            s_idx = _nearest_index(poly, start_c[0], start_c[1])
            e_idx = _nearest_index(poly, end_c[0], end_c[1])
            if s_idx is not None and e_idx is not None:
                if e_idx < s_idx: s_idx, e_idx = e_idx, s_idx
                seg = poly[s_idx:e_idx+1]
                df_seg = pd.DataFrame([{"lat": la, "lon": lo} for la, lo in seg])
                layers.append(alt.Chart(df_seg).mark_line(color="#d93025", strokeWidth=3).encode(x="lon:Q", y="lat:Q"))
    return alt.layer(*layers).properties(width="container", height=400).interactive()

# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------
st.set_page_config(page_title="TripModifications Viewer", layout="wide")
st.title("Visualisation des détours (encoded_polyline)")

with st.sidebar:
    st.header("Données")
    tripmods_file = st.file_uploader("TripModifications (.json)", type=["json"])
    run_btn = st.button("Analyser", type="primary")

if run_btn:
    if not tripmods_file:
        st.error("Veuillez charger un fichier TripModifications JSON.")
        st.stop()
    ents, rt = load_tripmods(tripmods_file.getvalue())
    st.success(f"{len(ents)} entités chargées | {len(rt.shapes)} shapes disponibles")
    for ent in ents[:50]:
        with st.expander(f"Entité {ent.entity_id}"):
            chart = build_chart(ent, rt)
            if chart:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Pas de shape disponible pour cette entité.")
else:
    st.info("Chargez un fichier TripModifications JSON et cliquez sur Analyser.")
