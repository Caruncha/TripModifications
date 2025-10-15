# streamlit_app.py
from __future__ import annotations
import streamlit as st
import json, csv, io, zipfile, sys, math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Tuple
from pathlib import Path

import pandas as pd
import altair as alt

# ------------------------------------------------------------------------------
# 0) Import des bindings protobuf locaux (gtfs_realtime_pb2.py à la racine)
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gtfs_realtime_pb2 as gtfs_local  # votre fichier local
except Exception:
    gtfs_local = None  # fallback vers les bindings pip si possible


# ------------------------------------------------------------------------------
# 1) Modèle en mémoire normalisé
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
    propagated_modification_delay: Optional[int] = None

@dataclass
class SelectedTrips:
    trip_ids: List[str] = field(default_factory=list)
    shape_id: Optional[str] = None

@dataclass
class TripModEntity:
    entity_id: str
    selected_trips: List[SelectedTrips]
    service_dates: List[str] = field(default_factory=list)
    start_times: List[str] = field(default_factory=list)
    modifications: List[Modification] = field(default_factory=list)


# ------------------------------------------------------------------------------
# 2) Utilitaires parsing + normalisation
# ------------------------------------------------------------------------------
def _detect_tripmods_format_bytes(b: bytes) -> str:
    head = (b[:2] or b'').lstrip()
    return 'json' if head.startswith(b'{') or head.startswith(b'[') else 'pb'

def _coerce_selector(obj: Dict[str, Any]) -> StopSelector:
    if not isinstance(obj, dict):
        return StopSelector()
    seq = obj.get('stop_sequence')
    sid = obj.get('stop_id') or None
    try:
        seq = int(seq) if seq is not None and f"{seq}".strip() != '' else None
    except Exception:
        seq = None
    return StopSelector(stop_sequence=seq, stop_id=sid if (sid or None) else None)

def _coerce_repl_stop(obj: Dict[str, Any]) -> Optional[ReplacementStop]:
    if not isinstance(obj, dict):
        return None
    sid = obj.get('stop_id')
    if not sid:
        return None
    t = obj.get('travel_time_to_stop', 0)
    try:
        t = int(t)
    except Exception:
        t = 0
    return ReplacementStop(stop_id=str(sid), travel_time_to_stop=t)

def _coerce_selected_trips(obj: Dict[str, Any]) -> SelectedTrips:
    trips = obj.get('trip_ids') or []
    if isinstance(trips, str):
        trips = [trips]
    trips = [str(t).strip() for t in trips if str(t).strip()]
    shape_id = obj.get('shape_id')
    shape_id = str(shape_id) if shape_id not in (None, '') else None
    return SelectedTrips(trip_ids=trips, shape_id=shape_id)


# ------------------------------------------------------------------------------
# 3) Parsing TripModifications JSON / PB
# ------------------------------------------------------------------------------
def parse_tripmods_json(feed: Dict[str, Any]) -> List[TripModEntity]:
    entities = feed.get('entity') or []
    out: List[TripModEntity] = []
    for e in entities:
        tm = e.get('trip_modifications')
        if not tm:
            continue
        sel_raw = tm.get('selected_trips') or []
        selected = [_coerce_selected_trips(s) for s in sel_raw]

        # Dates au niveau trip_modifications, sinon récup depuis selected_trips
        service_dates = tm.get('service_dates') or []
        if not service_dates:
            for s in sel_raw:
                dates = s.get('service_dates')
                if dates:
                    service_dates.extend(dates)
        service_dates = [str(d) for d in service_dates]
        start_times = [str(t) for t in tm.get('start_times') or []]

        mods: List[Modification] = []
        for m in tm.get('modifications') or []:
            repl_list = []
            for rs in m.get('replacement_stops') or []:
                r = _coerce_repl_stop(rs)
                if r:
                    repl_list.append(r)
            start_sel = _coerce_selector(m.get('start_stop_selector') or {})
            end_sel = _coerce_selector(m.get('end_stop_selector') or {})
            mods.append(Modification(
                start_stop_selector=start_sel,
                end_stop_selector=end_sel,
                replacement_stops=repl_list,
                propagated_modification_delay=m.get('propagated_modification_delay')
            ))

        out.append(TripModEntity(
            entity_id=str(e.get('id')),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=mods
        ))
    return out

def parse_tripmods_protobuf(data: bytes) -> List[TripModEntity]:
    # Priorité au fichier local
    proto = gtfs_local
    if proto is None:
        try:
            from google.transit import gtfs_realtime_pb2 as proto  # fallback pip
        except Exception as ex:
            raise RuntimeError(
                "Bindings Protobuf introuvables. "
                "Assure-toi d’avoir `gtfs_realtime_pb2.py` à la racine "
                "ou installe `gtfs-realtime-bindings` compatibles TripModifications."
            ) from ex

    feed = proto.FeedMessage()
    feed.ParseFromString(data)

    out: List[TripModEntity] = []
    for ent in feed.entity:
        if not hasattr(ent, 'trip_modifications'):
            continue
        tm = ent.trip_modifications

        selected: List[SelectedTrips] = []
        for s in getattr(tm, 'selected_trips', []):
            trips = list(getattr(s, 'trip_ids', []))
            shape_id = getattr(s, 'shape_id', None) or None
            selected.append(SelectedTrips(trip_ids=trips, shape_id=shape_id))

        service_dates = list(getattr(tm, 'service_dates', []))
        start_times = list(getattr(tm, 'start_times', []))

        mods: List[Modification] = []
        for m in getattr(tm, 'modifications', []):
            repl = [ReplacementStop(stop_id=rs.stop_id, travel_time_to_stop=int(getattr(rs, 'travel_time_to_stop', 0)))
                    for rs in getattr(m, 'replacement_stops', [])]
            start_sel = StopSelector(
                stop_sequence=getattr(m.start_stop_selector, 'stop_sequence', None) if hasattr(m, 'start_stop_selector') else None,
                stop_id=getattr(m.start_stop_selector, 'stop_id', None) if hasattr(m, 'start_stop_selector') else None,
            )
            end_sel = StopSelector(
                stop_sequence=getattr(m.end_stop_selector, 'stop_sequence', None) if hasattr(m, 'end_stop_selector') else None,
                stop_id=getattr(m.end_stop_selector, 'stop_id', None) if hasattr(m, 'end_stop_selector') else None,
            )
            mods.append(Modification(
                start_stop_selector=start_sel,
                end_stop_selector=end_sel,
                replacement_stops=repl,
                propagated_modification_delay=getattr(m, 'propagated_modification_delay', None)
            ))

        out.append(TripModEntity(
            entity_id=str(getattr(ent, 'id', '')),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=mods
        ))
    return out


# ------------------------------------------------------------------------------
# 4) Shapes & stops RT (depuis le feed TripMods) + décodage polyline
# ------------------------------------------------------------------------------
@dataclass
class RtShapesAndStops:
    shapes: Dict[str, List[Tuple[float, float]]]  # shape_id -> [(lat, lon), ...]
    rt_stops: Dict[str, Tuple[float, float]]      # stop_id -> (lat, lon)

def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    """Retourne une liste [(lat, lon), ...] à partir d'une polyline encodée Google."""
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

def _collect_shapes_stops_json(feed: Dict[str, Any]) -> RtShapesAndStops:
    shapes, rt_stops = {}, {}
    for e in feed.get('entity', []):
        if e.get('shape'):
            sh = e['shape']
            sid = str(sh.get('shape_id') or '')
            enc = sh.get('encoded_polyline')
            if sid and enc:
                try:
                    shapes[sid] = decode_polyline(enc)
                except Exception:
                    pass
        if e.get('stop'):
            stp = e['stop']
            sid = str(stp.get('stop_id') or '')
            lat = stp.get('stop_lat'); lon = stp.get('stop_lon')
            try:
                if sid and lat is not None and lon is not None:
                    rt_stops[sid] = (float(lat), float(lon))
            except Exception:
                pass
    return RtShapesAndStops(shapes=shapes, rt_stops=rt_stops)

def _collect_shapes_stops_pb(data: bytes) -> RtShapesAndStops:
    shapes, rt_stops = {}, {}
    proto = gtfs_local
    if proto is None:
        try:
            from google.transit import gtfs_realtime_pb2 as proto
        except Exception:
            return RtShapesAndStops(shapes, rt_stops)

    feed = proto.FeedMessage(); feed.ParseFromString(data)
    for ent in feed.entity:
        if hasattr(ent, 'shape') and ent.shape and getattr(ent.shape, 'encoded_polyline', None):
            sid = str(getattr(ent.shape, 'shape_id', '') or '')
            enc = ent.shape.encoded_polyline
            if sid and enc:
                try:
                    shapes[sid] = decode_polyline(enc)
                except Exception:
                    pass
        if hasattr(ent, 'stop') and ent.stop:
            sid = str(getattr(ent.stop, 'stop_id', '') or '')
            lat = getattr(ent.stop, 'stop_lat', None)
            lon = getattr(ent.stop, 'stop_lon', None)
            try:
                if sid and lat is not None and lon is not None:
                    rt_stops[sid] = (float(lat), float(lon))
            except Exception:
                pass
    return RtShapesAndStops(shapes=shapes, rt_stops=rt_stops)


def load_tripmods_bytes(file_bytes: bytes) -> Tuple[List[TripModEntity], RtShapesAndStops, Optional[Dict[str, Any]]]:
    fmt = _detect_tripmods_format_bytes(file_bytes)
    if fmt == 'json':
        feed = json.loads(file_bytes.decode('utf-8'))
        ents = parse_tripmods_json(feed)
        rt = _collect_shapes_stops_json(feed)
        return ents, rt, feed
    else:
        ents = parse_tripmods_protobuf(file_bytes)
        rt = _collect_shapes_stops_pb(file_bytes)
        return ents, rt, None


# ------------------------------------------------------------------------------
# 5) Chargement GTFS (zip) + index
# ------------------------------------------------------------------------------
@dataclass
class GtfsStatic:
    trips: Dict[str, Dict[str, str]]
    stop_times: Dict[str, List[Dict[str, str]]]
    stops: Dict[str, Dict[str, str]]

@st.cache_data(show_spinner=False)
def load_gtfs_zip_bytes(zip_bytes: bytes) -> GtfsStatic:
    trips: Dict[str, Dict[str, str]] = {}
    stop_times: Dict[str, List[Dict[str, str]]] = {}
    stops: Dict[str, Dict[str, str]] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        # trips.txt
        if 'trips.txt' in zf.namelist():
            with zf.open('trips.txt') as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    row = {k: (v or "").strip() for k, v in row.items()}
                    if 'trip_id' in row:
                        trips[row['trip_id']] = row

        # stop_times.txt
        if 'stop_times.txt' in zf.namelist():
            with zf.open('stop_times.txt') as f:
                rows = []
                for r in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    r = {k: (v or "").strip() for k, v in r.items()}
                    rows.append(r)
                for r in rows:
                    tid = r.get('trip_id') or ''
                    if not tid:
                        continue
                    lst = stop_times.setdefault(tid, [])
                    try:
                        r['stop_sequence'] = str(int(r.get('stop_sequence', '').strip()))
                    except Exception:
                        r['stop_sequence'] = ''
                    lst.append(r)
                for tid, lst in stop_times.items():
                    lst.sort(key=lambda x: int(x['stop_sequence'] or 0))

        # stops.txt
        if 'stops.txt' in zf.namelist():
            with zf.open('stops.txt') as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    row = {k: (v or "").strip() for k, v in row.items()}
                    if 'stop_id' in row:
                        stops[row['stop_id']] = row

    return GtfsStatic(trips=trips, stop_times=stop_times, stops=stops)


# ------------------------------------------------------------------------------
# 6) Analyse TripMods vs GTFS
# ------------------------------------------------------------------------------
@dataclass
class TripCheck:
    trip_id: str
    exists_in_gtfs: bool
    start_seq_valid: bool
    end_seq_valid: bool
    start_seq: Optional[int] = None
    end_seq: Optional[int] = None
    notes: List[str] = field(default_factory=list)

@dataclass
class EntityReport:
    entity_id: str
    total_selected_trip_ids: int
    service_dates: List[str]
    modification_count: int
    trips: List[TripCheck]
    replacement_stops_unknown_in_gtfs: List[str] = field(default_factory=list)

def _seq_from_selector(sel: StopSelector, stop_times_list: List[Dict[str, str]]) -> Optional[int]:
    if sel is None:
        return None
    if sel.stop_sequence is not None:
        return sel.stop_sequence
    if sel.stop_id:
        for r in stop_times_list:
            if r.get('stop_id') == sel.stop_id:
                try:
                    return int(r.get('stop_sequence') or 0)
                except Exception:
                    return None
    return None

def analyze_tripmods_with_gtfs(gtfs: GtfsStatic, ents: List[TripModEntity]) -> Tuple[List[EntityReport], Dict[str, int]]:
    reports: List[EntityReport] = []
    totals = dict(total_entities=len(ents), total_trip_ids=0, total_modifications=0,
                  missing_trip_ids=0, invalid_selectors=0, unknown_replacement_stops=0)

    for e in ents:
        trip_checks: List[TripCheck] = []
        repl_unknown: List[str] = []
        tot_trip_ids = sum(len(sel.trip_ids) for sel in e.selected_trips)

        for sel in e.selected_trips:
            for trip_id in sel.trip_ids:
                exists = trip_id in gtfs.trips
                st_list = gtfs.stop_times.get(trip_id, [])
                start_ok = end_ok = False
                start_seq = end_seq = None
                notes: List[str] = []
                if not exists:
                    notes.append("trip_id absent du GTFS")
                    totals["missing_trip_ids"] += 1
                else:
                    for m in e.modifications:
                        sseq = _seq_from_selector(m.start_stop_selector, st_list) if st_list else None
                        eseq = _seq_from_selector(m.end_stop_selector,   st_list) if st_list else None
                        if start_seq is None and sseq is not None:
                            start_seq = sseq
                        if end_seq is None and eseq is not None:
                            end_seq = eseq
                        if sseq is None or eseq is None:
                            notes.append("start/end selector non résolu sur ce trip")
                            totals["invalid_selectors"] += 1
                        else:
                            start_ok = True
                            end_ok = True

                trip_checks.append(TripCheck(trip_id=trip_id, exists_in_gtfs=exists,
                                             start_seq_valid=start_ok, end_seq_valid=end_ok,
                                             start_seq=start_seq, end_seq=end_seq, notes=notes))

        for m in e.modifications:
            for rs in m.replacement_stops:
                sid = rs.stop_id
                if sid and sid not in gtfs.stops:
                    repl_unknown.append(sid)
                    totals["unknown_replacement_stops"] += 1

        totals["total_trip_ids"] += tot_trip_ids
        totals["total_modifications"] += len(e.modifications)

        reports.append(EntityReport(
            entity_id=e.entity_id,
            total_selected_trip_ids=tot_trip_ids,
            service_dates=e.service_dates,
            modification_count=len(e.modifications),
            trips=trip_checks,
            replacement_stops_unknown_in_gtfs=sorted(set(repl_unknown))
        ))

    return reports, totals


# ------------------------------------------------------------------------------
# 7) Altair — construction des calques (route, tronçon, marqueurs)
# ------------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _nearest_index_on_polyline(poly: List[Tuple[float, float]], lat: float, lon: float) -> Optional[int]:
    if not poly:
        return None
    dmin, imin = float('inf'), None
    for i, (la, lo) in enumerate(poly):
        d = _haversine(lat, lon, la, lo)
        if d < dmin:
            dmin, imin = d, i
    return imin

def _build_altair_layers_for_entity(
    ent: TripModEntity,
    rt: RtShapesAndStops,
    gtfs: GtfsStatic
) -> Optional[alt.Chart]:
    # 1) shape_id : on prend le premier groupe selected_trips qui le fournit
    shape_id = None
    for s in ent.selected_trips:
        if s.shape_id:
            shape_id = s.shape_id
            break
    if not shape_id or shape_id not in rt.shapes:
        return None

    poly = rt.shapes[shape_id]  # [(lat, lon), ...]
    if len(poly) < 2:
        return None

    # DataFrame complet de la polyline
    df_route = pd.DataFrame([{"lat": la, "lon": lo, "seg": "route"} for la, lo in poly])

    # 2) Choisir un trip_id existant dans le GTFS pour résoudre les selectors
    trip_id_ref = None
    for s in ent.selected_trips:
        for tid in s.trip_ids:
            if tid in gtfs.trips:
                trip_id_ref = tid; break
        if trip_id_ref: break

    df_segment = None
    df_markers = []

    if trip_id_ref:
        st_list = gtfs.stop_times.get(trip_id_ref, [])

        if ent.modifications:
            m = ent.modifications[0]  # si plusieurs, on peut itérer/ajouter un selecteur

            def _coord_from_selector(sel: StopSelector) -> Optional[Tuple[float, float, str]]:
                if sel is None: return None
                # priorité stop_id -> coordonnées GTFS (ou RT si temporaire)
                if sel.stop_id:
                    if sel.stop_id in gtfs.stops:
                        r = gtfs.stops[sel.stop_id]
                        try: return (float(r.get('stop_lat')), float(r.get('stop_lon')), sel.stop_id)
                        except: pass
                    if sel.stop_id in rt.rt_stops:
                        la, lo = rt.rt_stops[sel.stop_id]; return (la, lo, sel.stop_id)
                # fallback sur stop_sequence
                if sel.stop_sequence is not None:
                    for r in st_list:
                        try:
                            if int(r.get('stop_sequence') or -999) == sel.stop_sequence:
                                return (float(r.get('stop_lat')), float(r.get('stop_lon')), r.get('stop_id'))
                        except: pass
                return None

            start_c = _coord_from_selector(m.start_stop_selector)
            end_c   = _coord_from_selector(m.end_stop_selector)

            if start_c and end_c:
                s_idx = _nearest_index_on_polyline(poly, start_c[0], start_c[1])
                e_idx = _nearest_index_on_polyline(poly, end_c[0], end_c[1])
                if s_idx is not None and e_idx is not None and e_idx > s_idx:
                    sub = poly[s_idx:e_idx+1]
                    df_segment = pd.DataFrame([{"lat": la, "lon": lo, "seg": "troncon"} for la, lo in sub])
                # marqueurs start/end
                df_markers.append({"type":"start", "lat": start_c[0], "lon": start_c[1], "label": start_c[2] or "start"})
                df_markers.append({"type":"end",   "lat": end_c[0],   "lon": end_c[1],   "label": end_c[2] or "end"})

            # 3) Arrêts de remplacement
            order = 1
            for rs in m.replacement_stops:
                sid = rs.stop_id
                la_lo = None
                if sid in gtfs.stops:
                    r = gtfs.stops[sid]
                    try: la_lo = (float(r.get('stop_lat')), float(r.get('stop_lon')))
                    except: pass
                if not la_lo and sid in rt.rt_stops:
                    la_lo = rt.rt_stops[sid]
                if la_lo:
                    df_markers.append({"type":"repl", "lat": la_lo[0], "lon": la_lo[1], "label": f"{order}·{sid}"})
                order += 1

    # DataFrames -> Altair
    base = alt.Chart(df_route).mark_line(color="#9aa0a6", strokeWidth=2).encode(
        x=alt.X("lon:Q", title="Longitude"),
        y=alt.Y("lat:Q", title="Latitude")
    )

    layers = [base]
    if df_segment is not None and not df_segment.empty:
        layers.append(
            alt.Chart(df_segment).mark_line(color="#d93025", strokeWidth=3).encode(
                x="lon:Q", y="lat:Q"
            )
        )

    if df_markers:
        dfm = pd.DataFrame(df_markers)
        color_scale = alt.Scale(domain=["start","end","repl"], range=["#188038","#202124","#f9ab00"])
        layers.append(
            alt.Chart(dfm).mark_point(size=80, filled=True).encode(
                x="lon:Q", y="lat:Q", color=alt.Color("type:N", scale=color_scale, legend=alt.Legend(title="Marqueur"))
            )
        )
        layers.append(
            alt.Chart(dfm).mark_text(dy=-10).encode(
                x="lon:Q", y="lat:Q", text="label:N", color=alt.Color("type:N", scale=color_scale, legend=None)
            )
        )

    chart = alt.layer(*layers).properties(width="container", height=380).interactive()
    return chart


# ------------------------------------------------------------------------------
# 8) UI Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Analyse TripModifications + GTFS", layout="wide")

st.title("Analyse TripModifications (JSON/PB) vs GTFS")
st.caption("Charge un GTFS statique (.zip) et un fichier TripModifications (.json ou .pb), puis lance l’analyse. Visualisation Altair des détours incluse.")

with st.sidebar:
    st.header("Données d’entrée")
    gtfs_file = st.file_uploader("GTFS (.zip)", type=["zip"])
    tripmods_file = st.file_uploader("TripModifications (.json/.pb)", type=["json", "pb", "pbf", "bin"])
    dump_first = st.checkbox("Afficher le 1er trip_mod normalisé", value=False)
    run_btn = st.button("Analyser", type="primary")

if run_btn:
    if not gtfs_file or not tripmods_file:
        st.error("Merci de sélectionner un GTFS (.zip) **et** un TripModifications (.json/.pb).")
        st.stop()

    with st.spinner("Chargement du GTFS…"):
        gtfs = load_gtfs_zip_bytes(gtfs_file.getvalue())

    with st.spinner("Parsing TripModifications…"):
        try:
            ents, rt, raw_json_feed = load_tripmods_bytes(tripmods_file.getvalue())
        except Exception as ex:
            st.exception(ex)
            st.stop()

    st.success(f"GTFS chargé : **{len(gtfs.trips):,} trips**, "
               f"**{sum(len(v) for v in gtfs.stop_times.values()):,} stop_times**, "
               f"**{len(gtfs.stops):,} stops**")
    st.success(f"TripModifications : **{len(ents)} entités**")

    if dump_first and ents:
        with st.expander("Aperçu du 1er trip_mod (normalisé)"):
            st.json(asdict(ents[0]))
        if raw_json_feed is not None:
            with st.expander("Aperçu brut du feed JSON (optionnel)"):
                st.json(raw_json_feed)

    with st.spinner("Analyse en cours…"):
        reports, totals = analyze_tripmods_with_gtfs(gtfs, ents)

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Entités", totals["total_entities"])
    c2.metric("trip_ids sélectionnés", totals["total_trip_ids"])
    c3.metric("modifications", totals["total_modifications"])
    c4.metric("trip_ids manquants", totals["missing_trip_ids"])
    c5.metric("selectors non résolus", totals["invalid_selectors"])
    c6.metric("repl. stops inconnus GTFS", totals["unknown_replacement_stops"])

    # Tableau synthèse par entité
    table = [{
        "entity_id": r.entity_id,
        "trip_ids (sélectionnés)": r.total_selected_trip_ids,
        "modifications": r.modification_count,
        "service_dates": ", ".join(r.service_dates),
        "repl_stops inconnus": ", ".join(r.replacement_stops_unknown_in_gtfs) if r.replacement_stops_unknown_in_gtfs else ""
    } for r in reports]
    st.subheader("Synthèse par entité")
    st.dataframe(table, use_container_width=True, height=360)

    # Détails par entité (avec Altair)
    st.subheader("Détails")
    for r in reports[:200]:  # limite d'affichage
        with st.expander(f"Entité {r.entity_id} — {r.total_selected_trip_ids} trips — {r.modification_count} modifications"):
            st.write("**Dates** :", ", ".join(r.service_dates) if r.service_dates else "—")
            st.write("**Replacement stops inconnus dans GTFS (peuvent être temporaires)** :",
                     ", ".join(r.replacement_stops_unknown_in_gtfs) if r.replacement_stops_unknown_in_gtfs else "—")

            # Tableau par trip
            detail_rows = [{
                "trip_id": t.trip_id,
                "existe dans GTFS": "oui" if t.exists_in_gtfs else "non",
                "start_seq": t.start_seq if t.start_seq is not None else "",
                "end_seq": t.end_seq if t.end_seq is not None else "",
                "selectors OK": "oui" if (t.start_seq_valid and t.end_seq_valid) else "non",
                "notes": "; ".join(t.notes) if t.notes else ""
            } for t in r.trips]
            st.dataframe(detail_rows, use_container_width=True, height=240)

            # Visualisation Altair (route + tronçon + marqueurs)
            ent = next((e for e in ents if e.entity_id == r.entity_id), None)
            if ent:
                chart = _build_altair_layers_for_entity(ent, rt, gtfs)
                if chart is not None:
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Pas de shape exploitable pour visualiser ce détour.")

    # Export du rapport brut
    report_json = {
        "totals": totals,
        "entities": [asdict(r) for r in reports]
    }
    st.download_button(
        "📥 Télécharger le rapport JSON",
        data=json.dumps(report_json, ensure_ascii=False, indent=2),
        file_name="rapport_tripmods.json",
        mime="application/json"
    )

else:
    st.info("Charge un GTFS (.zip) et un TripModifications (.json/.pb), puis clique **Analyser**.")
    st.caption("Le parseur suit la structure du feed (selected_trips, service_dates, modifications, etc.).")
