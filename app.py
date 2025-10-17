# app.py — Analyse TripModifications (JSON/PB) vs GTFS — Mémoire réduite + polyline fiable
# Remplacements : use_container_width -> width='stretch' ; déséchappage sans unicode_escape

from __future__ import annotations
import streamlit as st
import json, csv, io, zipfile, sys, re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Tuple, Set
from pathlib import Path

import pandas as pd
import altair as alt

# -----------------------------------------------------------------------------------------------
# 0) Import protobuf local si dispo (gtfs_realtime_pb2.py)
# -----------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import gtfs_realtime_pb2 as gtfs_local
except Exception:
    gtfs_local = None

# -----------------------------------------------------------------------------------------------
# 1) camelCase → snake_case
# -----------------------------------------------------------------------------------------------
_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')
def _camel_to_snake(name: str) -> str:
    return _CAMEL_RE.sub('_', name).lower()

def _normalize_json_keys(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = _camel_to_snake(k) if isinstance(k, str) else k
            out[nk] = _normalize_json_keys(v)
        return out
    if isinstance(obj, list):
        return [_normalize_json_keys(x) for x in obj]
    return obj

# -----------------------------------------------------------------------------------------------
# 2) Modèles
# -----------------------------------------------------------------------------------------------
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

@dataclass
class GtfsStatic:
    trips: Dict[str, Dict[str, str]]
    stop_times: Dict[str, List[Dict[str, str]]]
    stops_present: Set[str]

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

@dataclass
class RtShapes:
    shapes: Dict[str, List[Tuple[float, float]]]  # shape_id -> [(lat, lon), ...]

# -----------------------------------------------------------------------------------------------
# 3) Décodage polyline (nettoyage SANS unicode_escape) + Auto/5/6 + fallback
# -----------------------------------------------------------------------------------------------
def _sanitize_polyline(s: str) -> str:
    """
    Nettoyage minimal conforme aux recommandations Google :
    - retire espaces/retours (si collé depuis JSON)
    - remplace \\n, \\r, \\t par vide
    - remplace \\\\ par \\  (déséchappage 'backslash' simple)
    Réf. : l’outil Google demande d’« unescape » les polylignes JSON. 
    """
    if not s:
        return s
    s = s.strip()
    # nettoyer séquences JSON fréquentes
    if "\\n" in s or "\\r" in s or "\\t" in s or "\\\\" in s:
        s = s.replace("\\n", "").replace("\\r", "").replace("\\t", "")
        s = s.replace("\\\\", "\\")
    # supprimer espaces/blancs résiduels
    s = re.sub(r"\s+", "", s)
    return s

def _legacy_decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    coords = []
    index, lat, lon = 0, 0, 0
    encoded = (encoded or "").strip()
    L = len(encoded)
    while index < L:
        result = 0; shift = 0
        while True:
            b = ord(encoded[index]) - 63; index += 1
            result |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        result = 0; shift = 0
        while True:
            b = ord(encoded[index]) - 63; index += 1
            result |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlon = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += dlon
        coords.append((lat / 1e5, lon / 1e5))
    return coords

def _valid_coords(cs: List[Tuple[float,float]]) -> bool:
    return bool(cs) and all(-90 <= la <= 90 and -180 <= lo <= 180 for la, lo in cs)

def decode_polyline(encoded: str, mode: str = "auto") -> List[Tuple[float, float]]:
    enc = _sanitize_polyline(encoded)
    if not enc:
        return []
    try:
        import polyline as pl
    except Exception:
        return _legacy_decode_polyline(enc)
    if mode == "p5":
        try:
            c5 = pl.decode(enc, precision=5)
            return c5 if _valid_coords(c5) else _legacy_decode_polyline(enc)
        except Exception:
            return _legacy_decode_polyline(enc)
    if mode == "p6":
        try:
            c6 = pl.decode(enc, precision=6)
            return c6 if _valid_coords(c6) else _legacy_decode_polyline(enc)
        except Exception:
            return _legacy_decode_polyline(enc)
    # auto
    try:
        c5 = pl.decode(enc, precision=5)
    except Exception:
        c5 = []
    try:
        c6 = pl.decode(enc, precision=6)
    except Exception:
        c6 = []
    if _valid_coords(c5) and not _valid_coords(c6):
        return c5
    if _valid_coords(c6) and not _valid_coords(c5):
        return c6
    if _valid_coords(c5) and _valid_coords(c6):
        def span(cs): 
            lats = [la for la,_ in cs]; lons = [lo for _,lo in cs]
            return (max(lats)-min(lats)) + (max(lons)-min(lons))
        return c6 if span(c6) > span(c5)*1.3 else c5
    return _legacy_decode_polyline(enc)

# -----------------------------------------------------------------------------------------------
# 4) Parsing TripMods + shapes (encoded_polyline)
# -----------------------------------------------------------------------------------------------
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
    if not isinstance(obj, dict): return None
    sid = obj.get('stop_id')
    if not sid: return None
    t = obj.get('travel_time_to_stop', 0)
    try: t = int(t)
    except Exception: t = 0
    return ReplacementStop(stop_id=str(sid), travel_time_to_stop=t)

def _coerce_selected_trips(obj: Dict[str, Any]) -> SelectedTrips:
    trips = obj.get('trip_ids') or []
    if isinstance(trips, str): trips = [trips]
    trips = [str(t).strip() for t in trips if str(t).strip()]
    shape_id = obj.get('shape_id')
    shape_id = str(shape_id) if shape_id not in (None, '') else None
    return SelectedTrips(trip_ids=trips, shape_id=shape_id)

def parse_tripmods_json(feed: Dict[str, Any]) -> List[TripModEntity]:
    entities = feed.get('entity') or []
    out: List[TripModEntity] = []
    for e in entities:
        tm = e.get('trip_modifications')
        if not tm: continue
        sel_raw = tm.get('selected_trips') or []
        selected = [_coerce_selected_trips(s) for s in sel_raw]
        service_dates = tm.get('service_dates') or []
        if not service_dates:
            for s in sel_raw:
                dates = s.get('service_dates')
                if dates: service_dates.extend(dates)
        service_dates = [str(d) for d in service_dates]
        start_times = [str(t) for t in tm.get('start_times') or []]
        mods: List[Modification] = []
        for m in tm.get('modifications') or []:
            repl_list = []
            for rs in m.get('replacement_stops') or []:
                r = _coerce_repl_stop(rs)
                if r: repl_list.append(r)
            start_sel = _coerce_selector(m.get('start_stop_selector') or {})
            end_sel   = _coerce_selector(m.get('end_stop_selector') or {})
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

def _collect_shapes_json(feed: Dict[str, Any], decode_mode: str) -> RtShapes:
    shapes: Dict[str, List[Tuple[float, float]]] = {}
    for e in feed.get('entity', []):
        sh = e.get('shape')
        if sh and isinstance(sh, dict):
            sid = str(sh.get('shape_id') or '')
            enc = sh.get('encoded_polyline')
            if sid and enc:
                try:
                    shapes[sid] = decode_polyline(enc, mode=decode_mode)
                except Exception:
                    pass
    return RtShapes(shapes=shapes)

def parse_tripmods_protobuf(data: bytes) -> List[TripModEntity]:
    proto = gtfs_local
    if proto is None:
        from google.transit import gtfs_realtime_pb2 as proto
    feed = proto.FeedMessage(); feed.ParseFromString(data)
    out: List[TripModEntity] = []
    for ent in feed.entity:
        if not hasattr(ent, 'trip_modifications'): continue
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
            repl = [ReplacementStop(stop_id=rs.stop_id,
                                    travel_time_to_stop=int(getattr(rs, 'travel_time_to_stop', 0)))
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

def _collect_shapes_pb(data: bytes, decode_mode: str) -> RtShapes:
    shapes: Dict[str, List[Tuple[float, float]]] = {}
    proto = gtfs_local
    if proto is None:
        try:
            from google.transit import gtfs_realtime_pb2 as proto
        except Exception:
            return RtShapes(shapes)
    feed = proto.FeedMessage(); feed.ParseFromString(data)
    for ent in feed.entity:
        if hasattr(ent, 'shape') and ent.shape and getattr(ent.shape, 'encoded_polyline', None):
            sid = str(getattr(ent.shape, 'shape_id', '') or '')
            enc = ent.shape.encoded_polyline
            if sid and enc:
                try:
                    shapes[sid] = decode_polyline(enc, mode=decode_mode)
                except Exception:
                    pass
    return RtShapes(shapes=shapes)

def load_tripmods_bytes(file_bytes: bytes, decode_mode: str) -> Tuple[List[TripModEntity], Optional[Dict[str, Any]], RtShapes]:
    fmt = _detect_tripmods_format_bytes(file_bytes)
    if fmt == 'json':
        raw = json.loads(file_bytes.decode('utf-8'))
        feed = _normalize_json_keys(raw)
        ents = parse_tripmods_json(feed)
        shapes = _collect_shapes_json(feed, decode_mode)
        needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
        if needed_shape_ids:
            shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
        return ents, feed, shapes
    else:
        ents = parse_tripmods_protobuf(file_bytes)
        shapes = _collect_shapes_pb(file_bytes, decode_mode)
        needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
        if needed_shape_ids:
            shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
        return ents, None, shapes

# -----------------------------------------------------------------------------------------------
# 5) Ensembles cibles (GTFS filtré)
# -----------------------------------------------------------------------------------------------
def compute_needed_sets(ents: List[TripModEntity]) -> Tuple[Set[str], Set[str]]:
    needed_trip_ids: Set[str] = set()
    needed_stop_ids: Set[str] = set()
    for e in ents:
        for s in e.selected_trips:
            needed_trip_ids.update([str(tid) for tid in s.trip_ids if tid])
        for m in e.modifications:
            for rs in m.replacement_stops:
                if rs.stop_id:
                    needed_stop_ids.add(str(rs.stop_id))
            for sel in (m.start_stop_selector, m.end_stop_selector):
                if sel and sel.stop_id:
                    needed_stop_ids.add(str(sel.stop_id))
    return needed_trip_ids, needed_stop_ids

# -----------------------------------------------------------------------------------------------
# 6) Chargement GTFS FILTRÉ
# -----------------------------------------------------------------------------------------------
def load_gtfs_zip_filtered_bytes(zip_bytes: bytes, needed_trip_ids: Set[str], needed_stop_ids: Set[str]) -> GtfsStatic:
    trips: Dict[str, Dict[str, str]] = {}
    stop_times: Dict[str, List[Dict[str, str]]] = {}
    stops_present: Set[str] = set()
    if not needed_trip_ids and not needed_stop_ids:
        return GtfsStatic(trips=trips, stop_times=stop_times, stops_present=stops_present)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        if 'trips.txt' in zf.namelist() and needed_trip_ids:
            with zf.open('trips.txt') as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    tid = (row.get('trip_id') or '').strip()
                    if tid in needed_trip_ids:
                        trips[tid] = {k: (v or "").strip() for k, v in row.items()}
        if 'stop_times.txt' in zf.namelist() and needed_trip_ids:
            with zf.open('stop_times.txt') as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline=''))
                for r in reader:
                    tid = (r.get('trip_id') or '').strip()
                    if tid in needed_trip_ids:
                        rec = {k: (v or "").strip() for k, v in r.items()}
                        try:
                            rec['stop_sequence'] = str(int(rec.get('stop_sequence', '').strip()))
                        except Exception:
                            rec['stop_sequence'] = ''
                        stop_times.setdefault(tid, []).append(rec)
            for tid, lst in stop_times.items():
                lst.sort(key=lambda x: int(x['stop_sequence'] or 0))
        if 'stops.txt' in zf.namelist() and needed_stop_ids:
            with zf.open('stops.txt') as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    sid = (row.get('stop_id') or '').strip()
                    if sid in needed_stop_ids:
                        stops_present.add(sid)
    return GtfsStatic(trips=trips, stop_times=stop_times, stops_present=stops_present)

# -----------------------------------------------------------------------------------------------
# 7) Analyse
# -----------------------------------------------------------------------------------------------
def _seq_from_selector(sel: StopSelector, stop_times_list: List[Dict[str, str]]) -> Optional[int]:
    if sel is None:
        return None
    if sel.stop_sequence is not None:
        return sel.stop_sequence
    if sel.stop_id:
        for r in stop_times_list:
            if r.get('stop_id') == sel.stop_id:
                try: return int(r.get('stop_sequence') or 0)
                except Exception: return None
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
                    notes.append("trip_id absent du GTFS (filtré ou inexistant)")
                    totals["missing_trip_ids"] += 1
                else:
                    for m in e.modifications:
                        sseq = _seq_from_selector(m.start_stop_selector, st_list) if st_list else None
                        eseq = _seq_from_selector(m.end_stop_selector, st_list) if st_list else None
                        if start_seq is None and sseq is not None: start_seq = sseq
                        if end_seq is None and eseq is not None:   end_seq = eseq
                        if sseq is None or eseq is None:
                            notes.append("start/end selector non résolu sur ce trip")
                            totals["invalid_selectors"] += 1
                        else:
                            start_ok = True; end_ok = True
                trip_checks.append(TripCheck(
                    trip_id=trip_id, exists_in_gtfs=exists,
                    start_seq_valid=start_ok, end_seq_valid=end_ok,
                    start_seq=start_seq, end_seq=end_seq, notes=notes
                ))
        for m in e.modifications:
            for rs in m.replacement_stops:
                sid = rs.stop_id
                if sid and sid not in gtfs.stops_present:
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

# -----------------------------------------------------------------------------------------------
# 8) Visualisation (Altair)
# -----------------------------------------------------------------------------------------------
def _compute_domain(coords: List[Tuple[float, float]], frac: float = 0.03, floor: float = 1e-4):
    if not coords:
        return (-180.0, 180.0), (-90.0, 90.0)
    lats = [la for la, _ in coords]; lons = [lo for _, lo in coords]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    span_lat = max(lat_max - lat_min, floor)
    span_lon = max(lon_max - lon_min, floor)
    pad_lat = max(span_lat * frac, floor)
    pad_lon = max(span_lon * frac, floor)
    return (lon_min - pad_lon, lon_max + pad_lon), (lat_min - pad_lat, lat_max + pad_lat)

def build_chart_for_polyline(poly: List[Tuple[float, float]], shape_id: Optional[str] = None):
    if not poly or len(poly) < 2:
        return None
    df_route = pd.DataFrame([{"lat": la, "lon": lo} for la, lo in poly])
    (lon_min, lon_max), (lat_min, lat_max) = _compute_domain(poly)
    x_scale = alt.Scale(domain=[lon_min, lon_max], zero=False, nice=False)
    y_scale = alt.Scale(domain=[lat_min, lat_max], zero=False, nice=False)
    title = f"Tracé (encoded_polyline){' — ' + shape_id if shape_id else ''}"
    chart = (
        alt.Chart(df_route, title=title)
           .mark_line(color="#1f77b4", strokeWidth=3)
           .encode(
               x=alt.X("lon:Q", title="Longitude", scale=x_scale),
               y=alt.Y("lat:Q", title="Latitude", scale=y_scale)
           )
           .properties(width="container", height=360)  # largeur gérée par st.altair_chart(..., width=...)
           .interactive()
    )
    return chart

# -----------------------------------------------------------------------------------------------
# 9) UI Streamlit
# -----------------------------------------------------------------------------------------------
st.set_page_config(page_title="Analyse TripModifications + GTFS — Mémoire réduite + polyline fiable", layout="wide")
st.title("Analyse TripModifications (JSON/PB) vs GTFS — Mémoire réduite")
st.caption("Le JSON est normalisé (camelCase → snake_case). Le GTFS est filtré. "
           "Les polylines sont nettoyées (déséchappage léger) et décodées en Auto/1e-5/1e-6.")

with st.sidebar:
    st.header("Données d’entrée")
    tripmods_file = st.file_uploader("TripModifications (.json/.pb)", type=["json", "pb", "pbf", "bin"])
    gtfs_file = st.file_uploader("GTFS (.zip)", type=["zip"])
    decode_mode = st.selectbox("Décodage polylines", ["Auto (recommandé)", "Précision 1e-5", "Précision 1e-6"], index=0)
    dump_first = st.checkbox("Afficher le 1er trip_mod normalisé", value=False)
    run_btn = st.button("Analyser", type="primary")

decode_flag = {"Auto (recommandé)": "auto", "Précision 1e-5": "p5", "Précision 1e-6": "p6"}[decode_mode]

# Laboratoire polyline
with st.expander("🧪 Laboratoire polyline (tester une chaîne à part)"):
    s = st.text_area("Colle ici une encoded_polyline (copiée de ton outil Google).", height=100)
    if st.button("Décoder la chaîne de test"):
        pts = decode_polyline(s, mode=decode_flag)
        st.write(f"Points décodés : {len(pts)}")
        if pts:
            chart = build_chart_for_polyline(pts, shape_id="(test)")
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            st.dataframe(pd.DataFrame(pts, columns=["lat", "lon"]).head(20), width="stretch")

if run_btn:
    if not tripmods_file or not gtfs_file:
        st.error("Merci de sélectionner **TripModifications** (.json/.pb) **et** **GTFS** (.zip).")
        st.stop()

    with st.spinner("Parsing TripModifications…"):
        try:
            ents, feed_json, rt_shapes = load_tripmods_bytes(tripmods_file.getvalue(), decode_mode=decode_flag)
        except Exception as ex:
            st.exception(ex); st.stop()

    needed_trip_ids, needed_stop_ids = compute_needed_sets(ents)
    st.info(f"Filtrage GTFS → trips requis: {len(needed_trip_ids):,} · stops requis: {len(needed_stop_ids):,} · shapes disponibles: {len(rt_shapes.shapes):,}")

    with st.spinner("Chargement GTFS (filtré)…"):
        try:
            gtfs = load_gtfs_zip_filtered_bytes(gtfs_file.getvalue(), needed_trip_ids, needed_stop_ids)
        except Exception as ex:
            st.exception(ex); st.stop()

    st.success(f"GTFS filtré : **{len(gtfs.trips):,} trips conservés**, "
               f"**{sum(len(v) for v in gtfs.stop_times.values()):,} stop_times**, "
               f"**{len(gtfs.stops_present):,} stops présents**")
    st.success(f"TripModifications : **{len(ents)} entités**")

    if dump_first and ents:
        with st.expander("Aperçu du 1er trip_mod (normalisé)"):
            st.json(asdict(ents[0]))
    if feed_json is not None:
        with st.expander("Aperçu brut du feed JSON (après normalisation camel→snake)"):
            st.json(feed_json)

    with st.spinner("Analyse en cours…"):
        reports, totals = analyze_tripmods_with_gtfs(gtfs, ents)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Entités", totals["total_entities"])
    c2.metric("trip_ids sélectionnés", totals["total_trip_ids"])
    c3.metric("modifications", totals["total_modifications"])
    c4.metric("trip_ids manquants", totals["missing_trip_ids"])
    c5.metric("selectors non résolus", totals["invalid_selectors"])
    c6.metric("repl. stops inconnus GTFS", totals["unknown_replacement_stops"])

    table = [{
        "entity_id": r.entity_id,
        "trip_ids (sélectionnés)": r.total_selected_trip_ids,
        "modifications": r.modification_count,
        "service_dates": ", ".join(r.service_dates),
        "repl_stops inconnus": ", ".join(r.replacement_stops_unknown_in_gtfs) if r.replacement_stops_unknown_in_gtfs else ""
    } for r in reports]
    st.subheader("Synthèse par entité")
    st.dataframe(table, width="stretch", height=360)

    st.subheader("Détails")
    for r in reports[:200]:
        with st.expander(f"Entité {r.entity_id} — {r.total_selected_trip_ids} trips — {r.modification_count} modifications"):
            st.write("**Dates** :", ", ".join(r.service_dates) if r.service_dates else "—")
            st.write("**Replacement stops inconnus dans GTFS (peuvent être temporaires)** :",
                     ", ".join(r.replacement_stops_unknown_in_gtfs) if r.replacement_stops_unknown_in_gtfs else "—")

            ent_obj = next((e for e in ents if e.entity_id == r.entity_id), None)
            shape_ids_in_entity = []
            if ent_obj:
                for s in ent_obj.selected_trips:
                    if s.shape_id:
                        shape_ids_in_entity.append(s.shape_id)
            shape_ids_set = set(shape_ids_in_entity)
            mixed_shapes = len([sid for sid in shape_ids_set if sid]) > 1

            trip_counts: Dict[str, int] = {}
            if ent_obj:
                for s in ent_obj.selected_trips:
                    for tid in s.trip_ids:
                        trip_counts[tid] = trip_counts.get(tid, 0) + 1

            detail_rows = []
            for t in r.trips:
                st_list = gtfs.stop_times.get(t.trip_id, [])
                stop_times_count = len(st_list)
                trip_shape_id = None
                if ent_obj:
                    for s in ent_obj.selected_trips:
                        if t.trip_id in s.trip_ids and s.shape_id:
                            trip_shape_id = s.shape_id
                            break
                if not trip_shape_id and shape_ids_in_entity:
                    trip_shape_id = shape_ids_in_entity[0]
                poly = rt_shapes.shapes.get(trip_shape_id, []) if trip_shape_id else []
                poly_points = len(poly)

                def _is_poly_anormal(coords: List[Tuple[float,float]]) -> bool:
                    if not coords or len(coords) < 2: return True
                    lats = [la for la,_ in coords]; lons = [lo for _,lo in coords]
                    return (max(lats)-min(lats) + max(lons)-min(lons)) < 1e-4

                poly_anormal = _is_poly_anormal(poly)
                selectors_incomplets = not (t.start_seq_valid and t.end_seq_valid)
                ordre_ok = ""
                ecart_seq = ""
                if (t.start_seq is not None) and (t.end_seq is not None):
                    ordre_ok = "oui" if t.start_seq <= t.end_seq else "non"
                    ecart_seq = t.end_seq - t.start_seq
                duplicate_trip = trip_counts.get(t.trip_id, 0) > 1

                detail_rows.append({
                    "trip_id": t.trip_id,
                    "existe dans GTFS": "oui" if t.exists_in_gtfs else "non",
                    "start_seq": t.start_seq if t.start_seq is not None else "",
                    "end_seq": t.end_seq if t.end_seq is not None else "",
                    "selectors OK": "oui" if (t.start_seq_valid and t.end_seq_valid) else "non",
                    "notes": "; ".join(t.notes) if t.notes else "",
                    "shape_id (trip)": trip_shape_id or "",
                    "shape dispo": "oui" if (trip_shape_id and poly_points >= 2) else "non",
                    "pts polyline": poly_points,
                    "polyline anormale": "oui" if poly_anormal else "non",
                    "stop_times (nb)": stop_times_count,
                    "ordre start<=end": ordre_ok,
                    "écart seq": ecart_seq,
                    "selectors incomplets": "oui" if selectors_incomplets else "non",
                    "trip en double (entité)": "oui" if duplicate_trip else "non",
                    "mixed shapes (entité)": "oui" if mixed_shapes else "non",
                    "repl_stops inconnus (entité)": len(r.replacement_stops_unknown_in_gtfs)
                })

            st.dataframe(detail_rows, width="stretch", height=260)

            shape_id_for_plot = next((sid for sid in shape_ids_in_entity if sid in rt_shapes.shapes), None)
            if shape_id_for_plot:
                poly = rt_shapes.shapes.get(shape_id_for_plot, [])
                chart = build_chart_for_polyline(poly, shape_id=shape_id_for_plot)
                if chart is not None:
                    # APRÈS (compatible)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Polyline vide ou invalide pour cette entité.")
            else:
                st.info("Aucune polyline 'encoded_polyline' disponible pour cette entité.")

    report_json = {"totals": totals, "entities": [asdict(r) for r in reports]}
    st.download_button("📥 Télécharger le rapport JSON",
                       data=json.dumps(report_json, ensure_ascii=False, indent=2),
                       file_name="rapport_tripmods.json", mime="application/json")

else:
    st.info("Charge un TripModifications (.json/.pb) puis un GTFS (.zip), choisis la précision de décodage, et clique **Analyser**.")
    st.caption("Les polylignes copiées depuis du JSON doivent être « déséchappées » (retrait des \\ et \\n).")
