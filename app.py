# app.py — Analyse TripModifications (JSON/PB/Textproto) vs GTFS — "TripMods only"
# --------------------------------------------------------------------------------
# - Traite UNIQUEMENT les entités qui contiennent trip_modifications
# - Ignore totalement les entités 'shape' et les ReplacementStop (arrêts temporaires)
# - GTFS filtré (mémoire réduite) sur la base des trip_ids sélectionnés
# - Détails/anomalies: uniquement sur la structure TripMods + stop_times
# - Labo polyline (🧪) conservé pour debug manuel (hors analyse principale)
# - Compat Streamlit: st.altair_chart(..., use_container_width=True) ; st.dataframe(..., width="stretch")

from __future__ import annotations
import streamlit as st
import json, csv, io, zipfile, sys, re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Tuple, Set
from pathlib import Path

import pandas as pd
import altair as alt

# --------------------------------------------------------------------------------
# 0) Import protobuf local si dispo (gtfs_realtime_pb2.py)
# --------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import gtfs_realtime_pb2 as gtfs_local
except Exception:
    gtfs_local = None

# --------------------------------------------------------------------------------
# 1) camelCase → snake_case
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# 2) Modèles
# --------------------------------------------------------------------------------
@dataclass
class StopSelector:
    stop_sequence: Optional[int] = None
    stop_id: Optional[str] = None

@dataclass
class Modification:
    start_stop_selector: Optional[StopSelector] = None
    end_stop_selector: Optional[StopSelector] = None

@dataclass
class SelectedTrips:
    trip_ids: List[str] = field(default_factory=list)
    shape_id: Optional[str] = None  # peut rester utile en info, mais non tracée

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

# --------------------------------------------------------------------------------
# 3) Polyline — labo (facultatif), pas utilisé par l’analyse principale
# --------------------------------------------------------------------------------
def _sanitize_polyline(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if "\\n" in s or "\\r" in s or "\\t" in s or "\\\\" in s:
        s = s.replace("\\n", "").replace("\\r", "").replace("\\t", "")
        s = s.replace("\\\\", "\\")
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
    try: c5 = pl.decode(enc, precision=5)
    except Exception: c5 = []
    try: c6 = pl.decode(enc, precision=6)
    except Exception: c6 = []
    if _valid_coords(c5) and not _valid_coords(c6): return c5
    if _valid_coords(c6) and not _valid_coords(c5): return c6
    if _valid_coords(c5) and _valid_coords(c6):
        def span(cs):
            lats = [la for la,_ in cs]; lons = [lo for _,lo in cs]
            return (max(lats)-min(lats)) + (max(lons)-min(lons))
        return c6 if span(c6) > span(c5)*1.3 else c5
    return _legacy_decode_polyline(enc)

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

def build_chart_for_polyline(poly: List[Tuple[float, float]], show_index_labels: bool = True):
    if not poly or len(poly) < 2:
        return None
    df_route = pd.DataFrame([{"lat": la, "lon": lo, "idx": i} for i, (la, lo) in enumerate(poly)])
    (lon_min, lon_max), (lat_min, lat_max) = _compute_domain(poly)
    x_scale = alt.Scale(domain=[lon_min, lon_max], zero=False, nice=False)
    y_scale = alt.Scale(domain=[lat_min, lat_max], zero=False, nice=False)

    line = (
        alt.Chart(df_route)
           .mark_line(color="#1f77b4", strokeWidth=3)
           .encode(
               x=alt.X("lon:Q", title="Longitude", scale=x_scale),
               y=alt.Y("lat:Q", title="Latitude", scale=y_scale),
               order=alt.Order("idx:Q")
           )
    )
    points = (
        alt.Chart(df_route)
           .mark_circle(color="red", size=40, opacity=0.9)
           .encode(
               x=alt.X("lon:Q", scale=x_scale),
               y=alt.Y("lat:Q", scale=y_scale),
               tooltip=[alt.Tooltip("idx:Q", title="#"), alt.Tooltip("lat:Q"), alt.Tooltip("lon:Q")]
           )
    )
    labels = None
    if show_index_labels and len(df_route) <= 200:
        labels = (
            alt.Chart(df_route)
               .mark_text(align="left", dx=4, dy=-4, fontSize=10, color="red")
               .encode(x="lon:Q", y="lat:Q", text="idx:Q")
        )
    chart = (line + points if labels is None else line + points + labels).properties(
        width="container", height=360, title="Labo polyline (hors analyse TripMods)"
    ).interactive()
    return chart

# --------------------------------------------------------------------------------
# 4) Parsing TripMods — JSON / PB / TEXTPROTO (shapes ignorés)
# --------------------------------------------------------------------------------
def _detect_tripmods_format_bytes(b: bytes) -> str:
    head = (b[:4096] or b'')
    hs = head.lstrip()
    if hs.startswith(b'{') or hs.startswith(b'['):
        return 'json'
    try:
        txt = head.decode('utf-8', 'ignore')
    except Exception:
        return 'pb'
    if any(s in txt for s in ('entity', 'trip_modifications', 'shape', 'shape_id')):
        # On traitera "textproto" mais on ignorera les blocs 'shape'
        return 'textproto'
    return 'pb'

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

def _coerce_selected_trips(obj: Dict[str, Any]) -> SelectedTrips:
    trips = obj.get('trip_ids') or []
    if isinstance(trips, str): trips = [trips]
    trips = [str(t).strip() for t in trips if str(t).strip()]
    shape_id = obj.get('shape_id')
    shape_id = str(shape_id) if shape_id not in (None, '') else None
    return SelectedTrips(trip_ids=trips, shape_id=shape_id)

# ---- JSON ----
def parse_tripmods_json(feed: Dict[str, Any]) -> List[TripModEntity]:
    entities = feed.get('entity') or []
    out: List[TripModEntity] = []
    for e in entities:
        tm = e.get('trip_modifications')
        if not tm:
            continue  # ignore tout sauf trip_modifications
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
            start_sel = _coerce_selector(m.get('start_stop_selector') or {})
            end_sel   = _coerce_selector(m.get('end_stop_selector') or {})
            mods.append(Modification(start_stop_selector=start_sel, end_stop_selector=end_sel))
        out.append(TripModEntity(
            entity_id=str(e.get('id') or ''),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=mods
        ))
    return out

# ---- Protobuf binaire ----
def parse_tripmods_protobuf(data: bytes) -> List[TripModEntity]:
    proto = gtfs_local
    if proto is None:
        from google.transit import gtfs_realtime_pb2 as proto
    feed = proto.FeedMessage(); feed.ParseFromString(data)
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
            start_sel = StopSelector(
                stop_sequence=getattr(m.start_stop_selector, 'stop_sequence', None) if hasattr(m, 'start_stop_selector') else None,
                stop_id=getattr(m.start_stop_selector, 'stop_id', None) if hasattr(m, 'start_stop_selector') else None,
            )
            end_sel = StopSelector(
                stop_sequence=getattr(m.end_stop_selector, 'stop_sequence', None) if hasattr(m, 'end_stop_selector') else None,
                stop_id=getattr(m.end_stop_selector, 'stop_id', None) if hasattr(m, 'end_stop_selector') else None,
            )
            mods.append(Modification(start_stop_selector=start_sel, end_stop_selector=end_sel))
        out.append(TripModEntity(
            entity_id=str(getattr(ent, 'id', '')),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=mods
        ))
    return out

# ---- TEXTPROTO (dump ASCII) — shapes ignorés
def _lines(b: bytes):
    for raw in b.decode('utf-8', 'ignore').splitlines():
        yield raw.strip()

def parse_textproto_tripmods_only(b: bytes) -> List[TripModEntity]:
    ents: List[TripModEntity] = []
    cur_id: Optional[str] = None
    in_tm = False
    tm_buf: Dict[str, Any] = {}

    def _flush_tm():
        nonlocal tm_buf, cur_id
        if not tm_buf:
            return
        selected: List[SelectedTrips] = tm_buf.get('selected_trips', [])
        service_dates = tm_buf.get('service_dates', [])
        start_times = tm_buf.get('start_times', [])
        modifications = tm_buf.get('modifications', [])
        ents.append(TripModEntity(
            entity_id=str(cur_id or ''),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=modifications
        ))
        tm_buf = {}

    def _start_new_entity(new_id: Optional[str]):
        nonlocal in_tm, cur_id
        if in_tm:
            _flush_tm()
        in_tm = False
        cur_id = new_id

    for line in _lines(b):
        if not line:
            continue
        if line.startswith('entity') or line.startswith('id '):
            # nouvelle entité
            new_id = None
            if line.startswith('id '):
                new_id = line[3:].strip()
            else:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == 'id':
                    new_id = parts[2]
            _start_new_entity(new_id)
            continue

        if line.startswith('trip_modifications'):
            in_tm = True
            tm_buf = dict(selected_trips=[], service_dates=[], start_times=[], modifications=[])
            continue
        if line.startswith('shape'):
            # on ignore totalement les blocs shape
            in_tm = False
            tm_buf = {}
            continue

        if in_tm:
            if line.startswith('selected_trips'):
                continue
            if line.startswith('trip_ids '):
                ids = line[len('trip_ids '):].strip().split()
                if not tm_buf['selected_trips'] or tm_buf['selected_trips'][-1].trip_ids:
                    tm_buf['selected_trips'].append(SelectedTrips())
                tm_buf['selected_trips'][-1].trip_ids = ids
                continue
            if line.startswith('shape_id '):
                sid = line[len('shape_id '):].strip()
                if not tm_buf['selected_trips']:
                    tm_buf['selected_trips'].append(SelectedTrips())
                tm_buf['selected_trips'][-1].shape_id = sid
                continue
            if line.startswith('service_dates '):
                tm_buf['service_dates'] += line[len('service_dates '):].strip().split()
                continue
            if line.startswith('start_times '):
                tm_buf['start_times'] += line[len('start_times '):].strip().split()
                continue
            if line.startswith('modifications'):
                continue
            if line.startswith('start_stop_selector'):
                tm_buf['modifications'].append(Modification(
                    start_stop_selector=StopSelector(), end_stop_selector=StopSelector()
                ))
                continue
            if line.startswith('end_stop_selector'):
                if not tm_buf['modifications']:
                    tm_buf['modifications'].append(Modification(
                        start_stop_selector=StopSelector(), end_stop_selector=StopSelector()
                    ))
                continue
            if line.startswith('stop_sequence '):
                val = line[len('stop_sequence '):].strip()
                try: seq = int(val)
                except: seq = None
                if tm_buf['modifications']:
                    m = tm_buf['modifications'][-1]
                    if m.start_stop_selector and m.start_stop_selector.stop_sequence is None:
                        m.start_stop_selector.stop_sequence = seq
                    else:
                        if m.end_stop_selector is None: m.end_stop_selector = StopSelector()
                        m.end_stop_selector.stop_sequence = seq
                continue
            # toute autre ligne en TM ignorée
            continue

    if in_tm:
        _flush_tm()
    return ents

def load_tripmods_bytes(file_bytes: bytes):
    fmt = _detect_tripmods_format_bytes(file_bytes)
    if fmt == 'json':
        raw = json.loads(file_bytes.decode('utf-8'))
        feed = _normalize_json_keys(raw)
        ents = parse_tripmods_json(feed)
        return ents, feed
    if fmt == 'textproto':
        ents = parse_textproto_tripmods_only(file_bytes)
        return ents, None
    # PB binaire
    ents = parse_tripmods_protobuf(file_bytes)
    return ents, None

# --------------------------------------------------------------------------------
# 5) Ensembles cibles (GTFS filtré) — uniquement trip_ids (pas de stops)
# --------------------------------------------------------------------------------
def compute_needed_trip_ids(ents: List[TripModEntity]) -> Set[str]:
    needed_trip_ids: Set[str] = set()
    for e in ents:
        for s in e.selected_trips:
            needed_trip_ids.update([str(tid) for tid in s.trip_ids if tid])
    return needed_trip_ids

# --------------------------------------------------------------------------------
# 6) Chargement GTFS FILTRÉ
# --------------------------------------------------------------------------------
def load_gtfs_zip_filtered_bytes(zip_bytes: bytes, needed_trip_ids: Set[str]) -> GtfsStatic:
    trips: Dict[str, Dict[str, str]] = {}
    stop_times: Dict[str, List[Dict[str, str]]] = {}
    if not needed_trip_ids:
        return GtfsStatic(trips=trips, stop_times=stop_times)
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
    return GtfsStatic(trips=trips, stop_times=stop_times)

# --------------------------------------------------------------------------------
# 7) Analyse (sans ReplacementStop ni shapes)
# --------------------------------------------------------------------------------
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
                  missing_trip_ids=0, invalid_selectors=0)
    for e in ents:
        trip_checks: List[TripCheck] = []
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
                    # on teste chaque modification (si plusieurs, on retient la 1ère séquence résolue)
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
        totals["total_trip_ids"] += tot_trip_ids
        totals["total_modifications"] += len(e.modifications)
        reports.append(EntityReport(
            entity_id=e.entity_id,
            total_selected_trip_ids=tot_trip_ids,
            service_dates=e.service_dates,
            modification_count=len(e.modifications),
            trips=trip_checks
        ))
    return reports, totals

# --------------------------------------------------------------------------------
# 8) UI Streamlit
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Analyse TripModifications — TripMods only", layout="wide")
st.title("Analyse TripModifications (JSON/PB/Textproto) vs GTFS — TripMods uniquement")
st.caption("Seules les entités contenant **trip_modifications** sont traitées. "
           "**Aucune** analyse des entités shape, ni des ReplacementStop.")

with st.sidebar:
    st.header("Données d’entrée")
    tripmods_file = st.file_uploader("TripModifications (JSON / PB / textproto)", type=["json", "pb", "pbf", "bin", "txt"])
    gtfs_file = st.file_uploader("GTFS (.zip)", type=["zip"])
    # Labo polyline (hors analyse)
    st.markdown("---")
    st.subheader("🧪 Labo polyline (optionnel)")
    s = st.text_area("Colle une encoded_polyline (pour visualiser, hors analyse TripMods).", height=100)
    decode_mode = st.selectbox("Décodage (labo)", ["Auto", "1e-5", "1e-6"], index=0)
    if st.button("Tracer la polyline (labo)"):
        mode = {"Auto": "auto", "1e-5": "p5", "1e-6": "p6"}[decode_mode]
        pts = decode_polyline(s, mode=mode)
        st.write(f"Points décodés : {len(pts)}")
        chart = build_chart_for_polyline(pts, show_index_labels=True)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        if pts:
            st.dataframe(pd.DataFrame(pts, columns=["lat", "lon"]).head(50), width="stretch")

    st.markdown("---")
    dump_first = st.checkbox("Afficher le 1er trip_mod normalisé", value=False)
    run_btn = st.button("Analyser", type="primary")

if run_btn:
    if not tripmods_file or not gtfs_file:
        st.error("Merci de sélectionner **TripModifications** et **GTFS** (.zip).")
        st.stop()

    # 1) TripMods → parsing (shapes ignorés)
    with st.spinner("Parsing TripModifications…"):
        try:
            ents, feed_json = load_tripmods_bytes(tripmods_file.getvalue())
        except Exception as ex:
            st.exception(ex); st.stop()

    needed_trip_ids = compute_needed_trip_ids(ents)

    # 2) GTFS filtré
    with st.spinner("Chargement GTFS (filtré)…"):
        try:
            gtfs = load_gtfs_zip_filtered_bytes(gtfs_file.getvalue(), needed_trip_ids)
        except Exception as ex:
            st.exception(ex); st.stop()

    st.success(f"TripModifications : **{len(ents)} entités**")
    st.info(f"Filtrage GTFS → trips requis: {len(needed_trip_ids):,}")

    # 3) Analyse
    with st.spinner("Analyse en cours…"):
        reports, totals = analyze_tripmods_with_gtfs(gtfs, ents)

    # KPIs (sans ReplacementStop ni shapes)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Entités", totals["total_entities"])
    c2.metric("trip_ids sélectionnés", totals["total_trip_ids"])
    c3.metric("modifications", totals["total_modifications"])
    c4.metric("trip_ids manquants", totals["missing_trip_ids"])
    c5.metric("selectors non résolus", totals["invalid_selectors"])

    # Synthèse par entité
    table = [{
        "entity_id": r.entity_id,
        "trip_ids (sélectionnés)": r.total_selected_trip_ids,
        "modifications": r.modification_count,
        "service_dates": ", ".join(r.service_dates),
    } for r in reports]
    st.subheader("Synthèse par entité")
    st.dataframe(table, width="stretch", height=360)

    if dump_first and ents:
        with st.expander("Aperçu du 1er trip_mod (normalisé)"):
            st.json(asdict(ents[0]))
    if feed_json is not None:
        with st.expander("Aperçu brut du feed JSON (après normalisation camel→snake)"):
            st.json(feed_json)

    # 4) Détails (par entité) — colonnes d'anomalies "TripMods only"
    st.subheader("Détails")
    for r in reports[:200]:
        with st.expander(f"Entité {r.entity_id} — {r.total_selected_trip_ids} trips — {r.modification_count} modifications"):
            st.write("**Dates** :", ", ".join(r.service_dates) if r.service_dates else "—")

            # Prépare le tableau détail
            # NB: pas de 'shape dispo / pts polyline / replacement_stops', etc.
            #     on garde 'shape_id (trip)' comme info mais sans analyse de la forme
            ent_obj = next((e for e in ents if e.entity_id == r.entity_id), None)
            shape_ids_in_entity = []
            if ent_obj:
                for s in ent_obj.selected_trips:
                    if s.shape_id:
                        shape_ids_in_entity.append(s.shape_id)
            shape_ids_set = set(shape_ids_in_entity)
            mixed_shapes = len([sid for sid in shape_ids_set if sid]) > 1

            # duplications de trip dans l'entité
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
                    "shape_id (trip)": trip_shape_id or "",
                    "start_seq": t.start_seq if t.start_seq is not None else "",
                    "end_seq": t.end_seq if t.end_seq is not None else "",
                    "selectors OK": "oui" if (t.start_seq_valid and t.end_seq_valid) else "non",
                    "stop_times (nb)": stop_times_count,
                    "ordre start<=end": ordre_ok,
                    "écart seq": ecart_seq,
                    "selectors incomplets": "oui" if selectors_incomplets else "non",
                    "trip en double (entité)": "oui" if duplicate_trip else "non",
                    "mixed shapes (entité)": "oui" if mixed_shapes else "non",
                    "notes": "; ".join(t.notes) if t.notes else "",
                })

            st.dataframe(detail_rows, width="stretch", height=280)

    # Export JSON (résultats)
    report_json = {"totals": totals, "entities": [asdict(r) for r in reports]}
    st.download_button("📥 Télécharger le rapport JSON",
                       data=json.dumps(report_json, ensure_ascii=False, indent=2),
                       file_name="rapport_tripmods.json", mime="application/json")

else:
    st.info("Charge un TripModifications (JSON / PB / textproto) puis un GTFS (.zip), et clique **Analyser**.")
    st.caption("Seules les entités 'trip_modifications' sont prises en compte ; pas d'analyse des shapes ni des ReplacementStop.")
