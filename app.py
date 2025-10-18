# --- DOIT ÊTRE EN PREMIER ---
import streamlit as st
st.set_page_config(
    page_title="Analyse TripModifications + GTFS — JSON/PB/Textproto + carte",
    layout="wide"
)
# ----------------------------

from __future__ import annotations

import json, csv, io, zipfile, sys, re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Tuple, Set
from pathlib import Path
import pandas as pd

# Carte
import folium
from streamlit_folium import st_folium

# 0) Import protobuf local si dispo (gtfs_realtime_pb2.py)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import gtfs_realtime_pb2 as gtfs_local
except Exception:
    gtfs_local = None

# 1) camelCase → snake_case
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

# 2) Modèles
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
    stops_info: Dict[str, Dict[str, Any]]   # stop_id -> {"lat": float, "lon": float, "name": str}

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

# 3) Décodage polyline (nettoyage SANS unicode_escape) + Auto/5/6 + fallback
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

# 4) Parsing TripMods & Shapes — JSON / PB / TEXTPROTO
def _detect_tripmods_format_bytes(b: bytes) -> str:
    head = (b[:4096] or b'')
    hs = head.lstrip()
    if hs.startswith(b'{') or hs.startswith(b'['):   # <-- corrigé ici
        return 'json'
    try:
        txt = head.decode('utf-8', 'ignore')
    except Exception:
        return 'pb'
    if any(s in txt for s in ('entity', 'trip_modifications', 'shape', 'encoded_polyline', 'shape_id')):
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

# -- JSON --
def parse_tripmods_json(feed: Dict[str, Any]) -> List[TripModEntity]:
    entities = feed.get('entity') or []
    out: List[TripModEntity] = []
    for e in entities:
        tm = e.get('trip_modifications')
        if not tm:
            continue
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

# -- Protobuf binaire --
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

# -- TEXTPROTO (dump ASCII) --
def _lines(b: bytes):
    for raw in b.decode('utf-8', 'ignore').splitlines():
        yield raw.strip()

def parse_textproto_feed(b: bytes, decode_mode: str) -> Tuple[List[TripModEntity], RtShapes]:
    ents: List[TripModEntity] = []
    shapes: Dict[str, List[Tuple[float, float]]] = {}

    cur_id: Optional[str] = None
    in_tm = False
    in_shape = False
    capturing_poly = False
    tm_buf: Dict[str, Any] = {}
    shape_buf: Dict[str, Any] = {}

    def _flush_tm():
        nonlocal tm_buf, cur_id
        if not tm_buf:
            return
        selected: List[SelectedTrips] = tm_buf.get('selected_trips', [])
        service_dates = tm_buf.get('service_dates', [])
        start_times = tm_buf.get('start_times', [])
        modifications = tm_buf.get('modifications', [])
        ents.append(TripModEntity(
            entity_id=str(cur_id or ""),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=modifications
        ))
        tm_buf = {}

    def _flush_shape():
        nonlocal shape_buf
        if shape_buf.get('shape_id') and shape_buf.get('encoded_polyline'):
            sid = str(shape_buf['shape_id'])
            enc = str(shape_buf['encoded_polyline'])
            try:
                shapes[sid] = decode_polyline(enc, mode=decode_mode)
            except Exception:
                pass
        shape_buf = {}

    def _start_new_entity(new_id: Optional[str]):
        nonlocal in_tm, in_shape, capturing_poly, cur_id
        if in_tm: _flush_tm()
        if in_shape: _flush_shape()
        in_tm = False
        in_shape = False
        capturing_poly = False
        cur_id = new_id

    for line in _lines(b):
        if not line:
            continue
        if capturing_poly:
            if (line.startswith('id ') or line.startswith('entity')
                or line.startswith('shape_id ') or line.startswith('trip_modifications')
                or line.startswith('shape ') or line.startswith('selected_trips')
                or line.startswith('service_dates ') or line.startswith('start_times ')
                or line.startswith('modifications ') or line.startswith('start_stop_selector')
                or line.startswith('end_stop_selector') or line.startswith('replacement_stops')
                or line.startswith('encoded_polyline ')):
                capturing_poly = False
            else:
                shape_buf['encoded_polyline'] = shape_buf.get('encoded_polyline', '') + line
                continue

        if line.startswith('entity') or line.startswith('id '):
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
            in_tm, in_shape, capturing_poly = True, False, False
            tm_buf = dict(selected_trips=[], service_dates=[], start_times=[], modifications=[])
            continue

        if line.startswith('shape'):
            in_tm, in_shape, capturing_poly = False, True, False
            shape_buf = {}
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
                    start_stop_selector=StopSelector(), end_stop_selector=StopSelector(),
                    replacement_stops=[]
                ))
                continue
            if line.startswith('end_stop_selector'):
                if not tm_buf['modifications']:
                    tm_buf['modifications'].append(Modification(
                        start_stop_selector=StopSelector(), end_stop_selector=StopSelector(),
                        replacement_stops=[]
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
            if line.startswith('replacement_stops'):
                if not tm_buf['modifications']:
                    tm_buf['modifications'].append(Modification(
                        start_stop_selector=StopSelector(), end_stop_selector=StopSelector(),
                        replacement_stops=[]
                    ))
                continue
            if line.startswith('stop_id '):
                sid = line[len('stop_id '):].strip()
                if tm_buf['modifications']:
                    tm_buf['modifications'][-1].replacement_stops.append(
                        ReplacementStop(stop_id=sid, travel_time_to_stop=0)
                    )
                continue
            continue

        if in_shape:
            if line.startswith('shape_id '):
                shape_buf['shape_id'] = line[len('shape_id '):].strip()
                continue
            if line.startswith('encoded_polyline '):
                enc = line[len('encoded_polyline '):]
                shape_buf['encoded_polyline'] = enc
                capturing_poly = True
                continue
            continue

    if in_tm: _flush_tm()
    if in_shape: _flush_shape()
    return ents, RtShapes(shapes=shapes)

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
    if fmt == 'textproto':
        ents, shapes = parse_textproto_feed(file_bytes, decode_mode=decode_mode)
        needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
        if needed_shape_ids:
            shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
        return ents, None, shapes
    # PB binaire
    ents = parse_tripmods_protobuf(file_bytes)
    shapes = _collect_shapes_pb(file_bytes, decode_mode)
    needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
    if needed_shape_ids:
        shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
    return ents, None, shapes

# 5) Ensembles cibles (GTFS filtré)
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

# 6) Chargement GTFS FILTRÉ (+ coordonnées des stops nécessaires)
def load_gtfs_zip_filtered_bytes(zip_bytes: bytes, needed_trip_ids: Set[str], needed_stop_ids: Set[str]) -> GtfsStatic:
    trips: Dict[str, Dict[str, str]] = {}
    stop_times: Dict[str, List[Dict[str, str]]] = {}
    stops_present: Set[str] = set()
    stops_info: Dict[str, Dict[str, Any]] = {}

    if not needed_trip_ids and not needed_stop_ids:
        return GtfsStatic(trips=trips, stop_times=stop_times, stops_present=stops_present, stops_info=stops_info)

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
                        try:
                            lat = float((row.get('stop_lat') or '').strip())
                            lon = float((row.get('stop_lon') or '').strip())
                        except Exception:
                            lat = lon = None
                        name = (row.get('stop_name') or '').strip()
                        if lat is not None and lon is not None:
                            stops_info[sid] = {"lat": lat, "lon": lon, "name": name}
                        stops_present.add(sid)

    return GtfsStatic(trips=trips, stop_times=stop_times, stops_present=stops_present, stops_info=stops_info)

# 7) Analyse
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
                        if end_seq is None and eseq is not None: end_seq = eseq
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

# 8) Visualisation Folium (fond Montréal + arrêts temporaires orange)
def build_folium_map_for_polyline(
    poly: List[Tuple[float, float]],
    shape_id: Optional[str] = None,
    replacement_stop_points: Optional[List[Tuple[float, float, str]]] = None,
    montreal_center: Tuple[float, float] = (45.5017, -73.5673),
    zoom_start: int = 12
):
    """
    Construit une carte Folium centrée sur Montréal, trace la polyline,
    et ajoute en ORANGE les arrêts temporaires du détour (si fournis).
    - poly : liste de (lat, lon)
    - replacement_stop_points : liste de (lat, lon, label)
    """
    if not poly or len(poly) < 2:
        return None

    m = folium.Map(location=montreal_center, zoom_start=zoom_start,
                   tiles="OpenStreetMap", control_scale=True)

    latlons = [(la, lo) for la, lo in poly]
    folium.PolyLine(
        locations=latlons,
        color="#1f77b4",
        weight=5,
        opacity=0.9,
        tooltip=f"shape_id: {shape_id or 'n/a'}",
    ).add_to(m)

    # Départ / Arrivée
    folium.CircleMarker(latlons[0], radius=6, color="green",
                        fill=True, fill_opacity=0.9, tooltip="Départ").add_to(m)
    folium.CircleMarker(latlons[-1], radius=6, color="red",
                        fill=True, fill_opacity=0.9, tooltip="Arrivée").add_to(m)

    # Arrêts temporaires (orange)
    bounds_pts = latlons.copy()
    if replacement_stop_points:
        for la, lo, lab in replacement_stop_points:
            folium.CircleMarker(
                location=(la, lo),
                radius=7,
                color="orange",
                fill=True,
                fill_opacity=0.95,
                tooltip=lab or "Arrêt temporaire"
            ).add_to(m)
            bounds_pts.append((la, lo))

    # Ajuste la vue sur l’emprise (polyline + arrêts temporaires éventuels)
    if bounds_pts:
        m.fit_bounds(bounds_pts)

    return m

# 9) CACHE des calculs lourds
@st.cache_data(show_spinner=False)
def run_analysis_cached(tripmods_bytes: bytes, gtfs_bytes: bytes, decode_flag: str):
    ents, feed_json, rt_shapes = load_tripmods_bytes(tripmods_bytes, decode_mode=decode_flag)
    needed_trip_ids, needed_stop_ids = compute_needed_sets(ents)
    gtfs = load_gtfs_zip_filtered_bytes(gtfs_bytes, needed_trip_ids, needed_stop_ids)
    reports, totals = analyze_tripmods_with_gtfs(gtfs, ents)
    total_shapes = len(rt_shapes.shapes)
    return dict(
        ents=ents, feed_json=feed_json, rt_shapes=rt_shapes,
        gtfs=gtfs, reports=reports, totals=totals, total_shapes=total_shapes,
        needed_trip_ids=needed_trip_ids, needed_stop_ids=needed_stop_ids
    )

# 10) UI — formulaire + session_state (stable)
st.title("Analyse TripModifications (JSON/PB/Textproto) vs GTFS — Carte Folium")
st.caption(
    "Polylines nettoyées (déséchappage léger), décodées (Auto/1e‑5/1e‑6) et affichées sur une carte Folium (fond OSM). "
    "Support du format textproto (dump ASCII GTFS‑RT). L’analyse est stable (formulaire + cache + session)."
)

with st.sidebar:
    st.header("Données d’entrée")
    with st.form("inputs_form"):
        tripmods_file = st.file_uploader("TripModifications (JSON / PB / textproto)", type=["json", "pb", "pbf", "bin", "txt"], key="tripmods_up")
        gtfs_file = st.file_uploader("GTFS (.zip)", type=["zip"], key="gtfs_up")
        decode_mode = st.selectbox("Décodage polylines", ["Auto (recommandé)", "Précision 1e-5", "Précision 1e-6"], index=0, key="decode_sel")
        dump_first = st.checkbox("Afficher le 1er trip_mod normalisé", value=False, key="dump_first_cb")
        submitted = st.form_submit_button("Analyser", type="primary")

decode_flag = {"Auto (recommandé)": "auto", "Précision 1e-5": "p5", "Précision 1e-6": "p6"}[st.session_state.get("decode_sel", "Auto (recommandé)")]

if submitted:
    if not tripmods_file or not gtfs_file:
        st.error("Merci de sélectionner **TripModifications** (JSON/PB/textproto) **et** **GTFS** (.zip).")
    else:
        tripmods_bytes = tripmods_file.getvalue()
        gtfs_bytes = gtfs_file.getvalue()
        res = run_analysis_cached(tripmods_bytes, gtfs_bytes, decode_flag)
        st.session_state["last_results"] = res
        st.session_state["last_params"] = dict(decode_flag=decode_flag)
        st.success("Analyse terminée ✅")

res = st.session_state.get("last_results")

if not res:
    st.info("Charge un TripModifications (JSON / PB / textproto) puis un GTFS (.zip), choisis la précision de décodage, et clique **Analyser**.")
    st.caption("Astuce : si ta polyligne vient d’un JSON, elle doit être « déséchappée » (retrait des \\ et \\n).")
    st.stop()

ents = res["ents"]; feed_json = res["feed_json"]; rt_shapes = res["rt_shapes"]
gtfs = res["gtfs"]; reports = res["reports"]; totals = res["totals"]; total_shapes = res["total_shapes"]
needed_trip_ids = res["needed_trip_ids"]; needed_stop_ids = res["needed_stop_ids"]

# KPIs
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Entités", totals["total_entities"])
c2.metric("trip_ids sélectionnés", totals["total_trip_ids"])
c3.metric("modifications", totals["total_modifications"])
c4.metric("trip_ids manquants", totals["missing_trip_ids"])
c5.metric("selectors non résolus", totals["invalid_selectors"])
c6.metric("repl. stops inconnus GTFS", totals["unknown_replacement_stops"])
c7.metric("Shapes dans le feed", total_shapes)

# Infos filtrage et comptages
st.info(f"Filtrage GTFS → trips requis: {len(needed_trip_ids):,} · stops requis: {len(needed_stop_ids):,} · shapes disponibles: {total_shapes:,}")
st.success(f"GTFS filtré : **{len(gtfs.trips):,} trips conservés**, **{sum(len(v) for v in gtfs.stop_times.values()):,} stop_times**, **{len(gtfs.stops_present):,} stops présents**")
st.success(f"TripModifications : **{len(ents)} entités**")

# Aperçus optionnels
if st.session_state.get("dump_first_cb") and ents:
    with st.expander("Aperçu du 1er trip_mod (normalisé)"):
        st.json(asdict(ents[0]))
if feed_json is not None:
    with st.expander("Aperçu brut du feed JSON (après normalisation camel→snake)"):
        st.json(feed_json)

# Synthèse par entité
table = [{
    "entity_id": r.entity_id,
    "trip_ids (sélectionnés)": r.total_selected_trip_ids,
    "modifications": r.modification_count,
    "service_dates": ", ".join(r.service_dates),
    "repl_stops inconnus": ", ".join(r.replacement_stops_unknown_in_gtfs) if r.replacement_stops_unknown_in_gtfs else ""
} for r in reports]
st.subheader("Synthèse par entité")
st.dataframe(table, width="stretch", height=360)

# --- Détails + tracé par entité (AFFICHER UNIQUEMENT si trip_modifications présent) ---
st.subheader("Détails")
for r in reports[:200]:
    ent_obj = next((e for e in ents if e.entity_id == r.entity_id), None)
    if not ent_obj or not ent_obj.modifications:
        continue

    with st.expander(f"Entité {r.entity_id} — {r.total_selected_trip_ids} trips — {r.modification_count} modifications"):
        st.write("**Dates** :", ", ".join(r.service_dates) if r.service_dates else "—")
        st.write("**Replacement stops inconnus dans GTFS (peuvent être temporaires)** :",
                 ", ".join(r.replacement_stops_unknown_in_gtfs) if r.replacement_stops_unknown_in_gtfs else "—")

        # Liste des shape_id déclarés côté selected_trips
        shape_ids_in_entity = [s.shape_id for s in ent_obj.selected_trips if s.shape_id]
        shape_id_for_plot = next((sid for sid in shape_ids_in_entity if sid in rt_shapes.shapes), None)
        st.write(f"**Shape utilisé pour le tracé** : {shape_id_for_plot or '—'}")

        # Construit la liste des arrêts temporaires (remplacement_stops) avec coordonnées (si dispo)
        tmp_stop_points: List[Tuple[float, float, str]] = []
        tmp_stop_ids_seen: Set[str] = set()
        for m in ent_obj.modifications:
            for rs in m.replacement_stops:
                sid = rs.stop_id
                if not sid or sid in tmp_stop_ids_seen:
                    continue
                info = gtfs.stops_info.get(sid)
                if info and "lat" in info and "lon" in info:
                    label = f"{sid} — {info.get('name') or ''}".strip(" —")
                    tmp_stop_points.append((info["lat"], info["lon"], label))
                    tmp_stop_ids_seen.add(sid)

        # Tracé carte (Folium)
        if shape_id_for_plot:
            poly = rt_shapes.shapes.get(shape_id_for_plot, [])
            if poly and len(poly) >= 2:
                fmap = build_folium_map_for_polyline(
                    poly,
                    shape_id=shape_id_for_plot,
                    replacement_stop_points=tmp_stop_points
                )
                if fmap is not None:
                    # PATCH 2 : clé unique par entité
                    st_folium(fmap, height=440, width=None, key=f"map_{r.entity_id}")
            else:
                st.info("Polyline vide ou invalide pour cette entité.")
        else:
            st.info("Aucune polyline 'encoded_polyline' disponible pour cette entité.")

# Export JSON
report_json = {"totals": totals, "total_shapes": total_shapes, "entities": [asdict(r) for r in reports]}
st.download_button("📥 Télécharger le rapport JSON",
                   data=json.dumps(report_json, ensure_ascii=False, indent=2),
                   file_name="rapport_tripmods.json", mime="application/json")
