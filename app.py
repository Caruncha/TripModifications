from __future__ import annotations

import streamlit as st
st.set_page_config(
    page_title="Analyse TripModifications + GTFS — JSON/PB/Textproto + carte",
    layout="wide"
)

import json, csv, io, zipfile, sys, re, hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Tuple, Set
from pathlib import Path

import pandas as pd
import folium
import streamlit.components.v1 as components

# --- Version de schéma (invalide cache data si on change la structure de sortie) ---
# Bump pour forcer le rafraîchissement des cartes (passage des stops en rose)
SCHEMA_VERSION = "2025-10-18-resource-data-split-v3-pink-stops"

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
    trips: Dict[str, Dict[str, str]] = field(default_factory=dict)
    stop_times: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    stops_present: Set[str] = field(default_factory=set)
    # stop_id -> {"lat": float, "lon": float, "name": str}
    stops_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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

# 3) Décodage polyline
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
    if hs.startswith(b'{') or hs.startswith(b'['):
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

# 8) Folium — centrage STRICT sur le tracé + replacement stops en ROSE (sans impact emprise)
def build_folium_map_for_polyline(
    poly: List[Tuple[float, float]],
    shape_id: Optional[str] = None,
    replacement_stop_points: Optional[List[Tuple[float, float, str]]] = None,
    montreal_center: Tuple[float, float] = (45.5017, -73.5673),
    zoom_start: int = 12
):
    def _valid_ll(la, lo) -> bool:
        return (la is not None and lo is not None
                and -90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0
                and not (abs(la) < 1e-8 and abs(lo) < 1e-8))  # évite (0,0)

    if not poly or len(poly) < 2:
        return None
    latlons_poly = [(la, lo) for la, lo in poly if _valid_ll(la, lo)]
    if len(latlons_poly) < 2:
        return None

    m = folium.Map(
        location=montreal_center,  # sera remplacé par fit_bounds
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
        min_zoom=8
    )

    folium.PolyLine(
        locations=latlons_poly,
        color="#1f77b4",
        weight=5,
        opacity=0.9,
        tooltip=f"shape_id: {shape_id or 'n/a'}",
    ).add_to(m)
    folium.CircleMarker(latlons_poly[0], radius=6, color="green",
                        fill=True, fill_opacity=0.9, tooltip="Départ").add_to(m)
    folium.CircleMarker(latlons_poly[-1], radius=6, color="red",
                        fill=True, fill_opacity=0.9, tooltip="Arrivée").add_to(m)

    # Replacement stops (ROSE) visibles mais sans impact sur l’emprise
    if replacement_stop_points:
        for la, lo, lab in replacement_stop_points:
            if _valid_ll(la, lo):
                folium.CircleMarker(
                    location=(la, lo),
                    radius=7,
                    color="#ff69b4",          # contour rose
                    fill=True,
                    fill_color="#ff69b4",     # remplissage rose
                    fill_opacity=0.95,
                    weight=2,
                    tooltip=lab or "Arrêt temporaire (remplacement)"
                ).add_to(m)

    # Emprise STRICTE sur le tracé (min/max + padding)
    lats = [la for la, _ in latlons_poly]
    lons = [lo for _, lo in latlons_poly]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    span_lat = max(max_lat - min_lat, 1e-5)
    span_lon = max(max_lon - min_lon, 1e-5)
    pad_lat = max(span_lat * 0.15, 0.002)
    pad_lon = max(span_lon * 0.15, 0.002)

    sw = (min_lat - pad_lat, min_lon - pad_lon)
    ne = (max_lat + pad_lat, max_lon + pad_lon)
    m.fit_bounds([sw, ne], padding=(30, 30))

    return m

# 9) CACHE — split resource/data + carte HTML en cache resource
def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

@st.cache_resource(show_spinner=True)
def resource_parse_tripmods(tripmods_bytes: bytes, decode_flag: str):
    ents, feed_json, rt_shapes = load_tripmods_bytes(tripmods_bytes, decode_mode=decode_flag)
    return ents, feed_json, rt_shapes

@st.cache_resource(show_spinner=True)
def resource_load_gtfs(gtfs_bytes: bytes, needed_trip_ids_sorted: Tuple[str, ...], needed_stop_ids_sorted: Tuple[str, ...]):
    gtfs = load_gtfs_zip_filtered_bytes(gtfs_bytes, set(needed_trip_ids_sorted), set(needed_stop_ids_sorted))
    return gtfs

@st.cache_resource(show_spinner=False)
def resource_build_map_html(
    shape_id_key: str,
    poly_key: Tuple[Tuple[float, float], ...],
    stops_key: Tuple[Tuple[float, float, str], ...],
    schema: str
) -> str:
    """
    Construit la carte Folium et renvoie **l'HTML** (string) mis en cache.
    Les clés (poly_key, stops_key) sont des tuples hashables.
    """
    poly = [(la, lo) for (la, lo) in poly_key]
    stops = [(la, lo, lab) for (la, lo, lab) in stops_key]
    fmap = build_folium_map_for_polyline(poly, shape_id=shape_id_key, replacement_stop_points=stops)
    if fmap is None:
        return "<div>Carte indisponible</div>"
    return fmap.get_root().render()

@st.cache_data(show_spinner=True, ttl=86400, max_entries=32, hash_funcs={bytes: _hash_bytes})
def cache_views(tripmods_bytes: bytes, gtfs_bytes: bytes, decode_flag: str, schema: str) -> Dict[str, Any]:
    # 1) Parse TripMods
    ents, feed_json, rt_shapes = resource_parse_tripmods(tripmods_bytes, decode_flag)
    needed_trip_ids, needed_stop_ids = compute_needed_sets(ents)
    tid_sorted = tuple(sorted(needed_trip_ids))
    sid_sorted = tuple(sorted(needed_stop_ids))

    # 2) GTFS filtré
    gtfs = resource_load_gtfs(gtfs_bytes, tid_sorted, sid_sorted)

    # 3) Analyse (dataclasses → dicts)
    reports, totals = analyze_tripmods_with_gtfs(gtfs, ents)
    reports_plain = [asdict(r) for r in reports]
    total_shapes = len(rt_shapes.shapes)

    # 4) Prépare les vues “Détails” (tout JSON‑compatible)
    details_tables_by_entity: Dict[str, List[Dict[str, Any]]] = {}
    temp_stops_points_by_entity: Dict[str, List[List[Any]]] = {}
    shape_for_plot_by_entity: Dict[str, Optional[str]] = {}

    def _is_poly_anormal(coords: List[Tuple[float,float]]) -> bool:
        if not coords or len(coords) < 2: return True
        lats = [la for la,_ in coords]; lons = [lo for _,lo in coords]
        return (max(lats)-min(lats) + max(lons)-min(lons)) < 1e-4

    stops_info = getattr(gtfs, "stops_info", {}) or {}

    for r in reports:
        ent_id = r.entity_id
        ent_obj = next((e for e in ents if e.entity_id == ent_id), None)
        if not ent_obj or not ent_obj.modifications:
            continue

        shape_ids_in_entity = [s.shape_id for s in ent_obj.selected_trips if s.shape_id]
        shape_id_for_plot = next((sid for sid in shape_ids_in_entity if sid in rt_shapes.shapes), None)
        shape_for_plot_by_entity[ent_id] = shape_id_for_plot

        # Comptage dupliqués
        trip_counts: Dict[str, int] = {}
        for s in ent_obj.selected_trips:
            for tid in s.trip_ids:
                trip_counts[tid] = trip_counts.get(tid, 0) + 1
        mixed_shapes = len({sid for sid in shape_ids_in_entity if sid}) > 1

        # Tableau diagnostics
        detail_rows: List[Dict[str, Any]] = []
        for t in r.trips:
            st_list = gtfs.stop_times.get(t.trip_id, [])
            stop_times_count = len(st_list)
            trip_shape_id = None
            for s in ent_obj.selected_trips:
                if t.trip_id in s.trip_ids and s.shape_id:
                    trip_shape_id = s.shape_id
                    break
            if not trip_shape_id and shape_ids_in_entity:
                trip_shape_id = shape_ids_in_entity[0]
            poly = rt_shapes.shapes.get(trip_shape_id, []) if trip_shape_id else []
            poly_points = len(poly)
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
            })
        details_tables_by_entity[ent_id] = detail_rows

        # replacement stops (liste [lat, lon, label]) — en ROSE sur la carte
        tmp_points: List[List[Any]] = []
        seen: Set[str] = set()
        for m in ent_obj.modifications:
            for rs in m.replacement_stops:
                sid = rs.stop_id
                if not sid or sid in seen:
                    continue
                info = stops_info.get(sid)
                if info and "lat" in info and "lon" in info:
                    tmp_points.append([info["lat"], info["lon"], f"{sid} — {info.get('name') or ''}".strip(" —")])
                    seen.add(sid)
        temp_stops_points_by_entity[ent_id] = tmp_points

    # 5) shapes → JSON‑compatibles (listes)
    shapes_plain: Dict[str, List[List[float]]] = {
        sid: [[la, lo] for (la, lo) in coords] for sid, coords in rt_shapes.shapes.items()
    }

    # KPI GTFS
    gtfs_kpi = dict(
        trips=len(gtfs.trips),
        stop_times=sum(len(v) for v in gtfs.stop_times.values()),
        stops_present=len(gtfs.stops_present),
    )

    return {
        "schema_version": schema,
        "feed_json": feed_json,
        "reports": reports_plain,
        "totals": totals,
        "total_shapes": total_shapes,
        "needed_trip_ids": list(tid_sorted),
        "needed_stop_ids": list(sid_sorted),
        "details_tables_by_entity": details_tables_by_entity,
        "temp_stops_points_by_entity": temp_stops_points_by_entity,
        "shape_for_plot_by_entity": shape_for_plot_by_entity,
        "shapes_plain": shapes_plain,
        "gtfs_kpi": gtfs_kpi,
    }

# 10) UI — formulaire + session_state (stable, pas de rafraîchissement de carte)
st.title("Analyse TripModifications (JSON/PB/Textproto) vs GTFS — Carte Folium")
st.caption(
    "Polylines nettoyées (déséchappage léger), décodées (Auto/1e‑5/1e‑6) et affichées sur une carte Folium (fond OSM). "
    "Analyse et diagnostics pré‑calculés et mis en cache. La carte est centrée strictement sur le tracé et mise en cache HTML. "
    "Les arrêts de remplacement sont affichés en **rose**."
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
        res = cache_views(tripmods_bytes, gtfs_bytes, decode_flag, SCHEMA_VERSION)
        st.session_state["last_results"] = res
        st.session_state["last_params"] = dict(decode_flag=decode_flag, schema_version=SCHEMA_VERSION)
        st.success("Analyse terminée ✅")

# Récupération des résultats (et vérif de version de schéma)
res = st.session_state.get("last_results")
if res and res.get("schema_version") != SCHEMA_VERSION:
    res = None
    st.session_state.pop("last_results", None)

if not res:
    st.info("Charge un TripModifications (JSON / PB / textproto) puis un GTFS (.zip), choisis la précision de décodage, et clique **Analyser**.")
    st.caption("Astuce : si ta polyligne vient d’un JSON, elle doit être « déséchappée » (retrait des \\ et \\n).")
    st.stop()

feed_json = res["feed_json"]
reports_plain = res["reports"]; totals = res["totals"]; total_shapes = res["total_shapes"]
needed_trip_ids = res["needed_trip_ids"]; needed_stop_ids = res["needed_stop_ids"]
details_tables_by_entity = res["details_tables_by_entity"]
temp_stops_points_by_entity = res["temp_stops_points_by_entity"]
shape_for_plot_by_entity = res["shape_for_plot_by_entity"]
shapes_plain = res["shapes_plain"]
gtfs_kpi = res["gtfs_kpi"]

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
st.success(f"GTFS filtré : **{gtfs_kpi['trips']:,} trips conservés**, **{gtfs_kpi['stop_times']:,} stop_times**, **{gtfs_kpi['stops_present']:,} stops présents**")
st.success(f"TripModifications : **{len(reports_plain)} entités**")

# Aperçus optionnels
if st.session_state.get("dump_first_cb") and reports_plain:
    with st.expander("Aperçu du 1er trip_mod (normalisé)"):
        st.json(reports_plain[0])  # déjà dict

if feed_json is not None:
    with st.expander("Aperçu brut du feed JSON (après normalisation camel→snake)"):
        st.json(feed_json)

# Synthèse par entité
table = [{
    "entity_id": r["entity_id"],
    "trip_ids (sélectionnés)": r["total_selected_trip_ids"],
    "modifications": r["modification_count"],
    "service_dates": ", ".join(r["service_dates"]),
    "repl_stops inconnus": ", ".join(r["replacement_stops_unknown_in_gtfs"]) if r["replacement_stops_unknown_in_gtfs"] else ""
} for r in reports_plain]
st.subheader("Synthèse par entité")
st.dataframe(table, width="stretch", height=360)

# --- Détails + tableau + carte par entité (carte HTML en cache resource, sans rafraîchissement) ---
st.subheader("Détails")
for r in reports_plain[:200]:
    ent_id = r["entity_id"]
    if r["modification_count"] <= 0:
        continue

    with st.expander(f"Entité {ent_id} — {r['total_selected_trip_ids']} trips — {r['modification_count']} modifications"):
        st.write("**Dates** :", ", ".join(r["service_dates"]) if r["service_dates"] else "—")
        st.write("**Replacement stops inconnus dans GTFS (peuvent être temporaires)** :",
                 ", ".join(r["replacement_stops_unknown_in_gtfs"]) if r["replacement_stops_unknown_in_gtfs"] else "—")

        # Tableau d'analyse (pré-calculé)
        rows = details_tables_by_entity.get(ent_id, [])
        if rows:
            st.dataframe(rows, width="stretch", height=260)
        else:
            st.info("Aucune ligne de diagnostic pour cette entité.")

        # Carte (shape + replacement stops en ROSE) — HTML en cache
        shape_id_for_plot = shape_for_plot_by_entity.get(ent_id)
        st.write(f"**Shape utilisé pour le tracé** : {shape_id_for_plot or '—'}")
        if shape_id_for_plot:
            coords_list = shapes_plain.get(shape_id_for_plot, [])
            poly = [(float(la), float(lo)) for la, lo in coords_list]
            if poly and len(poly) >= 2:
                # clés hashables et compactes
                poly_key = tuple((round(la, 6), round(lo, 6)) for la, lo in poly)
                tmp_pts = temp_stops_points_by_entity.get(ent_id, [])
                stops_key = tuple((round(p[0], 6), round(p[1], 6), str(p[2])) for p in tmp_pts)
                # HTML de la carte en cache resource
                map_html = resource_build_map_html(shape_id_for_plot or "", poly_key, stops_key, SCHEMA_VERSION)
                components.html(map_html, height=440, scrolling=False)
            else:
                st.info("Polyline vide ou invalide pour cette entité.")
        else:
            st.info("Aucune polyline 'encoded_polyline' disponible pour cette entité.")

# Export JSON (rapports déjà en dict)
export_json = {
    "schema_version": SCHEMA_VERSION,
    "totals": totals,
    "total_shapes": total_shapes,
    "entities": reports_plain,
}
st.download_button("📥 Télécharger le rapport JSON",
                   data=json.dumps(export_json, ensure_ascii=False, indent=2),
                   file_name="rapport_tripmods.json", mime="application/json")
