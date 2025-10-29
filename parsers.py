# parsers.py
from __future__ import annotations
import json, sys, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from models import (
    RtShapes, TripModEntity, SelectedTrips, ReplacementStop, StopSelector,
    Modification
)
from polyline_utils import decode_polyline

# Option: proto local (préféré), sinon google.transit
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import gtfs_realtime_pb2 as gtfs_local  # type: ignore
except Exception:
    gtfs_local = None

# --- Normalisation JSON (camelCase -> snake_case)
import re as _re
_CAMEL_RE = _re.compile(r'(?<!^)(?=[A-Z])')

def _camel_to_snake(name: str) -> str:
    return _CAMEL_RE.sub('_', name).lower()

def normalize_json_keys(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = _camel_to_snake(k) if isinstance(k, str) else k
            out[nk] = normalize_json_keys(v)
        return out
    if isinstance(obj, list):
        return [normalize_json_keys(x) for x in obj]
    return obj

def detect_tripmods_format_bytes(b: bytes) -> str:
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

def _to_float(v) -> Optional[float]:
    try:
        if v is None: return None
        s = str(v).strip()
        if s == "": return None
        return float(s)
    except Exception:
        return None

# --- JSON
def parse_tripmods_json(feed: Dict[str, Any]) -> List[TripModEntity]:
    out: List[TripModEntity] = []
    for e in feed.get('entity') or []:
        tm = e.get('trip_modifications')
        if not tm:
            continue
        selected = []
        for s in tm.get('selected_trips') or []:
            trips = s.get('trip_ids') or []
            if isinstance(trips, str): trips = [trips]
            trips = [str(t).strip() for t in trips if str(t).strip()]
            sid = s.get('shape_id')
            sid = str(sid) if sid not in (None, '') else None
            selected.append(SelectedTrips(trip_ids=trips, shape_id=sid))
        service_dates = tm.get('service_dates') or []
        if not service_dates:
            for s in tm.get('selected_trips') or []:
                dates = s.get('service_dates')
                if dates: service_dates.extend(dates)
        service_dates = [str(d) for d in service_dates]
        start_times = [str(t) for t in tm.get('start_times') or []]
        mods: List[Modification] = []
        for m in tm.get('modifications') or []:
            repl: List[ReplacementStop] = []
            for rs in m.get('replacement_stops') or []:
                sid = rs.get('stop_id')
                rid = rs.get('id')
                la = _to_float(rs.get('stop_lat'))
                lo = _to_float(rs.get('stop_lon'))
                t = rs.get('travel_time_to_stop', 0)
                try: t = int(t)
                except Exception: t = 0
                if (sid is None) and (la is None or lo is None):
                    continue
                repl.append(ReplacementStop(
                    stop_id=str(sid) if sid is not None else None,
                    id=str(rid) if rid not in (None, "") else None,
                    stop_lat=la, stop_lon=lo, travel_time_to_stop=t
                ))
            start_sel = StopSelector(
                stop_sequence=(lambda v: int(v) if (v not in (None, "")) else None)(m.get('start_stop_selector', {}).get('stop_sequence')),
                stop_id=m.get('start_stop_selector', {}).get('stop_id')
            ) if m.get('start_stop_selector') else StopSelector()
            end_sel = StopSelector(
                stop_sequence=(lambda v: int(v) if (v not in (None, "")) else None)(m.get('end_stop_selector', {}).get('stop_sequence')),
                stop_id=m.get('end_stop_selector', {}).get('stop_id')
            ) if m.get('end_stop_selector') else StopSelector()
            mods.append(Modification(start_stop_selector=start_sel, end_stop_selector=end_sel, replacement_stops=repl,
                                     propagated_modification_delay=m.get('propagated_modification_delay')))
        out.append(TripModEntity(
            entity_id=str(e.get('id') or ''),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=mods
        ))
    return out

def collect_shapes_json(feed: Dict[str, Any], decode_mode: str) -> RtShapes:
    shapes = {}
    added_by_sid = {}
    canceled_by_sid = {}
    def _as_list(v):
        if v is None: return []
        return v if isinstance(v, list) else [v]
    for e in feed.get('entity', []):
        sh = e.get('shape')
        if not sh or not isinstance(sh, dict): continue
        sid = str(sh.get('shape_id') or '')
        enc = sh.get('encoded_polyline')
        if sid and enc:
            try: shapes[sid] = decode_polyline(enc, mode=decode_mode)
            except Exception: pass
        for enc2 in _as_list(sh.get('added_encoded_polylines')):
            try:
                coords = decode_polyline(enc2, mode=decode_mode)
                if len(coords) >= 2:
                    added_by_sid.setdefault(sid, []).append(coords)
            except Exception: pass
        for enc2 in _as_list(sh.get('canceled_encoded_polylines')):
            try:
                coords = decode_polyline(enc2, mode=decode_mode)
                if len(coords) >= 2:
                    canceled_by_sid.setdefault(sid, []).append(coords)
            except Exception: pass
    return RtShapes(shapes=shapes, added_segments=added_by_sid, canceled_segments=canceled_by_sid)

# --- Protobuf binaire
def parse_tripmods_protobuf(data: bytes) -> List[TripModEntity]:
    proto = gtfs_local
    if proto is None:
        from google.transit import gtfs_realtime_pb2 as proto  # type: ignore
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
            repl: List[ReplacementStop] = []
            for rs in getattr(m, 'replacement_stops', []):
                sid = getattr(rs, 'stop_id', None)
                rid = getattr(rs, 'id', None) if hasattr(rs, 'id') else None
                la = getattr(rs, 'stop_lat', None) if hasattr(rs, 'stop_lat') else None
                lo = getattr(rs, 'stop_lon', None) if hasattr(rs, 'stop_lon') else None
                try: la = float(la) if la is not None else None
                except Exception: la = None
                try: lo = float(lo) if lo is not None else None
                except Exception: lo = None
                tt = int(getattr(rs, 'travel_time_to_stop', 0)) if hasattr(rs, 'travel_time_to_stop') else 0
                if (sid is None) and (la is None or lo is None):
                    continue
                repl.append(ReplacementStop(stop_id=sid if sid is not None else None,
                                            id=rid if rid not in (None, "") else None,
                                            stop_lat=la, stop_lon=lo, travel_time_to_stop=tt))
            start_sel = StopSelector(
                stop_sequence=getattr(m.start_stop_selector, 'stop_sequence', None) if hasattr(m, 'start_stop_selector') else None,
                stop_id=getattr(m.start_stop_selector, 'stop_id', None) if hasattr(m, 'start_stop_selector') else None)
            end_sel = StopSelector(
                stop_sequence=getattr(m.end_stop_selector, 'stop_sequence', None) if hasattr(m, 'end_stop_selector') else None,
                stop_id=getattr(m.end_stop_selector, 'stop_id', None) if hasattr(m, 'end_stop_selector') else None)
            mods.append(Modification(start_stop_selector=start_sel, end_stop_selector=end_sel,
                                     replacement_stops=repl,
                                     propagated_modification_delay=getattr(m, 'propagated_modification_delay', None)))
        out.append(TripModEntity(
            entity_id=str(getattr(ent, 'id', '')),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=mods
        ))
    return out

def collect_shapes_pb(data: bytes, decode_mode: str) -> RtShapes:
    shapes = {}; added_by_sid = {}; canceled_by_sid = {}
    proto = gtfs_local
    if proto is None:
        try:
            from google.transit import gtfs_realtime_pb2 as proto  # type: ignore
        except Exception:
            return RtShapes(shapes)
    feed = proto.FeedMessage(); feed.ParseFromString(data)
    for ent in feed.entity:
        if hasattr(ent, 'shape') and ent.shape:
            sid = str(getattr(ent.shape, 'shape_id', '') or '')
            enc = getattr(ent.shape, 'encoded_polyline', None)
            if sid and enc:
                try: shapes[sid] = decode_polyline(enc, mode=decode_mode)
                except Exception: pass
            if sid:
                if hasattr(ent.shape, 'added_encoded_polylines'):
                    for enc2 in getattr(ent.shape, 'added_encoded_polylines'):
                        try:
                            coords = decode_polyline(enc2, mode=decode_mode)
                            if len(coords) >= 2:
                                added_by_sid.setdefault(sid, []).append(coords)
                        except Exception: pass
                if hasattr(ent.shape, 'canceled_encoded_polylines'):
                    for enc2 in getattr(ent.shape, 'canceled_encoded_polylines'):
                        try:
                            coords = decode_polyline(enc2, mode=decode_mode)
                            if len(coords) >= 2:
                                canceled_by_sid.setdefault(sid, []).append(coords)
                        except Exception: pass
    return RtShapes(shapes=shapes, added_segments=added_by_sid, canceled_segments=canceled_by_sid)

# --- TEXTPROTO
def parse_textproto_feed(b: bytes, decode_mode: str) -> Tuple[List[TripModEntity], RtShapes]:
    ents: List[TripModEntity] = []
    shapes: Dict[str, List[Tuple[float, float]]] = {}
    added_by_sid: Dict[str, List[List[Tuple[float, float]]]] = {}
    canceled_by_sid: Dict[str, List[List[Tuple[float, float]]]] = {}

    cur_id: Optional[str] = None
    in_tm = False
    in_shape = False
    capturing_poly = False
    capturing_added = False
    capturing_canceled = False

    tm_buf: Dict[str, Any] = {}
    shape_buf: Dict[str, Any] = {}

    def _lines(b: bytes):
        for raw in b.decode('utf-8', 'ignore').splitlines():
            yield raw.strip()

    def _flush_tm():
        nonlocal tm_buf, cur_id
        if not tm_buf:
            return
        ents.append(TripModEntity(
            entity_id=str(cur_id or ""),
            selected_trips=tm_buf.get('selected_trips', []),
            service_dates=tm_buf.get('service_dates', []),
            start_times=tm_buf.get('start_times', []),
            modifications=tm_buf.get('modifications', []),
        ))
        tm_buf = {}

    def _flush_shape():
        nonlocal shape_buf
        sid = shape_buf.get('shape_id')
        if sid:
            if shape_buf.get('encoded_polyline'):
                enc = str(shape_buf['encoded_polyline'])
                try: shapes[sid] = decode_polyline(enc, mode=decode_mode)
                except Exception: pass
            for enc2 in (shape_buf.get('added_encoded_polylines') or []):
                try:
                    coords = decode_polyline(str(enc2), mode=decode_mode)
                    if len(coords) >= 2:
                        added_by_sid.setdefault(sid, []).append(coords)
                except Exception: pass
            for enc2 in (shape_buf.get('canceled_encoded_polylines') or []):
                try:
                    coords = decode_polyline(str(enc2), mode=decode_mode)
                    if len(coords) >= 2:
                        canceled_by_sid.setdefault(sid, []).append(coords)
                except Exception: pass
        shape_buf = {}

    def _start_new_entity(new_id: Optional[str]):
        nonlocal in_tm, in_shape, capturing_poly, capturing_added, capturing_canceled, cur_id
        if in_tm: _flush_tm()
        if in_shape: _flush_shape()
        in_tm = False; in_shape = False
        capturing_poly = capturing_added = capturing_canceled = False
        cur_id = new_id

    for line in _lines(b):
        if not line:
            continue

        if in_shape and (capturing_poly or capturing_added or capturing_canceled):
            if (line.startswith('id ') or line.startswith('entity')
                or line.startswith('shape_id ') or line.startswith('trip_modifications')
                or line.startswith('shape ') or line.startswith('selected_trips')
                or line.startswith('service_dates ') or line.startswith('start_times ')
                or line.startswith('modifications ') or line.startswith('start_stop_selector')
                or line.startswith('end_stop_selector') or line.startswith('replacement_stops')
                or line.startswith('encoded_polyline ')
                or line.startswith('added_encoded_polylines ')
                or line.startswith('canceled_encoded_polylines ')):
                capturing_poly = capturing_added = capturing_canceled = False
            else:
                if capturing_poly:
                    shape_buf['encoded_polyline'] = shape_buf.get('encoded_polyline', '') + line
                elif capturing_added:
                    lst = shape_buf.setdefault('added_encoded_polylines', [])
                    if lst: lst[-1] = lst[-1] + line
                elif capturing_canceled:
                    lst = shape_buf.setdefault('canceled_encoded_polylines', [])
                    if lst: lst[-1] = lst[-1] + line
                continue

        if line.startswith('entity') or line.startswith('id '):
            new_id = None
            if line.startswith('id '):
                new_id = line[3:].strip()
            else:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == 'id':
                    new_id = parts[2]
            _start_new_entity(new_id); continue

        if line.startswith('trip_modifications'):
            in_tm, in_shape = True, False
            capturing_poly = capturing_added = capturing_canceled = False
            tm_buf = dict(selected_trips=[], service_dates=[], start_times=[], modifications=[])
            continue

        if line.startswith('shape'):
            in_tm, in_shape = False, True
            capturing_poly = capturing_added = capturing_canceled = False
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
                    start_stop_selector=StopSelector(), end_stop_selector=StopSelector(), replacement_stops=[]))
                continue
            if line.startswith('end_stop_selector'):
                if not tm_buf['modifications']:
                    tm_buf['modifications'].append(Modification(
                        start_stop_selector=StopSelector(), end_stop_selector=StopSelector(), replacement_stops=[]))
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
                        start_stop_selector=StopSelector(), end_stop_selector=StopSelector(), replacement_stops=[]))
                continue
            if line.startswith('stop_id '):
                sid = line[len('stop_id '):].strip()
                if tm_buf['modifications']:
                    m = tm_buf['modifications'][-1]
                    if not m.replacement_stops or (m.replacement_stops[-1].stop_id is not None):
                        m.replacement_stops.append(ReplacementStop())
                    m.replacement_stops[-1].stop_id = sid
                continue
            if line.startswith('id '):
                rid = line[len('id '):].strip()
                if tm_buf['modifications'] and tm_buf['modifications'][-1].replacement_stops:
                    tm_buf['modifications'][-1].replacement_stops[-1].id = rid
                continue
            if line.startswith('stop_lat '):
                sla = line[len('stop_lat '):].strip()
                try: la = float(sla) if sla != "" else None
                except: la = None
                if tm_buf['modifications'] and tm_buf['modifications'][-1].replacement_stops:
                    tm_buf['modifications'][-1].replacement_stops[-1].stop_lat = la
                continue
            if line.startswith('stop_lon '):
                slo = line[len('stop_lon '):].strip()
                try: lo = float(slo) if slo != "" else None
                except: lo = None
                if tm_buf['modifications'] and tm_buf['modifications'][-1].replacement_stops:
                    tm_buf['modifications'][-1].replacement_stops[-1].stop_lon = lo
                continue
            continue

        if in_shape:
            if line.startswith('shape_id '):
                shape_buf['shape_id'] = line[len('shape_id '):].strip(); continue
            if line.startswith('encoded_polyline '):
                enc = line[len('encoded_polyline '):]
                shape_buf['encoded_polyline'] = enc
                capturing_poly = True
                continue
            if line.startswith('added_encoded_polylines '):
                enc2 = line[len('added_encoded_polylines '):]
                shape_buf.setdefault('added_encoded_polylines', []).append(enc2)
                capturing_added = True
                continue
            if line.startswith('canceled_encoded_polylines '):
                enc2 = line[len('canceled_encoded_polylines '):]
                shape_buf.setdefault('canceled_encoded_polylines', []).append(enc2)
                capturing_canceled = True
                continue
            continue

    if in_tm: _flush_tm()
    if in_shape: _flush_shape()
    return ents, RtShapes(shapes=shapes, added_segments=added_by_sid, canceled_segments=canceled_by_sid)

# --- Entrée unique
def load_tripmods_bytes(file_bytes: bytes, decode_mode: str) -> Tuple[List[TripModEntity], Optional[Dict[str, Any]], RtShapes]:
    fmt = detect_tripmods_format_bytes(file_bytes)
    if fmt == 'json':
        raw = json.loads(file_bytes.decode('utf-8'))
        feed = normalize_json_keys(raw)
        ents = parse_tripmods_json(feed)
        shapes = collect_shapes_json(feed, decode_mode)
        needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
        if needed_shape_ids:
            shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
            shapes.added_segments = {sid: segs for sid, segs in shapes.added_segments.items() if sid in needed_shape_ids}
            shapes.canceled_segments = {sid: segs for sid, segs in shapes.canceled_segments.items() if sid in needed_shape_ids}
        return ents, feed, shapes
    if fmt == 'textproto':
        ents, shapes = parse_textproto_feed(file_bytes, decode_mode=decode_mode)
        needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
        if needed_shape_ids:
            shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
            shapes.added_segments = {sid: segs for sid, segs in shapes.added_segments.items() if sid in needed_shape_ids}
            shapes.canceled_segments = {sid: segs for sid, segs in shapes.canceled_segments.items() if sid in needed_shape_ids}
        return ents, None, shapes
    # pb
    ents = parse_tripmods_protobuf(file_bytes)
    shapes = collect_shapes_pb(file_bytes, decode_mode)
    needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
    if needed_shape_ids:
        shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
        shapes.added_segments = {sid: segs for sid, segs in shapes.added_segments.items() if sid in needed_shape_ids}
        shapes.canceled_segments = {sid: segs for sid, segs in shapes.canceled_segments.items() if sid in needed_shape_ids}
    return ents, None, shapes