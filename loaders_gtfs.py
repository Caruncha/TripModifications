# loaders_gtfs.py
import csv, io, zipfile
from typing import Dict, Set, Tuple, List
from models import GtfsStatic

def load_gtfs_zip_filtered_bytes(zip_bytes: bytes, needed_trip_ids: Set[str], needed_stop_ids: Set[str]) -> GtfsStatic:
    trips: Dict[str, Dict[str, str]] = {}
    stop_times: Dict[str, List[Dict[str, str]]] = {}
    stops_present: Set[str] = set()
    stops_info: Dict[str, Dict[str, float]] = {}
    shapes_points: Dict[str, List[Tuple[float, float]]] = {}

    if not needed_trip_ids and not needed_stop_ids:
        return GtfsStatic(trips=trips, stop_times=stop_times, stops_present=stops_present,
                          stops_info=stops_info, shapes_points=shapes_points)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        # trips.txt
        if 'trips.txt' in zf.namelist() and needed_trip_ids:
            with zf.open('trips.txt') as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    tid = (row.get('trip_id') or '').strip()
                    if tid in needed_trip_ids:
                        trips[tid] = {k: (v or "").strip() for k, v in row.items()}

        # stop_times.txt
        stops_from_trips: Set[str] = set()
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
                        sid = (rec.get('stop_id') or '').strip()
                        if sid: stops_from_trips.add(sid)
            for tid, lst in stop_times.items():
                lst.sort(key=lambda x: int(x['stop_sequence'] or 0))

        # stops.txt
        all_needed_stop_ids: Set[str] = set(needed_stop_ids) | stops_from_trips
        if 'stops.txt' in zf.namelist() and all_needed_stop_ids:
            with zf.open('stops.txt') as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    sid = (row.get('stop_id') or '').strip()
                    if sid in all_needed_stop_ids:
                        try:
                            lat = float((row.get('stop_lat') or '').strip())
                            lon = float((row.get('stop_lon') or '').strip())
                        except Exception:
                            lat = lon = None
                        name = (row.get('stop_name') or '').strip()
                        if lat is not None and lon is not None:
                            stops_info[sid] = {"lat": lat, "lon": lon, "name": name}
                            stops_present.add(sid)

        # shapes.txt
        needed_shape_ids: Set[str] = set()
        for tid, rec in trips.items():
            sid = rec.get('shape_id') or ''
            if sid:
                needed_shape_ids.add(sid)
        if 'shapes.txt' in zf.namelist() and needed_shape_ids:
            temp: Dict[str, List[Tuple[int, float, float]]] = {}
            with zf.open('shapes.txt') as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline=''))
                for r in reader:
                    sid = (r.get('shape_id') or '').strip()
                    if sid in needed_shape_ids:
                        try:
                            la = float((r.get('shape_pt_lat') or '').strip())
                            lo = float((r.get('shape_pt_lon') or '').strip())
                            seq = int((r.get('shape_pt_sequence') or '').strip())
                        except Exception:
                            continue
                        temp.setdefault(sid, []).append((seq, la, lo))
            for sid, rows in temp.items():
                rows.sort(key=lambda x: x[0])
                shapes_points[sid] = [(la, lo) for _, la, lo in rows]

    return GtfsStatic(trips=trips, stop_times=stop_times, stops_present=stops_present,
                      stops_info=stops_info, shapes_points=shapes_points)