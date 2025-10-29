# analysis.py
from typing import List, Dict, Tuple, Optional
from models import (
    TripModEntity, GtfsStatic, StopSelector, EntityReport, TripCheck
)

def compute_needed_sets(ents: List[TripModEntity]) -> Tuple[set[str], set[str]]:
    needed_trip_ids: set[str] = set()
    needed_stop_ids: set[str] = set()
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

def _seq_from_selector(sel: StopSelector, stop_times_list: List[Dict[str, str]]) -> Optional[int]:
    if sel is None: return None
    if sel.stop_sequence is not None: return sel.stop_sequence
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
``