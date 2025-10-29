# caching.py
from dataclasses import asdict
from typing import Dict, Any, Tuple, List
import hashlib
import streamlit as st

from tripmodifications.analysis import compute_needed_sets, analyze_tripmods_with_gtfs
from tripmodifications.loaders_gtfs import load_gtfs_zip_filtered_bytes
from tripmodifications.parsers import load_tripmods_bytes
from tripmodifications.map_view import build_folium_map_for_polyline
from tripmodifications.config import SCHEMA_VERSION  # si utilisé quelque par


def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

@st.cache_resource(show_spinner=True)
def resource_parse_tripmods(tripmods_bytes: bytes, decode_flag: str, schema: str):
    ents, feed_json, rt_shapes = load_tripmods_bytes(tripmods_bytes, decode_mode=decode_flag)
    return ents, feed_json, rt_shapes

@st.cache_resource(show_spinner=True)
def resource_load_gtfs(gtfs_bytes: bytes, needed_trip_ids_sorted: Tuple[str, ...], needed_stop_ids_sorted: Tuple[str, ...], schema: str):
    gtfs = load_gtfs_zip_filtered_bytes(gtfs_bytes, set(needed_trip_ids_sorted), set(needed_stop_ids_sorted))
    if not hasattr(gtfs, "shapes_points") or gtfs.shapes_points is None:
        gtfs.shapes_points = {}
    return gtfs

@st.cache_resource(show_spinner=False)
def resource_build_map_html(
    shape_id_key: str,
    poly_key: Tuple[Tuple[float, float], ...],
    stops_key: Tuple[Tuple[float, float, str], ...],
    original_poly_key: Tuple[Tuple[float, float], ...],
    original_stop_points_key: Tuple[Tuple[float, float, str], ...],
    original_shape_id_key: str,
    added_segments_key: Tuple[Tuple[Tuple[float, float], ...], ...],
    canceled_segments_key: Tuple[Tuple[Tuple[float, float], ...], ...],
    schema: str
) -> str:
    poly = [(la, lo) for (la, lo) in poly_key]
    stops = [(la, lo, lab) for (la, lo, lab) in stops_key]
    orig = [(la, lo) for (la, lo) in original_poly_key] if original_poly_key else None
    orig_stops = [(la, lo, lab) for (la, lo, lab) in original_stop_points_key] if original_stop_points_key else None
    added_segments = [[(la, lo) for (la, lo) in seg] for seg in added_segments_key] if added_segments_key else None
    canceled_segments = [[(la, lo) for (la, lo) in seg] for seg in canceled_segments_key] if canceled_segments_key else None

    fmap = build_folium_map_for_polyline(
        poly, shape_id=shape_id_key,
        replacement_stop_points=stops,
        original_poly=orig,
        original_stop_points=orig_stops,
        original_shape_id=original_shape_id_key or None,
        added_segments=added_segments,
        canceled_segments=canceled_segments
    )
    if fmap is None:
        return "<div>Carte indisponible</div>"
    return fmap.get_root().render()

@st.cache_data(show_spinner=True, ttl=86400, max_entries=32, hash_funcs={bytes: _hash_bytes})
def cache_views(tripmods_bytes: bytes, gtfs_bytes: bytes, decode_flag: str, schema: str) -> Dict[str, Any]:
    # 1) Parse TripMods
    ents, feed_json, rt_shapes = resource_parse_tripmods(tripmods_bytes, decode_flag, schema)
    needed_trip_ids, needed_stop_ids = compute_needed_sets(ents)
    tid_sorted = tuple(sorted(needed_trip_ids))
    sid_sorted = tuple(sorted(needed_stop_ids))

    # 2) GTFS filtré
    gtfs = resource_load_gtfs(gtfs_bytes, tid_sorted, sid_sorted, schema)
    present_trip_ids = sorted(gtfs.trips.keys())
    missing_trip_ids = sorted(set(tid_sorted) - set(present_trip_ids))

    # 3) Analyse
    reports, totals = analyze_tripmods_with_gtfs(gtfs, ents)
    reports_plain = [asdict(r) for r in reports]
    total_shapes = len(rt_shapes.shapes)

    # 4) Détails (tout JSON-compatible)
    details_tables_by_entity: Dict[str, List[Dict[str, Any]]] = {}
    temp_stops_points_by_entity: Dict[str, List[List[Any]]] = {}
    shape_for_plot_by_entity: Dict[str, str | None] = {}
    original_poly_by_entity: Dict[str, List[List[float]]] = {}
    original_shape_id_by_entity: Dict[str, str | None] = {}
    original_stop_points_by_entity: Dict[str, List[List[Any]]] = {}
    original_stop_ids_by_entity: Dict[str, List[str]] = {}
    added_segments_by_entity: Dict[str, List[List[List[float]]]] = {}
    canceled_segments_by_entity: Dict[str, List[List[List[float]]]] = {}

    def _is_poly_anormal(coords):  # util interne
        if not coords or len(coords) < 2: return True
        lats = [la for la,_ in coords]; lons = [lo for _,lo in coords]
        return (max(lats)-min(lats) + max(lons)-min(lons)) < 1e-4

    stops_info = getattr(gtfs, "stops_info", {}) or {}
    shapes_pts = getattr(gtfs, "shapes_points", {}) or {}

    for r in reports:
        ent_id = r.entity_id
        ent_obj = next((e for e in ents if e.entity_id == ent_id), None)
        if not ent_obj or not ent_obj.modifications: continue

        shape_ids_in_entity = [s.shape_id for s in ent_obj.selected_trips if s.shape_id]
        shape_id_for_plot = next((sid for sid in shape_ids_in_entity if sid in rt_shapes.shapes), None)
        shape_for_plot_by_entity[ent_id] = shape_id_for_plot

        # segments RT pour la shape retenue
        if shape_id_for_plot:
            add_segs = rt_shapes.added_segments.get(shape_id_for_plot, [])
            can_segs = rt_shapes.canceled_segments.get(shape_id_for_plot, [])
            added_segments_by_entity[ent_id] = [[[la, lo] for (la, lo) in seg] for seg in add_segs]
            canceled_segments_by_entity[ent_id] = [[[la, lo] for (la, lo) in seg] for seg in can_segs]
        else:
            added_segments_by_entity[ent_id] = []
            canceled_segments_by_entity[ent_id] = []

        # tableau diagnostics
        trip_counts: Dict[str, int] = {}
        for s in ent_obj.selected_trips:
            for tid in s.trip_ids:
                trip_counts[tid] = trip_counts.get(tid, 0) + 1
        mixed_shapes = len({sid for sid in shape_ids_in_entity if sid}) > 1
        detail_rows: List[Dict[str, Any]] = []

        chosen_original_shape_id = None
        chosen_original_trip_id = None

        for t in r.trips:
            trip_id = t.trip_id if hasattr(t, "trip_id") else t["trip_id"]
            st_list = gtfs.stop_times.get(trip_id, [])
            stop_times_count = len(st_list)
            trip_shape_id = None
            for s in ent_obj.selected_trips:
                if trip_id in s.trip_ids and s.shape_id:
                    trip_shape_id = s.shape_id
                    break
            if not trip_shape_id and shape_ids_in_entity:
                trip_shape_id = shape_ids_in_entity[0]
            poly = rt_shapes.shapes.get(trip_shape_id, []) if trip_shape_id else []
            poly_points = len(poly)
            poly_anormal = _is_poly_anormal(poly)

            # convertir TripCheck en dict si nécessaire
            if hasattr(t, "__dict__"):
                exists = t.exists_in_gtfs; start_ok = t.start_seq_valid; end_ok = t.end_seq_valid
                start_seq = t.start_seq; end_seq = t.end_seq
                notes = "; ".join(t.notes) if t.notes else ""
            else:
                exists = t["exists_in_gtfs"]; start_ok = t["start_seq_valid"]; end_ok = t["end_seq_valid"]
                start_seq = t.get("start_seq"); end_seq = t.get("end_seq")
                notes = "; ".join(t.get("notes", [])) if t.get("notes") else ""

            selectors_incomplets = not (start_ok and end_ok)
            ordre_ok = ""
            ecart_seq = ""
            if (start_seq is not None) and (end_seq is not None):
                ordre_ok = "oui" if start_seq <= end_seq else "non"
                if isinstance(start_seq, int) and isinstance(end_seq, int):
                    ecart_seq = end_seq - start_seq

            duplicate_trip = trip_counts.get(trip_id, 0) > 1
            detail_rows.append({
                "trip_id": trip_id,
                "existe dans GTFS": "oui" if exists else "non",
                "start_seq": start_seq if start_seq is not None else "",
                "end_seq": end_seq if end_seq is not None else "",
                "selectors OK": "oui" if (start_ok and end_ok) else "non",
                "notes": notes,
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

            if chosen_original_shape_id is None and exists:
                trip_row = gtfs.trips.get(trip_id, {})
                static_sid = (trip_row.get('shape_id') or '').strip()
                if static_sid and static_sid in shapes_pts and len(shapes_pts.get(static_sid, [])) >= 2:
                    chosen_original_shape_id = static_sid
                    chosen_original_trip_id = trip_id

        details_tables_by_entity[ent_id] = detail_rows

        # Replacement stops (rose)
        tmp_pts: List[List[Any]] = []
        seen = set()
        for m in ent_obj.modifications:
            for rs in m.replacement_stops:
                label = (rs.id or rs.stop_id or "").strip()
                la = rs.stop_lat; lo = rs.stop_lon
                if (la is None or lo is None) and rs.stop_id:
                    info = stops_info.get(rs.stop_id)
                    if info:
                        la = info.get("lat"); lo = info.get("lon")
                if la is None or lo is None: continue
                key = (round(la, 7), round(lo, 7), label)
                if key in seen: continue
                tmp_pts.append([la, lo, label if label else "replacement_stop"])
                seen.add(key)
        temp_stops_points_by_entity[ent_id] = tmp_pts

        # Tracé originel + arrêts originels
        if chosen_original_shape_id:
            orig = shapes_pts.get(chosen_original_shape_id, [])
            original_poly_by_entity[ent_id] = [[la, lo] for (la, lo) in orig]
            original_shape_id_by_entity[ent_id] = chosen_original_shape_id
        else:
            original_poly_by_entity[ent_id] = []
            original_shape_id_by_entity[ent_id] = None

        orig_pts: List[List[Any]] = []
        orig_ids: List[str] = []
        if chosen_original_trip_id:
            for rec in gtfs.stop_times.get(chosen_original_trip_id, []):
                sid = (rec.get('stop_id') or '').strip()
                if not sid: continue
                info = stops_info.get(sid)
                if not info: continue
                la, lo = info.get("lat"), info.get("lon")
                if la is None or lo is None: continue
                orig_pts.append([la, lo, sid])
                orig_ids.append(sid)
        original_stop_points_by_entity[ent_id] = orig_pts
        original_stop_ids_by_entity[ent_id] = orig_ids

    shapes_plain = {sid: [[la, lo] for (la, lo) in coords] for sid, coords in rt_shapes.shapes.items()}
    gtfs_kpi = dict(trips=len(gtfs.trips),
                    stop_times=sum(len(v) for v in gtfs.stop_times.values()),
                    stops_present=len(gtfs.stops_present))

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
        "original_poly_by_entity": original_poly_by_entity,
        "original_shape_id_by_entity": original_shape_id_by_entity,
        "original_stop_points_by_entity": original_stop_points_by_entity,
        "original_stop_ids_by_entity": original_stop_ids_by_entity,
        "added_segments_by_entity": added_segments_by_entity,
        "canceled_segments_by_entity": canceled_segments_by_entity,
        "gtfs_kpi": gtfs_kpi,
        "present_trip_ids": present_trip_ids,
        "missing_trip_ids": missing_trip_ids,
    }

