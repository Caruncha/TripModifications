# app.py
from __future__ import annotations
import json, csv, argparse, dataclasses, sys, io, zipfile
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Tuple
from pathlib import Path

################################################################################
# 0) Import des bindings Protobuf locaux (gtfs_realtime_pb2.py à la racine)
################################################################################
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # priorité au projet
try:
    import gtfs_realtime_pb2 as gtfs_local  # votre fichier local
except Exception:
    gtfs_local = None  # on tentera les bindings pip si nécessaire


################################################################################
# 1) Modèle normalisé en mémoire
################################################################################
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


################################################################################
# 2) Utilitaires: détection JSON/PB, coercions
################################################################################
def _detect_tripmods_format(path: Path) -> str:
    if path.suffix.lower() == '.json':
        return 'json'
    if path.suffix.lower() in ('.pb', '.pbf', '.bin'):
        return 'pb'
    head = path.read_bytes()[:2].lstrip()
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


################################################################################
# 3) Parsing TripModifications (JSON et Protobuf)
################################################################################
def parse_tripmods_json(feed: Dict[str, Any]) -> List[TripModEntity]:
    entities = feed.get('entity') or []
    out: List[TripModEntity] = []
    for e in entities:
        tm = e.get('trip_modifications')
        if not tm:
            continue
        sel_raw = tm.get('selected_trips') or []
        selected = [_coerce_selected_trips(s) for s in sel_raw]

        # Dates au niveau trip_modifications (sinon normalisation depuis selected_trips)
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
    """
    Utilise d'abord le gtfs_realtime_pb2.py local (racine du repo).
    Si indisponible ou incompatible, tente les bindings pip (google.transit...).
    """
    proto = gtfs_local
    if proto is None:
        try:
            from google.transit import gtfs_realtime_pb2 as proto  # fallback pip
        except Exception as ex:
            raise RuntimeError(
                "Impossible d'importer les bindings Protobuf. "
                "Placez `gtfs_realtime_pb2.py` à la racine ou installez "
                "`gtfs-realtime-bindings` compatibles TripModifications."
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
            if hasattr(m, 'start_stop_selector'):
                start_sel = StopSelector(
                    stop_sequence=getattr(m.start_stop_selector, 'stop_sequence', None),
                    stop_id=getattr(m.start_stop_selector, 'stop_id', None),
                )
            else:
                start_sel = StopSelector()
            if hasattr(m, 'end_stop_selector'):
                end_sel = StopSelector(
                    stop_sequence=getattr(m.end_stop_selector, 'stop_sequence', None),
                    stop_id=getattr(m.end_stop_selector, 'stop_id', None),
                )
            else:
                end_sel = StopSelector()

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


def load_tripmods(path: Path) -> List[TripModEntity]:
    fmt = _detect_tripmods_format(path)
    if fmt == 'json':
        feed = json.loads(path.read_text(encoding='utf-8'))
        return parse_tripmods_json(feed)
    else:
        data = path.read_bytes()
        return parse_tripmods_protobuf(data)


################################################################################
# 4) Chargement GTFS statique (zip ou dossier) + index utiles
################################################################################
@dataclass
class GtfsStatic:
    trips: Dict[str, Dict[str, str]]                      # trip_id -> row
    stop_times: Dict[str, List[Dict[str, str]]]           # trip_id -> [ {stop_sequence:int, stop_id:str, ...}, ... ]
    stops: Dict[str, Dict[str, str]]                      # stop_id -> row

def _read_csv(path_or_bytes: io.BytesIO | Path, fname: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if isinstance(path_or_bytes, Path) and path_or_bytes.is_dir():
        p = path_or_bytes / fname
        if not p.exists():
            return rows
        with p.open('r', encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                rows.append({k: (v or "").strip() for k, v in row.items()})
        return rows

    # zip
    if isinstance(path_or_bytes, Path):
        with zipfile.ZipFile(path_or_bytes, 'r') as zf:
            if fname not in zf.namelist():
                return rows
            with zf.open(fname) as f:
                text = io.TextIOWrapper(f, encoding='utf-8-sig', newline='')
                for row in csv.DictReader(text):
                    rows.append({k: (v or "").strip() for k, v in row.items()})
        return rows

    # not expected
    return rows

def load_gtfs(gtfs_path: str | Path) -> GtfsStatic:
    p = Path(gtfs_path)
    trips = {r['trip_id']: r for r in _read_csv(p, 'trips.txt') if 'trip_id' in r}
    # stop_times triés par stop_sequence (int)
    st_rows = _read_csv(p, 'stop_times.txt')
    stop_times: Dict[str, List[Dict[str, str]]] = {}
    for r in st_rows:
        trip_id = r.get('trip_id')
        if not trip_id:
            continue
        lst = stop_times.setdefault(trip_id, [])
        # cast safe
        try:
            r['stop_sequence'] = str(int(r.get('stop_sequence', '').strip()))
        except Exception:
            r['stop_sequence'] = ''
        lst.append(r)
    for trip_id, lst in stop_times.items():
        lst.sort(key=lambda x: int(x['stop_sequence'] or 0))
    stops = {r['stop_id']: r for r in _read_csv(p, 'stops.txt') if 'stop_id' in r}
    return GtfsStatic(trips=trips, stop_times=stop_times, stops=stops)


################################################################################
# 5) Analyse TripModifications vs GTFS
################################################################################
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
    """Résout un stop_selector vers un stop_sequence (priorité au sequence, fallback par stop_id)."""
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
                    # on évalue sur TOUTES les modifications de l’entité (même si plusieurs)
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

        # replacement_stops (on signale ceux inconnus dans stops.txt; NB: des arrêts temporaires peuvent être normaux)
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


################################################################################
# 6) CLI de l’app
################################################################################
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Analyse GTFS + TripModifications (JSON ou PB)."
    )
    ap.add_argument("--gtfs", required=True, help="Chemin GTFS (.zip ou dossier)")
    ap.add_argument("--tripmods", required=True, help="Chemin TripModifications (.json ou .pb)")
    ap.add_argument("--report-json", default="", help="Fichier où écrire le rapport JSON (optionnel)")
    ap.add_argument("--dump-first", action="store_true", help="Afficher le 1er trip_mod normalisé")
    args = ap.parse_args(argv)

    gtfs = load_gtfs(args.gtfs)
    ents = load_tripmods(Path(args.tripmods))

    # Résumé console
    print(f"Chargé: {len(gtfs.trips):,} trips, {sum(len(v) for v in gtfs.stop_times.values()):,} stop_times, {len(gtfs.stops):,} stops")
    print(f"TripModifications: {len(ents)} entités")
    if args.dump_first and ents:
        print("\n--- 1er trip_mod (normalisé) ---")
        print(json.dumps(asdict(ents[0]), ensure_ascii=False, indent=2))

    reports, totals = analyze_tripmods_with_gtfs(gtfs, ents)

    print("\n=== Résumé ===")
    print(f"- Entités: {totals['total_entities']}")
    print(f"- trip_ids sélectionnés: {totals['total_trip_ids']}")
    print(f"- modifications: {totals['total_modifications']}")
    print(f"- trip_ids manquants: {totals['missing_trip_ids']}")
    print(f"- selectors non résolus: {totals['invalid_selectors']}")
    print(f"- replacement_stops inconnus dans GTFS: {totals['unknown_replacement_stops']} (peuvent être des arrêts temporaires)")

    if args.report_json:
        out = {
            "totals": totals,
            "entities": [asdict(r) for r in reports]
        }
        Path(args.report_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nRapport écrit: {args.report_json}")


if __name__ == "__main__":
    main()
