# models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple, Set

@dataclass
class StopSelector:
    stop_sequence: Optional[int] = None
    stop_id: Optional[str] = None

@dataclass
class ReplacementStop:
    stop_id: Optional[str] = None
    id: Optional[str] = None
    stop_lat: Optional[float] = None
    stop_lon: Optional[float] = None
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
    stops_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # stop_id -> {"lat","lon","name"}
    shapes_points: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)  # shape_id -> [(lat,lon),...]

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
    # shape_id -> [(lat, lon), ...] (polyline principale RT)
    shapes: Dict[str, List[Tuple[float, float]]]
    # shape_id -> [ [(lat,lon),...], ... ] segments ajoutés (turquoise)
    added_segments: Dict[str, List[List[Tuple[float, float]]]] = field(default_factory=dict)
    # shape_id -> [ [(lat,lon),...], ... ] segments annulés (violet)
    canceled_segments: Dict[str, List[List[Tuple[float, float]]]] = field(default_factory=dict)