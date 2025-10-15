# -*- coding: utf-8 -*-
"""
App Streamlit : Validation & Visualisation des détours (GTFS-realtime TripModifications) + GTFS

Fonctions :
- Upload GTFS (.zip) + TripModifications (.pb/.json/.proto)
- Parsing Protobuf (binaire) ou JSON -> message TripModifications
- Validation sémantique minimale (selected_trips, service_dates, modifications)
- Reconstruit le trajet de base (shapes.txt) et la géométrie du détour (Shape ou ReplacementStops)
- Affiche "trajet vs détour" sur une carte Altair

Prérequis :
- Fichier local 'gtfs_realtime_pb2.py' (généré avec protoc depuis le proto officiel GTFS-rt)
- requirements.txt : streamlit, altair, pandas, protobuf

Références :
- Proto officiel GTFS‑rt (inclut TripModifications/Shape/ReplacementStop, statut expérimental) :
  https://gtfs.org/documentation/realtime/proto/
- Référence sémantique "Trip Modifications" :
  https://gtfs.org/documentation/realtime/feed-entities/trip-modifications/
- Altair & GeoJSON/mark_geoshape dans Streamlit (recommandations & workaround) :
  https://stackoverflow.com/questions/55923300/how-can-i-make-a-map-using-geojson-data-in-altair
"""

from __future__ import annotations

import re
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# ⚠️ IMPORTANT : import LOCAL (fichier généré par protoc, versionné dans le repo)
import gtfs_realtime_pb2  # noqa: E402

# Utilitaires Protobuf JSON <-> Message
from google.protobuf.json_format import ParseDict  # noqa: E402


# ------------------------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------------------------
st.set_page_config(page_title="TripModifications (GTFS-rt) + Carte des détours", layout="wide")


# ------------------------------------------------------------------------------
# Utilitaires GTFS
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_gtfs_tables(gtfs_zip: bytes) -> Dict[str, pd.DataFrame]:
    """Lit routes/trips/stops/shapes depuis un GTFS .zip."""
    with zipfile.ZipFile(io.BytesIO(gtfs_zip)) as z:
        def rd(name, dtype=None):
            with z.open(name) as f:
                return pd.read_csv(f, dtype=dtype)

        trips = rd("trips.txt", dtype=str)
        routes = rd("routes.txt", dtype=str)
        stops = rd("stops.txt", dtype=str)
        shapes = rd("shapes.txt", dtype={"shape_id": str, "shape_pt_lat": float,
                                         "shape_pt_lon": float, "shape_pt_sequence": int})

    shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"])
    return {"trips": trips, "routes": routes, "stops": stops, "shapes": shapes}

_CAMEL_TO_SNAKE_RE_1 = re.compile('(.)([A-Z][a-z]+)')
_CAMEL_TO_SNAKE_RE_2 = re.compile('([a-z0-9])([A-Z])')

# Mapping explicite pour certains noms "officiels" du proto
# (afin de gérer proprement serviceDates -> service_dates, etc.)
JSON_FIELD_MAPPING = {
    # Top-level TripModifications
    "selectedTrips": "selected_trips",
    "startTimes": "start_times",
    "serviceDates": "service_dates",
    "modifications": "modifications",

    # SelectedTrips
    "tripIds": "trip_ids",
    "routeId": "route_id",
    "directionId": "direction_id",

    # Modification (détours)
    "replacementStops": "replacement_stops",
    "startStopSelector": "start_stop_selector",
    "endStopSelector": "end_stop_selector",

    # ReplacementStop
    "stopId": "stop_id",
    "travelTimeToStop": "travel_time_to_stop",

    # Shape (RT)
    "encodedPolyline": "encoded_polyline",
    "shapeId": "shape_id",
    # Points (si présents)
    "shapePtLat": "shape_pt_lat",
    "shapePtLon": "shape_pt_lon",

    # Divers sélecteurs (parfois via EntitySelector équivalents)
    "stopIdSelector": "stop_id_selector",  # par précaution
}


def camel_to_snake(name: str) -> str:
    """Convertit lowerCamelCase/PascalCase vers snake_case."""
    s1 = _CAMEL_TO_SNAKE_RE_1.sub(r'\1_\2', name)
    return _CAMEL_TO_SNAKE_RE_2.sub(r'\1_\2', s1).lower()


def normalize_keys(obj):
    """
    Normalise récursivement les clés de dict :
      1) mapping explicite (JSON_FIELD_MAPPING)
      2) camelCase -> snake_case
    Conserve les listes telles quelles (en normalisant leur contenu).
    """
    if isinstance(obj, dict):
        new_d = {}
        for k, v in obj.items():
            if k in JSON_FIELD_MAPPING:
                nk = JSON_FIELD_MAPPING[k]
            else:
                nk = camel_to_snake(k)
            new_d[nk] = normalize_keys(v)
        return new_d
    elif isinstance(obj, list):
        return [normalize_keys(x) for x in obj]
    else:
        return obj

def shape_id_for_trip(trips: pd.DataFrame, trip_id: str) -> Optional[str]:
    row = trips.loc[trips["trip_id"] == str(trip_id)]
    if not row.empty and "shape_id" in row:
        return str(row.iloc[0]["shape_id"])
    return None


def polyline_for_shape(shapes: pd.DataFrame, shape_id: str) -> Optional[List[Tuple[float, float]]]:
    segs = shapes.loc[shapes["shape_id"] == shape_id]
    if segs.empty:
        return None
    return list(zip(segs["shape_pt_lon"].tolist(), segs["shape_pt_lat"].tolist()))


def stops_lookup(stops: pd.DataFrame, stop_id: str) -> Optional[Tuple[float, float]]:
    row = stops.loc[stops["stop_id"] == str(stop_id)]
    if row.empty:
        return None
    return float(row.iloc[0]["stop_lon"]), float(row.iloc[0]["stop_lat"])


# ------------------------------------------------------------------------------
# Utilitaires GeoJSON/Polyline/Altair
# ------------------------------------------------------------------------------
def decode_google_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """Décodage 'Google Encoded Polyline' -> [(lon, lat), ...]."""
    coords = []
    index, lat, lon = 0, 0, 0
    length = len(polyline_str)

    def _decode():
        nonlocal index
        result, shift = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        return ~(result >> 1) if (result & 1) else (result >> 1)

    while index < length:
        lat += _decode()
        lon += _decode()
        coords.append((lon / 1e5, lat / 1e5))
    return coords


def feature_line(coords: List[Tuple[float, float]], props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "LineString", "coordinates": coords}}


def fc(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


# ------------------------------------------------------------------------------
# Parsing & Validation TripModifications (réf. GTFS‑rt)
# ------------------------------------------------------------------------------
DATE_RE = re.compile(r"^\d{8}$")  # YYYYMMDD attendu

def parse_trip_modifications_from_pb(pb_bytes: bytes) -> gtfs_realtime_pb2.TripModifications:
    """Parse binaire .pb -> message TripModifications."""
    msg = gtfs_realtime_pb2.TripModifications()
    msg.ParseFromString(pb_bytes)
    return msg


def parse_trip_modifications_from_json(file_obj):
    """Parse JSON -> message TripModifications en tolérant camelCase et snake_case."""
    # 1) Charger le JSON brut
    try:
        tm_dict_raw = json.load(file_obj)
    except Exception as e:
        st.error(f"Erreur lecture JSON : {e}")
        st.stop()

    # 2) Normaliser les clés vers snake_case attendu par le proto
    tm_dict_norm = normalize_keys(tm_dict_raw)

    # 3) Tenter ParseDict sur la version normalisée
    msg = gtfs_realtime_pb2.TripModifications()
    try:
        ParseDict(tm_dict_norm, msg, ignore_unknown_fields=True)
        return msg, tm_dict_norm
    except Exception as e_norm:
        # 4) En fallback, tenter la version brute (si elle était déjà conforme)
        try:
            msg2 = gtfs_realtime_pb2.TripModifications()
            ParseDict(tm_dict_raw, msg2, ignore_unknown_fields=True)
            return msg2, tm_dict_raw
        except Exception as e_raw:
            expected = {"selected_trips", "service_dates", "modifications"}
            present = set(tm_dict_norm.keys()) if isinstance(tm_dict_norm, dict) else set()
            st.error(
                "Le JSON ne correspond pas au schéma TripModifications.\n\n"
                f"- Erreur (normalisé): {e_norm}\n"
                f"- Erreur (brut)     : {e_raw}\n\n"
                f"Clés attendues (indicatif): {sorted(expected)}\n"
                f"Clés présentes top-level  : {sorted(present) if present else '—'}"
            )
            st.stop()


def basic_validation(tm: gtfs_realtime_pb2.TripModifications) -> List[str]:
    """Contrôles sémantiques minimaux recommandés (réf. Trip Modifications, statut expérimental)."""
    issues = []
    if len(tm.selected_trips) == 0:
        issues.append("`selected_trips` est vide : aucun trip ciblé.")
    if len(tm.service_dates) == 0:
        issues.append("`service_dates` est vide : aucune date de validité.")
    else:
        bad = [d for d in tm.service_dates if not DATE_RE.match(d)]
        if bad:
            issues.append(f"`service_dates` mal formatés (YYYYMMDD attendu) : {', '.join(bad)}")
    if len(tm.modifications) == 0:
        issues.append("`modifications` est vide : aucune modification/détour fournie.")
    return issues


def infer_trip_ids_from_selected_trips(tm: gtfs_realtime_pb2.TripModifications,
                                       trips: pd.DataFrame) -> List[str]:
    """
    Best-effort pour récupérer des trip_id depuis selected_trips (message expérimental).
    Si le champ trip_ids existe -> on l'utilise ; sinon fallback route_id/direction_id.
    """
    trip_ids = set()
    for sel in tm.selected_trips:
        if hasattr(sel, "trip_ids") and len(sel.trip_ids) > 0:
            for tid in sel.trip_ids:
                trip_ids.add(str(tid))
        else:
            route_id = getattr(sel, "route_id", "")
            direction_id = getattr(sel, "direction_id", None)
            if route_id:
                subset = trips.loc[trips["route_id"] == str(route_id)]
                if direction_id in ("0", "1"):
                    subset = subset.loc[subset["direction_id"] == str(direction_id)]
                for tid in subset["trip_id"].tolist():
                    trip_ids.add(str(tid))
    return list(trip_ids)


def extract_detour_geometry(modif: Any, stops_df: pd.DataFrame) -> Optional[List[Tuple[float, float]]]:
    """
    Produit une polyligne (lon, lat) pour la modification :
      1) Shape.encoded_polyline si présent (message Shape du RT)
      2) À défaut, chaînage des ReplacementStop (stop_id -> coordonnées de stops.txt)
    """
    # 1) Shape encodé
    if hasattr(modif, "shape") and modif.HasField("shape"):
        shp = modif.shape
        if hasattr(shp, "encoded_polyline") and shp.encoded_polyline:
            return decode_google_polyline(shp.encoded_polyline)
        if hasattr(shp, "points") and len(shp.points) > 0:
            coords = [(p.lon, p.lat) for p in shp.points]
            if coords:
                return coords

    # 2) Replacement stops
    if hasattr(modif, "replacement_stops") and len(modif.replacement_stops) > 0:
        rs = list(modif.replacement_stops)
        try:
            rs.sort(key=lambda r: getattr(r, "travel_time_to_stop", 0))
        except Exception:
            pass
        coords = []
        for r in rs:
            sid = getattr(r, "stop_id", "")
            if sid:
                xy = stops_lookup(stops_df, sid)
                if xy:
                    coords.append(xy)
        if len(coords) >= 2:
            return coords

    return None


# ------------------------------------------------------------------------------
# UI — Upload, Parsing multi-formats, Validation, Carte
# ------------------------------------------------------------------------------
st.title("TripModifications (GTFS‑realtime) — Validation & Carte des détours")

with st.sidebar:
    st.header("Fichiers d’entrée")
    gtfs_file = st.file_uploader("GTFS (.zip)", type=["zip"])
    tm_file = st.file_uploader(
        "TripModifications (.pb, .json, ou .proto)",
        type=["pb", "bin", "pbf", "dat", "json", "proto"]
    )
    st.markdown("---")
    base_color = st.color_picker("Couleur du trajet de base", "#888888")
    detour_color = st.color_picker("Couleur du détour", "#d62728")
    st.markdown("---")
    with st.expander("Debug (chemin du module proto)"):
        st.caption(f"gtfs_realtime_pb2 chargé depuis : {getattr(gtfs_realtime_pb2, '__file__', '???')}")

if not gtfs_file or not tm_file:
    st.info("Charge un **GTFS (.zip)** et un **TripModifications** (.pb recommandé).")
    st.stop()

# Lire GTFS
try:
    gtfs = read_gtfs_tables(gtfs_file.read())
    trips_df, routes_df, stops_df, shapes_df = gtfs["trips"], gtfs["routes"], gtfs["stops"], gtfs["shapes"]
except Exception as e:
    st.error(f"Erreur lecture GTFS : {e}")
    st.stop()

# Détection extension & parsing
ext = Path(tm_file.name).suffix.lower()
tm_msg: Optional[gtfs_realtime_pb2.TripModifications] = None
tm_raw_dict: Optional[Dict[str, Any]] = None

if ext in [".pb", ".bin", ".pbf", ".dat"]:
    try:
        tm_msg = parse_trip_modifications_from_pb(tm_file.read())
        st.success(f"Binaire TripModifications chargé : {tm_file.name}")
    except Exception as e:
        st.error(f"Fichier non conforme (TripModifications .pb attendu) : {e}")
        st.stop()

elif ext == ".json":
    try:
        tm_msg, tm_raw_dict = parse_trip_modifications_from_json(tm_file)
        st.success(f"JSON TripModifications chargé : {tm_file.name}")
        with st.expander("Aperçu JSON (normalisé)"):
            st.json(tm_raw_dict)
    except Exception as e:
        st.error(f"Erreur lors du parsing JSON -> TripModifications : {e}")
        st.stop()

elif ext == ".proto":
    preview = tm_file.read(4096).decode("utf-8", errors="ignore")
    st.warning(
        "Le fichier chargé est un **schéma Protobuf (.proto)** — pas un message TripModifications.\n"
        "→ Compile-le localement en .pb ou en module Python, puis recharge le .pb pour l’analyse."
    )
    with st.expander("Aperçu du .proto (4096 premiers octets)"):
        st.code(preview, language="protobuf")
    st.stop()
else:
    st.error(f"Extension non supportée : {ext}")
    st.stop()

# Validation sémantique minimale
issues = basic_validation(tm_msg)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Contrôles de conformité")
    if issues:
        for it in issues:
            st.error("• " + it)
    else:
        st.success("Structure minimale OK (selected_trips, service_dates, modifications).")
with col2:
    st.subheader("Résumé")
    st.write(f"- selected_trips : **{len(tm_msg.selected_trips)}**")
    st.write(f"- service_dates  : **{len(tm_msg.service_dates)}**")
    st.write(f"- modifications  : **{len(tm_msg.modifications)}**")
    st.caption("Rappel : Trip Modifications est **expérimental** dans GTFS‑rt et peut évoluer.")

# Trip(s) impactés (pour tracer le trajet de base)
impacted_trip_ids = infer_trip_ids_from_selected_trips(tm_msg, trips_df)
if impacted_trip_ids:
    st.info(f"Trip(s) impacté(s) détecté(s) : {', '.join(impacted_trip_ids[:10]) + ('…' if len(impacted_trip_ids)>10 else '')}")
else:
    st.warning("Aucun `trip_id` déduit de `selected_trips` — le trajet de base pourrait ne pas s’afficher.")

# Préparer la liste de détours cartographiables (une entrée par modification)
detour_items = []
for idx, modif in enumerate(tm_msg.modifications):
    geom = extract_detour_geometry(modif, stops_df)
    base_trip = impacted_trip_ids[0] if impacted_trip_ids else None
    detour_items.append({"index": idx, "geom": geom, "base_trip_id": base_trip})

if not detour_items:
    st.warning("Aucune modification exploitable pour la carte (Shape/ReplacementStops absents).")
    st.stop()

labels = [f"Modification #{it['index']} — geom={'oui' if it['geom'] else 'non'} — base_trip_id={it['base_trip_id'] or '—'}"
          for it in detour_items]
sel = st.selectbox("Choisir une modification à visualiser", labels)
sel_item = detour_items[labels.index(sel)]

# Construire GeoJSON
features = []

# Trajet de base (shape depuis GTFS) si on a un trip_id
if sel_item["base_trip_id"]:
    sid = shape_id_for_trip(trips_df, sel_item["base_trip_id"])
    if sid:
        base_coords = polyline_for_shape(shapes_df, sid)
        if base_coords:
            features.append(feature_line(base_coords, {"kind": "base", "shape_id": sid}))
    else:
        st.warning("`shape_id` introuvable pour le trip représentatif.")

# Détour (Shape RT ou ReplacementStops)
if sel_item["geom"]:
    features.append(feature_line(sel_item["geom"], {"kind": "detour", "mod_index": sel_item["index"]}))
else:
    st.warning("Pas de géométrie de détour disponible pour cette modification.")

if not features:
    st.error("Aucune géométrie à afficher.")
    st.stop()

# Altair (inline features)
data_values = json.loads(json.dumps(fc(features)))["features"]

base_layer = alt.Chart(alt.Data(values=[f for f in data_values if f["properties"]["kind"] == "base"])) \
    .mark_geoshape(fill=None, stroke=base_color, strokeWidth=2) \
    .encode(tooltip=[alt.Tooltip("properties.shape_id:N", title="shape_id")]) \
    .project(type="mercator")

detour_layer = alt.Chart(alt.Data(values=[f for f in data_values if f["properties"]["kind"] == "detour"])) \
    .mark_geoshape(fill=None, stroke=detour_color, strokeDash=[6, 4], strokeWidth=3) \
    .encode(tooltip=[alt.Tooltip("properties.mod_index:N", title="modification")]) \
    .project(type="mercator")

chart = (base_layer + detour_layer).properties(title="Trajet (GTFS) vs Détour (TripModifications)",
                                               width="container", height=520)
st.altair_chart(chart, use_container_width=True)

# Aperçu brut
with st.expander("Aperçu (résumé du message)"):
    st.json({
        "service_dates": list(tm_msg.service_dates),
        "selected_trips_count": len(tm_msg.selected_trips),
        "modifications_count": len(tm_msg.modifications)
    })
