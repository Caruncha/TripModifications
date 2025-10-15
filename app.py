# -*- coding: utf-8 -*-
# Streamlit - Validation et visualisation d'un fichier TripModifications (GTFS-rt) + GTFS
# Par M365 Copilot - Exemple

import io
import json
import re
import zipfile
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pandas as pd
import altair as alt

# Bindings officiels GTFS-rt -> inclut TripModifications, Shape, ReplacementStop, etc.
# cf. proto officiel et ajout de TripModifications (expérimental).
# Réf : https://gtfs.org/documentation/realtime/proto/ ; https://gtfs.org/documentation/realtime/feed-entities/trip-modifications/
from google.transit import gtfs_realtime_pb2  # type: ignore

st.set_page_config(page_title="Validation TripModifications (GTFS-rt) + Carte des détours", layout="wide")


# -----------------------------
# Helpers GTFS
# -----------------------------
def read_gtfs_tables(gtfs_zip: bytes) -> Dict[str, pd.DataFrame]:
    with zipfile.ZipFile(io.BytesIO(gtfs_zip)) as z:
        def rd(name, dtype=None):
            with z.open(name) as f:
                return pd.read_csv(f, dtype=dtype)
        trips = rd("trips.txt", dtype=str)
        routes = rd("routes.txt", dtype=str)
        stops  = rd("stops.txt",  dtype=str)
        # shapes peut être volumineux ; caster les numériques
        shapes = rd("shapes.txt", dtype={"shape_id": str, "shape_pt_lat": float, "shape_pt_lon": float, "shape_pt_sequence": int})
    shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"])
    return {"trips": trips, "routes": routes, "stops": stops, "shapes": shapes}


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


# -----------------------------
# Helpers Geo / Altair
# -----------------------------
def decode_google_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    coords = []
    index, lat, lon = 0, 0, 0
    length = len(polyline_str)

    def _decode():
        nonlocal index
        result = 0
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
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
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}


def fc(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


# -----------------------------
# Validation TripModifications (réf GTFS-rt)
# -----------------------------
DATE_RE = re.compile(r"^\d{8}$")  # YYYYMMDD

def parse_trip_modifications(pb_bytes: bytes) -> gtfs_realtime_pb2.TripModifications:
    """Parse le binaire en message TripModifications (proto officiel)."""
    msg = gtfs_realtime_pb2.TripModifications()
    msg.ParseFromString(pb_bytes)  # échoue si fichier invalide/structure différente
    return msg


def basic_validation(tm: gtfs_realtime_pb2.TripModifications) -> List[str]:
    """
    Contrôles sémantiques minimaux selon la doc Trip Modifications (expérimental) :
      - selected_trips non vide (liste des trips ciblés)
      - service_dates au format YYYYMMDD
      - modifications non vide
    Voir : https://gtfs.org/documentation/realtime/feed-entities/trip-modifications/
    """
    issues = []

    if len(tm.selected_trips) == 0:
        issues.append("selected_trips est vide : aucun trip ciblé.")

    if len(tm.service_dates) == 0:
        issues.append("service_dates est vide : aucune date de validité déclarée.")
    else:
        bad = [d for d in tm.service_dates if not DATE_RE.match(d)]
        if bad:
            issues.append(f"service_dates mal formatés (YYYYMMDD attendu) : {', '.join(bad)}")

    if len(tm.modifications) == 0:
        issues.append("modifications est vide : aucune modification/détour fourni.")

    # Rappel : TripModifications est expérimental dans GTFS-rt.
    # Les producteurs devraient limiter aux détours proches (≈ semaine à venir).
    # (On ne force pas cette règle, mais on l'affiche comme recommandation.)
    return issues


def infer_trip_ids_from_selected_trips(tm: gtfs_realtime_pb2.TripModifications, trips: pd.DataFrame) -> List[str]:
    """
    Essaie de récupérer des trip_id ciblés :
      - si SelectedTrips expose un repeated trip_ids (implémentation courante), on l'utilise
      - sinon, on tente route_id/direction_id comme fallback (sélection large)
    NB : SelectedTrips/Modification restant expérimentaux, on fait un best-effort.
    """
    trip_ids = set()

    for sel in tm.selected_trips:
        # Heuristiques de champ (sans casser si le champ n'existe pas dans la version courante)
        if hasattr(sel, "trip_ids") and len(sel.trip_ids) > 0:
            for tid in sel.trip_ids:
                trip_ids.add(str(tid))
        else:
            # Fallback par route/direction
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
      1) si modif.shape.encoded_polyline est présent -> on le décode (réf 'Shape')
      2) sinon, si modif.replacement_stops (ReplacementStop) -> on chaîne les stop_id
    """
    # 1) Shape encodé (message 'Shape' expérimental dans GTFS-rt)
    if hasattr(modif, "shape") and modif.HasField("shape"):
        shp = modif.shape
        if hasattr(shp, "encoded_polyline") and shp.encoded_polyline:
            return decode_google_polyline(shp.encoded_polyline)

        # Alternative : liste de points shape_pt_lat/lon si existants (rare côté RT)
        if hasattr(shp, "points") and len(shp.points) > 0:
            coords = [(p.lon, p.lat) for p in shp.points]
            if coords:
                return coords

    # 2) Replacement stops -> on trace entre arrêts de remplacement
    if hasattr(modif, "replacement_stops") and len(modif.replacement_stops) > 0:
        # Ordre approximatif : travel_time_to_stop croissant si disponible
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


# -----------------------------
# UI
# -----------------------------
st.title("Validation d’un fichier TripModifications (GTFS‑realtime) + visualisation des détours")

with st.sidebar:
    st.header("Importer les fichiers")
    gtfs_file = st.file_uploader("GTFS (.zip)", type=["zip"])
    pb_file   = st.file_uploader("TripModifications (.pb)", type=["pb", "bin", "pbf", "dat"])

    st.markdown("---")
    base_color   = st.color_picker("Couleur du trajet de base", "#888888")
    detour_color = st.color_picker("Couleur du détour", "#d62728")

if not gtfs_file or not pb_file:
    st.info("Charge un **GTFS (.zip)** et un **Protobuf TripModifications (.pb)** pour démarrer.")
    st.stop()

# 1) Lecture GTFS
try:
    gtfs = read_gtfs_tables(gtfs_file.read())
    trips_df  = gtfs["trips"]
    routes_df = gtfs["routes"]
    stops_df  = gtfs["stops"]
    shapes_df = gtfs["shapes"]
except Exception as e:
    st.error(f"Erreur lors de la lecture du GTFS : {e}")
    st.stop()

# 2) Parsing TripModifications
try:
    tm = parse_trip_modifications(pb_file.read())
except Exception as e:
    st.error(f"Le fichier fourni n'est pas un message TripModifications valide : {e}")
    st.stop()

# 3) Validation sémantique minimale
issues = basic_validation(tm)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Contrôles de conformité (TripModifications)")
    if issues:
        for it in issues:
            st.error("• " + it)
    else:
        st.success("Structure minimale OK (selected_trips, service_dates, modifications).")

with col2:
    st.subheader("Résumé")
    st.write(f"- selected_trips : **{len(tm.selected_trips)}**")
    st.write(f"- service_dates : **{len(tm.service_dates)}**")
    st.write(f"- modifications : **{len(tm.modifications)}**")
    st.caption("Trip Modifications est un champ **expérimental** du GTFS‑rt et peut évoluer. Cf. référence officielle.")  # info

# 4) Identifier les trip_id(s) impactés (pour tracer le trajet de base)
impacted_trip_ids = infer_trip_ids_from_selected_trips(tm, trips_df)
if not impacted_trip_ids:
    st.warning("Aucun trip_id explicitement déduit de selected_trips. "
               "Le tracé du trajet de base ne sera disponible que si un trip_id précis est fourni.")
else:
    st.info(f"Trip(s) impacté(s) détecté(s) : {', '.join(impacted_trip_ids[:10]) + ('…' if len(impacted_trip_ids)>10 else '')}")

# 5) Construire la liste d’items cartographiables (une carte par modification)
detour_items = []
for idx, modif in enumerate(tm.modifications):
    detour_geom = extract_detour_geometry(modif, stops_df)
    # on associe un trip_id "représentatif" pour le tracé de base (le premier dispo)
    base_trip_id = impacted_trip_ids[0] if impacted_trip_ids else None
    detour_items.append({"index": idx, "base_trip_id": base_trip_id, "geom": detour_geom})

if len(detour_items) == 0:
    st.warning("Aucune 'modification' exploitable pour la carte. Vérifie que le fichier contient un Shape ou des ReplacementStops.")
    st.stop()

# 6) Sélection de la modification à afficher
labels = [f"Modification #{it['index']} — geom={'oui' if it['geom'] else 'non'} — base_trip_id={it['base_trip_id'] or '—'}"
          for it in detour_items]
sel_label = st.selectbox("Choisir une modification (détour) à visualiser :", labels)
sel_idx = labels.index(sel_label)
sel_item = detour_items[sel_idx]

# 7) Construire les GeoJSON (trajet de base + détour)
features = []

# (a) trajet de base depuis shapes.txt si trip_id identifié
if sel_item["base_trip_id"]:
    sid = shape_id_for_trip(trips_df, sel_item["base_trip_id"])
    if sid:
        base_coords = polyline_for_shape(shapes_df, sid)
        if base_coords:
            features.append(feature_line(base_coords, {"kind": "base", "shape_id": sid}))
    else:
        st.warning("shape_id introuvable pour le trip_id représentatif — affichage du détour seul.")

# (b) détour (shape encodé RT ou trajet via ReplacementStops)
if sel_item["geom"]:
    features.append(feature_line(sel_item["geom"], {"kind": "detour", "mod_index": sel_item["index"]}))
else:
    st.warning("Cette modification ne contient pas de géométrie de détour exploitable (Shape / ReplacementStops).")

if not features:
    st.error("Aucune géométrie à afficher.")
    st.stop()

# 8) Carte Altair
data_values = json.loads(json.dumps(fc(features)))["features"]

base_layer = alt.Chart(alt.Data(values=[f for f in data_values if f["properties"]["kind"] == "base"])) \
    .mark_geoshape(fill=None, stroke=st.session_state.get("base_color", base_color), strokeWidth=2) \
    .encode(tooltip=[alt.Tooltip("properties.shape_id:N", title="shape_id")]) \
    .project(type="mercator")

detour_layer = alt.Chart(alt.Data(values=[f for f in data_values if f["properties"]["kind"] == "detour"])) \
    .mark_geoshape(fill=None, stroke=st.session_state.get("detour_color", detour_color), strokeDash=[6, 4], strokeWidth=3) \
    .encode(tooltip=[alt.Tooltip("properties.mod_index:N", title="modification")]) \
    .project(type="mercator")

chart = (base_layer + detour_layer).properties(title="Trajet (GTFS) vs Détour (TripModifications)", width="container", height=520)
st.altair_chart(chart, use_container_width=True)

# 9) Aperçu brut (debug)
with st.expander("Aperçu brut du message TripModifications"):
    # On montre les principaux champs pour inspection
    preview = {
        "service_dates": list(tm.service_dates),
        "selected_trips_count": len(tm.selected_trips),
        "modifications_count": len(tm.modifications)
    }
    st.json(preview)
