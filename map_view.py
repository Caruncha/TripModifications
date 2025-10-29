# map_view.py
from typing import List, Tuple, Optional
import folium

from tripmodifications.config import (
    COLOR_DETOUR, COLOR_ORIGINAL, COLOR_REPLACEMENT,
    COLOR_ADDED, COLOR_CANCELED, DEFAULT_CENTER, DEFAULT_ZOOM
)


def build_folium_map_for_polyline(
    poly: List[Tuple[float, float]],
    shape_id: Optional[str] = None,
    replacement_stop_points: Optional[List[Tuple[float, float, str]]] = None,
    original_poly: Optional[List[Tuple[float, float]]] = None,
    original_stop_points: Optional[List[Tuple[float, float, str]]] = None,
    original_shape_id: Optional[str] = None,
    added_segments: Optional[List[List[Tuple[float, float]]]] = None,
    canceled_segments: Optional[List[List[Tuple[float, float]]]] = None,
    center: Tuple[float, float] = DEFAULT_CENTER,
    zoom_start: int = DEFAULT_ZOOM
):
    def _valid_ll(la, lo) -> bool:
        return (la is not None and lo is not None
                and -90 <= la <= 90 and -180 <= lo <= 180
                and not (abs(la) < 1e-8 and abs(lo) < 1e-8))
    if not poly or len(poly) < 2:
        return None
    latlons_poly = [(la, lo) for la, lo in poly if _valid_ll(la, lo)]
    if len(latlons_poly) < 2:
        return None

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap", control_scale=True, min_zoom=8)

    # Détour (rouge)
    folium.PolyLine(latlons_poly, color=COLOR_DETOUR, weight=5, opacity=0.9,
                    tooltip=f"shape_id (détour): {shape_id or 'n/a'}").add_to(m)
    folium.CircleMarker(latlons_poly[0], radius=6, color="green", fill=True, fill_opacity=0.9,
                        tooltip="Départ du détour").add_to(m)
    folium.CircleMarker(latlons_poly[-1], radius=6, color="red", fill=True, fill_opacity=0.9,
                        tooltip="Arrivée du détour").add_to(m)

    # Segments ajoutés/annulés
    all_ext = []

    def _draw_segments(segments, color, label):
        nonlocal all_ext
        if not segments: return
        for seg in segments:
            seg_ll = [(la, lo) for la, lo in seg if _valid_ll(la, lo)]
            if len(seg_ll) >= 2:
                folium.PolyLine(seg_ll, color=color, weight=6, opacity=0.95,
                                tooltip=f"{label} (shape_id: {shape_id or 'n/a'})").add_to(m)
                all_ext.extend(seg_ll)

    _draw_segments(added_segments, COLOR_ADDED, "Ajouté")
    _draw_segments(canceled_segments, COLOR_CANCELED, "Annulé")

    # Tracé originel
    if original_poly and len(original_poly) >= 2:
        latlons_orig = [(la, lo) for la, lo in original_poly if _valid_ll(la, lo)]
        if len(latlons_orig) >= 2:
            folium.PolyLine(latlons_orig, color=COLOR_ORIGINAL, weight=4, opacity=0.85,
                            tooltip=f"Tracé originel (shapes.txt): {original_shape_id or 'n/a'}").add_to(m)

    # Arrêts originels
    if original_stop_points:
        n = len(original_stop_points)
        for idx, (la, lo, lab) in enumerate(original_stop_points):
            if not _valid_ll(la, lo): continue
            if idx == 0:
                color = "green"; fill = "green"; radius = 7
            elif idx == n-1:
                color = "red"; fill = "red"; radius = 7
            else:
                color = "#666666"; fill = "#ffffff"; radius = 5
            folium.CircleMarker((la, lo), radius=radius, color=color, fill=True,
                                fill_color=fill, fill_opacity=0.95, weight=2,
                                tooltip=str(lab or "stop_id")).add_to(m)

    # Replacement stops (rose)
    if replacement_stop_points:
        for la, lo, lab in replacement_stop_points:
            if _valid_ll(la, lo):
                folium.CircleMarker((la, lo), radius=7, color=COLOR_REPLACEMENT,
                                    fill=True, fill_color=COLOR_REPLACEMENT, fill_opacity=0.95,
                                    weight=2, tooltip=lab or "Arrêt de remplacement").add_to(m)

    # Emprise stricte (incluant segments)
    pts = latlons_poly + all_ext if all_ext else latlons_poly
    lats = [la for la, _ in pts]
    lons = [lo for _, lo in pts]
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
