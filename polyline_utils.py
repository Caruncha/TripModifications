# polyline_utils.py
import re
from typing import List, Tuple

def _sanitize_polyline(s: str) -> str:
    if not s:
        return s
    s = s.strip()
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
        import polyline as pl  # type: ignore
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