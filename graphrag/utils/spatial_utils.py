"""
Spatial verbalization — convert raw coordinates to natural language.

Instead of sending raw JSON/arrays of coordinates to Gemini
(which wastes tokens), we pre-compute distances and directions
locally and send a compact Turkish text summary.

Example:
    Raw  (≈60 tokens):
        {"latitude": 41.00861, "longitude": 28.98029}
        {"latitude": 41.00553, "longitude": 28.97693}
    Verbalized (≈15 tokens):
        "Sultanahmet Camii, Ayasofya'nın 400m güneybatısındadır"
"""

import logging
from math import radians, sin, cos, sqrt, atan2
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Math
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two WGS-84 points."""
    R = 6_371_000  # Earth radius in metres
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(Δλ / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def cardinal_direction_tr(lat1: float, lon1: float,
                          lat2: float, lon2: float) -> str:
    """Return Turkish cardinal direction from point-1 → point-2."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    if abs(dlat) > abs(dlon) * 2:
        return "kuzey" if dlat > 0 else "güney"
    if abs(dlon) > abs(dlat) * 2:
        return "doğu" if dlon > 0 else "batı"

    ns = "kuzey" if dlat > 0 else "güney"
    ew = "doğu" if dlon > 0 else "batı"
    return f"{ns}{ew}"


def format_distance(metres: float) -> str:
    """Human-friendly Turkish distance string."""
    if metres < 1:
        return "aynı noktada"
    if metres < 100:
        return f"{metres:.0f}m"
    if metres < 1000:
        return f"yaklaşık {round(metres / 10) * 10:.0f}m"
    return f"yaklaşık {metres / 1000:.1f}km"


# ---------------------------------------------------------------------------
# Coordinate extraction helper
# ---------------------------------------------------------------------------

def extract_coords(props: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Safely extract (lat, lon) from a properties dict."""
    lat = props.get("latitude")
    lon = props.get("longitude")
    if lat is not None and lon is not None:
        try:
            return (float(lat), float(lon))
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Main verbalisation functions
# ---------------------------------------------------------------------------

def verbalize_entity_location(entity_name: str,
                              props: Dict[str, Any]) -> Optional[str]:
    """
    Convert a single entity's spatial properties to a short text.

    Returns None if the entity has no coordinates.
    """
    coords = extract_coords(props)
    if coords is None:
        return None

    parts = [f"{entity_name}: {coords[0]:.5f}°K, {coords[1]:.5f}°D"]

    loc = props.get("location") or props.get("district")
    if loc:
        parts[0] += f" ({loc})"

    return parts[0]


def verbalize_pair_relation(name_a: str, props_a: Dict[str, Any],
                            name_b: str, props_b: Dict[str, Any]) -> Optional[str]:
    """
    Produce a compact Turkish sentence describing the spatial relation
    between two entities.

    >>> verbalize_pair_relation(
    ...     "Sultanahmet Camii", {"latitude": 41.00553, "longitude": 28.97693},
    ...     "Ayasofya",         {"latitude": 41.00861, "longitude": 28.98029})
    "Sultanahmet Camii, Ayasofya'nın yaklaşık 400m güneybatısındadır"
    """
    ca = extract_coords(props_a)
    cb = extract_coords(props_b)
    if ca is None or cb is None:
        return None

    dist = haversine(ca[0], ca[1], cb[0], cb[1])
    direction = cardinal_direction_tr(cb[0], cb[1], ca[0], ca[1])
    dist_str = format_distance(dist)

    return f"{name_a}, {name_b}'nın {dist_str} {direction}ındadır"


def verbalize_spatial_context(entities: Dict[str, Dict[str, Any]]) -> str:
    """
    Given a dict of {entity_name: properties}, produce a full spatial
    summary paragraph.  This replaces raw coordinate JSON in the LLM prompt.

    Args:
        entities: Mapping of entity names to their property dicts

    Returns:
        Multi-line Turkish spatial summary
    """
    lines: List[str] = []
    names = list(entities.keys())

    # 1. Individual locations
    for name in names:
        loc = verbalize_entity_location(name, entities[name])
        if loc:
            lines.append(loc)

    # 2. Pair-wise relations (only for small sets)
    if 2 <= len(names) <= 5:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rel = verbalize_pair_relation(
                    names[i], entities[names[i]],
                    names[j], entities[names[j]]
                )
                if rel:
                    lines.append(rel)

    return "\n".join(lines) if lines else ""
