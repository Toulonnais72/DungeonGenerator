# helpers.py
from typing import Sequence, Tuple, List
import math
from collections import defaultdict

Point = Tuple[int, int]

def _poly_length(poly: Sequence[Point]) -> float:
    """Longueur totale d'un polyline (somme des segments)."""
    return sum(
        math.hypot(poly[i+1][0] - poly[i][0], poly[i+1][1] - poly[i][1])
        for i in range(len(poly) - 1)
    )

def _merge_wall_segments(
    polys: Sequence[Sequence[Point]], snap: int = 4
) -> List[List[Point]]:
    """
    Fusionne des petits segments (souvent axis-alignés) en segments plus longs.
    On 'snap' sur une grille, on regroupe horizontaux/verticaux et on fusionne
    les intervalles qui se touchent ou se chevauchent.
    """
    def _snap(p: Point) -> Point:
        if snap <= 0:
            return p
        return (round(p[0]/snap)*snap, round(p[1]/snap)*snap)

    horiz = defaultdict(list)  # key = y, value = list[(x1, x2)]
    vert  = defaultdict(list)  # key = x, value = list[(y1, y2)]

    for seg in polys:
        if len(seg) < 2:
            continue
        a = _snap(seg[0]); b = _snap(seg[-1])
        # on décide orientation selon la plus grande variation
        if abs(a[1]-b[1]) <= abs(a[0]-b[0]):  # horizontal
            y = a[1]
            x1, x2 = sorted((a[0], b[0]))
            if x1 != x2:
                horiz[y].append((x1, x2))
        else:                                  # vertical
            x = a[0]
            y1, y2 = sorted((a[1], b[1]))
            if y1 != y2:
                vert[x].append((y1, y2))

    def merge_intervals(iv: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not iv:
            return []
        iv.sort()
        merged = [list(iv[0])]
        for s, e in iv[1:]:
            if s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        return [tuple(x) for x in merged]

    result: List[List[Point]] = []
    for y, lst in horiz.items():
        for x1, x2 in merge_intervals(lst):
            result.append([(x1, y), (x2, y)])
    for x, lst in vert.items():
        for y1, y2 in merge_intervals(lst):
            result.append([(x, y1), (x, y2)])
    return result
