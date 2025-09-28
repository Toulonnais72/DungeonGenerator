from typing import Sequence, Tuple, List
import math
from collections import defaultdict


def _poly_length(poly: Sequence[Tuple[int, int]]) -> float:
    """
    Calcule la longueur totale d'une polyligne.
    """
    return sum(
        math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
        for i in range(len(poly) - 1)
    )


def _merge_wall_segments(
    polys: Sequence[Sequence[Tuple[int, int]]],
    snap: int
) -> List[List[Tuple[int, int]]]:
    """
    Fusionne les segments voisins issus du marching squares
    en polylignes continues (évite les petits traits isolés aux coins).
    """

    def q(p: Tuple[int, int]) -> Tuple[int, int]:
        # On "snape" les coordonnées sur une grille pour éviter les erreurs d'arrondi
        return (
            int(round(p[0] / snap) * snap),
            int(round(p[1] / snap) * snap)
        )

    # Conversion en segments normalisés
    segs = []
    for s in polys:
        if len(s) >= 2:
            segs.append((q(s[0]), q(s[-1])))

    # Construction de la liste d'adjacence
    adj = defaultdict(list)  # point -> [(idx, autre_point)]
    for i, (a, b) in enumerate(segs):
        adj[a].append((i, b))
        adj[b].append((i, a))

    used = [False] * len(segs)
    merged: List[List[Tuple[int, int]]] = []

    for i in range(len(segs)):
        if used[i]:
            continue

        a, b = segs[i]
        path = [a, b]
        used[i] = True

        # On étend la polyligne à ses deux extrémités
        extended = True
        while extended:
            extended = False
            for end in (0, 1):  # 0 = tête, 1 = queue
                node = path[0] if end == 0 else path[-1]
                for j, other in list(adj[node]):
                    if used[j]:
                        continue
                    u, v = segs[j]
                    nxt = v if u == node else (u if v == node else None)
                    if nxt is None:
                        continue
                    if end == 0:
                        path.insert(0, nxt)
                    else:
                        path.append(nxt)
                    used[j] = True
                    extended = True
        merged.append(path)

    return merged
