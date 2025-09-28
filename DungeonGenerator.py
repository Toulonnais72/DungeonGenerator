import datetime
import math
import os
import random
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pygame
import streamlit as st
from PIL import Image
import helpers as hp

from symbols import SymbolLibrary

# --- Palette 'parchemin' (tons chauds) ---------------------------------------
INK_BROWN        = (45, 30, 20)            # contours/murs
ROOM_FILL_RGBA   = (245, 230, 200, 90)     # remplissage salles (alpha)
CORRIDOR_FILL_RGBA = (242, 226, 195, 70)   # remplissage couloirs (alpha)
GRID_COLOR       = (190, 170, 140)         # quadrillage ocre
BACKGROUND_COLOR = (242, 228, 206)         # beige de base (fallback si pas d'image)
VIGNETTE_COLOR   = (0, 0, 0)               # vignette noire -> SUB pour assombrir





DOOR_LABELS = {
    "door": "door",
    "locked": "locked door",
    "secret": "secret door",
    "trapped": "trapped door",
}

class FontStore:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, bool], pygame.font.Font] = {}

    def get(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._cache:
            self._cache[key] = pygame.font.SysFont("timesnewroman", size, bold=bold)
        return self._cache[key]

FONT_STORE = FontStore()

# --- Dataclasses --------------------------------------------------------------
@dataclass
class DungeonSettings:
    canvas_width: int = 1600
    canvas_height: int = 900
    tilesize: int = 15
    vignette_strength: float = 0.25
    n_rooms: int = 28
    min_room_width: int = 6
    max_room_width: int = 18
    min_room_height: int = 6
    max_room_height: int = 16
    margin_tiles: int = 2
    max_placement_tries: int = 2000
    min_room_spacing: int = 1
    allow_overlap: bool = False
    corridor_width_tiles: Tuple[int, int] = (1, 2)
    door_distribution: Tuple[Tuple[str, float], ...] = field(
        default_factory=lambda: (
            ("door",   0.7),
            ("locked", 0.15),
            ("secret", 0.1),
            ("trapped",0.05),
        )
    )
    trap_room_ratio: float = 0.12
    treasure_room_ratio: float = 0.10
    column_room_ratio: float = 0.18
    parchment_path: Optional[str] = "images/parchment.jpg"
    image_folder: str = "images"
    export_with_timestamp: bool = True
    write_to_disk: bool = False
    output_directory: str = "."
    seed: Optional[Union[int, str]] = None

    def corridor_width_range(self) -> Tuple[int, int]:
        lo, hi = self.corridor_width_tiles
        if lo > hi:
            lo, hi = hi, lo
        return max(1, lo), max(1, hi)

@dataclass
class DungeonResult:
    image_bytes: bytes
    filename: str
    legend_lines: List[str]
    room_contents: Dict[int, List[str]]
    settings: DungeonSettings
    seed_used: Optional[Union[int, str]] = None

# --- Utils -------------------------------------------------------------------
def wall_thickness(settings: DungeonSettings) -> int:
    base = max(3, settings.tilesize // 3)
    return base if base % 2 == 1 else base + 1

def ensure_pygame_ready() -> None:
    if not pygame.get_init():
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    if pygame.display.get_surface() is None:
        flags = getattr(pygame, "HIDDEN", 0)
        try:
            pygame.display.set_mode((1, 1), flags)
        except pygame.error:
            pygame.display.set_mode((1, 1))

def normalize_door_percentages(raw: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    order = ("door", "locked", "secret", "trapped")
    total = sum(max(0.0, raw.get(k, 0.0)) for k in order)
    if total <= 0:
        return (("door", 1.0), ("locked", 0.0), ("secret", 0.0), ("trapped", 0.0))
    return tuple((k, max(0.0, raw.get(k, 0.0)) / total) for k in order)

def weighted_choice(rng: random.Random, weighted_items: Sequence[Tuple[str, float]]) -> str:
    total = sum(w for _, w in weighted_items) or 1.0
    r, acc = rng.random()*total, 0.0
    for item, w in weighted_items:
        acc += w
        if r <= acc:
            return item
    return weighted_items[-1][0]


def make_layers(size: Tuple[int, int]) -> Dict[str, pygame.Surface]:
    w, h = size
    return {
        "background": pygame.Surface((w, h)).convert(),                 # sans alpha
        "fills":      pygame.Surface((w, h), pygame.SRCALPHA),          # salles/couloirs
        "walls":      pygame.Surface((w, h), pygame.SRCALPHA),          # murs/traits
        "grid":       pygame.Surface((w, h), pygame.SRCALPHA),          # quadrillage
        "symbols":    pygame.Surface((w, h), pygame.SRCALPHA),          # portes, escaliers...
        "fx":         pygame.Surface((w, h), pygame.SRCALPHA),          # vignette & co
    }


def draw_background_layer(bg: pygame.Surface, parchment_path: Optional[str]) -> None:
    """Remplit le fond : d'abord un beige, puis le parchemin s'il existe (blit normal)."""
    bg.fill(BACKGROUND_COLOR)
    if parchment_path and os.path.exists(parchment_path):
        tex = pygame.image.load(parchment_path).convert()
        tex = pygame.transform.smoothscale(tex, bg.get_size())
        # blit normal : on ne multiplie pas, on ne soustrait pas → respecte les couleurs
        bg.blit(tex, (0, 0))



def draw_room_fill(fills: pygame.Surface,
                   walls: pygame.Surface,
                   rect: pygame.Rect,
                   number: Optional[int],
                   rng: random.Random) -> None:
    # fill (alpha normal)
    pygame.draw.rect(fills, ROOM_FILL_RGBA, rect)

    # petite patine douce en alpha (pas de MULT)
    inner = rect.inflate(-max(2, rect.width // 20), -max(2, rect.height // 20))
    if inner.width > 10 and inner.height > 10:
        shade = pygame.Surface((inner.width, inner.height), pygame.SRCALPHA)
        for _ in range(3):
            r = rng.randint(min(inner.width, inner.height)//4, max(inner.width, inner.height))
            cx, cy = rng.randint(0, inner.width), rng.randint(0, inner.height)
            pygame.draw.circle(shade, (230, 210, 180, 35), (cx, cy), r)
        fills.blit(shade, inner.topleft)

    # contour du rectangle (mur “local” fin)
    pygame.draw.rect(walls, INK_BROWN, rect, width=1)
    # numéro au centre (facultatif) → on le met sur les murs/symbols pour qu'il reste lisible
    if number is not None:
        font = FONT_STORE.get(16, bold=False)
        img = font.render(str(number), True, INK_BROWN)
        walls.blit(img, (rect.centerx - img.get_width() // 2,
                         rect.centery - img.get_height() // 2))


def draw_corridor_fill(fills: pygame.Surface,
                       walls: pygame.Surface,
                       rect: pygame.Rect) -> pygame.Rect:
    pygame.draw.rect(fills, CORRIDOR_FILL_RGBA, rect)
    pygame.draw.rect(walls, INK_BROWN, rect, width=1)
    return rect.inflate(-2, -2)


def corridor_between_layers(fills: pygame.Surface,
                            walls: pygame.Surface,
                            settings: DungeonSettings,
                            rng: random.Random,
                            rect_a: pygame.Rect,
                            rect_b: pygame.Rect) -> List[pygame.Rect]:
    ax, ay = rect_a.center
    bx, by = rect_b.center
    lo, hi = settings.corridor_width_range()
    w_tiles = rng.randint(lo, hi)
    wpx = max(1, w_tiles) * settings.tilesize

    def snap(v: int) -> int:
        return (v // settings.tilesize) * settings.tilesize + settings.tilesize // 2

    midx, ay, by = snap(bx), snap(ay), snap(by)
    inners: List[pygame.Rect] = []

    x1, x2 = min(ax, midx), max(ax, midx)
    hrect = pygame.Rect(x1 - wpx // 2, ay - wpx // 2, (x2 - x1) + wpx, wpx)
    inners.append(draw_corridor_fill(fills, walls, hrect))

    y1, y2 = min(ay, by), max(ay, by)
    vrect = pygame.Rect(midx - wpx // 2, y1 - wpx // 2, wpx, (y2 - y1) + wpx)
    inners.append(draw_corridor_fill(fills, walls, vrect))
    return inners


def place_doors_for_connection_layers(symbols_surf: pygame.Surface,
                                      room_rect: pygame.Rect,
                                      corridor_inners: Sequence[pygame.Rect],
                                      room_contents: Dict[int, List[str]],
                                      room_idx: int,
                                      symbols: SymbolLibrary,
                                      door_distribution: Sequence[Tuple[str, float]],
                                      rng: random.Random) -> None:
    # inchangé sauf la cible de dessin → symbols_surf
    best = None
    best_area = -1
    for c in corridor_inners:
        if not room_rect.colliderect(c):
            continue
        clip = room_rect.clip(c)
        area = clip.width * clip.height
        if area > best_area:
            best_area = area
            best = (clip, c)
    if not best:
        return
    _, corridor = best
    door_symbol = weighted_choice(rng, door_distribution)
    if corridor.width >= corridor.height:
        cy = int(max(room_rect.top, min(room_rect.bottom, corridor.centery)))
        cx = room_rect.left if corridor.centerx < room_rect.centerx else room_rect.right
    else:
        cx = int(max(room_rect.left, min(room_rect.right, corridor.centerx)))
        cy = room_rect.top if corridor.centery < room_rect.centery else room_rect.bottom
    # → on dessine sur la couche symbols
    symbols.draw(symbols_surf, door_symbol, cx, cy, scale=0.7)
    room_contents.setdefault(room_idx, []).append(DOOR_LABELS.get(door_symbol, door_symbol))

def draw_grid_layer(grid: pygame.Surface, tilesize: int) -> None:
    w, h = grid.get_size()
    color = (*GRID_COLOR, 55)  # léger
    for x in range(0, w + 1, tilesize):
        pygame.draw.line(grid, color, (x, 0), (x, h))
    for y in range(0, h + 1, tilesize):
        pygame.draw.line(grid, color, (0, y), (w, y))


def draw_global_walls(walls: pygame.Surface,
                      wall_polys: Sequence[Sequence[Tuple[int, int]]],
                      thickness: int) -> None:
    if not wall_polys:
        return
    inner = max(1, thickness // 2)
    for poly in wall_polys:
        if len(poly) < 2:
            continue
        pygame.draw.lines(walls, INK_BROWN, False, poly, thickness)
        pygame.draw.lines(walls, INK_BROWN, False, poly, inner)


def apply_vignette_fx(fx: pygame.Surface, strength: float = 0.35) -> None:
    w, h = fx.get_size()
    base = 256
    grad = pygame.Surface((base, base), pygame.SRCALPHA)
    c = base / 2
    maxd = math.hypot(c, c)
    for y in range(base):
        for x in range(base):
            d = math.hypot(x - c, y - c) / maxd
            a = int((d ** 2.2) * 255 * strength)  # noir + alpha
            grad.set_at((x, y), (0, 0, 0, a))
    vignette = pygame.transform.smoothscale(grad, (w, h))
    # On soustrait (assombrissement) → pas de dérive bleue/verte
    fx.blit(vignette, (0, 0))



##########################################################################################


def draw_grid(surface: pygame.Surface, settings: DungeonSettings) -> None:
    width, height = surface.get_size()
    ts = max(4, settings.tilesize)
    grid = pygame.Surface((width, height), pygame.SRCALPHA)
    color = (180, 160, 120, 35)
    for x in range(0, width + 1, ts):
        pygame.draw.line(grid, color, (x, 0), (x, height))
    for y in range(0, height + 1, ts):
        pygame.draw.line(grid, color, (0, y), (width, y))
    surface.blit(grid, (0, 0))

# --- Sol des pièces & couloirs ------------------------------------------------
def draw_room_grid(surface: pygame.Surface, rect: pygame.Rect, settings: DungeonSettings) -> None:
    ts = settings.tilesize
    if rect.width <= ts or rect.height <= ts:
        return
    grid = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    color = (200, 175, 140, 40)
    for x in range(ts, rect.width, ts):
        pygame.draw.line(grid, color, (x, 0), (x, rect.height))
    for y in range(ts, rect.height, ts):
        pygame.draw.line(grid, color, (0, y), (rect.width, y))
    surface.blit(grid, rect.topleft)

def apply_floor_shading(surface: pygame.Surface, rect: pygame.Rect, rng: random.Random, intensity: float = 0.55) -> None:
    if rect.width <= 0 or rect.height <= 0:
        return
    shading = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    passes = max(4, int(6 * intensity))
    for _ in range(passes):
        radius = rng.randint(max(6, min(rect.width, rect.height) // 3), max(rect.width, rect.height))
        cx = rng.randint(0, rect.width)
        cy = rng.randint(0, rect.height)
        alpha = rng.randint(18, 45)
        tint  = rng.randint(-12, 12)
        color = (
    max(180, min(255, 240 + tint)),   # base beige chaud
    max(160, min(255, 210 + tint)),
    max(130, min(255, 180 + tint)),
    alpha)

        pygame.draw.circle(shading, color, (cx, cy), radius)
    surface.blit(shading, rect.topleft)


# --- Placement & graph --------------------------------------------------------
def rects_intersect(a: pygame.Rect, b: pygame.Rect, spacing_px: int = 0) -> bool:
    if spacing_px <= 0:
        return a.colliderect(b)
    expanded = pygame.Rect(b.x - spacing_px, b.y - spacing_px, b.width + 2 * spacing_px, b.height + 2 * spacing_px)
    return a.colliderect(expanded)

def generate_rooms(settings: DungeonSettings, rng: random.Random) -> List[pygame.Rect]:
    rooms: List[pygame.Rect] = []
    width_tiles  = settings.canvas_width  // settings.tilesize
    height_tiles = settings.canvas_height // settings.tilesize
    max_w_tiles = width_tiles -  2 * settings.margin_tiles
    max_h_tiles = height_tiles - 2 * settings.margin_tiles
    if max_w_tiles <= 0 or max_h_tiles <= 0:
        return rooms

    tries = 0
    spacing_px = settings.min_room_spacing * settings.tilesize
    while len(rooms) < settings.n_rooms and tries < settings.max_placement_tries:
        tries += 1
        min_w = max(1, settings.min_room_width)
        max_w = min(settings.max_room_width, max_w_tiles)
        min_h = max(1, settings.min_room_height)
        max_h = min(settings.max_room_height, max_h_tiles)
        if min_w > max_w or min_h > max_h:
            break
        w_t = rng.randint(min_w, max_w)
        h_t = rng.randint(min_h, max_h)
        x_min = settings.margin_tiles
        x_max = settings.margin_tiles + max_w_tiles - w_t
        y_min = settings.margin_tiles
        y_max = settings.margin_tiles + max_h_tiles - h_t
        if x_min > x_max or y_min > y_max:
            break
        x_t = rng.randint(x_min, x_max)
        y_t = rng.randint(y_min, y_max)
        rect = pygame.Rect(x_t * settings.tilesize, y_t * settings.tilesize,
                           w_t * settings.tilesize, h_t * settings.tilesize)
        if not settings.allow_overlap and any(rects_intersect(rect, r, spacing_px) for r in rooms):
            continue
        rooms.append(rect)

    if rooms:
        minx = min(r.left for r in rooms);  miny = min(r.top for r in rooms)
        maxx = max(r.right for r in rooms); maxy = max(r.bottom for r in rooms)
        bbox = pygame.Rect(minx, miny, maxx - minx, maxy - miny)
        dx = (settings.canvas_width - bbox.width) // 2 - bbox.left
        dy = (settings.canvas_height - bbox.height) // 2 - bbox.top
        rooms = [r.move(dx, dy) for r in rooms]
    return rooms

def prim_mst(points: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    n = len(points)
    if n <= 1:
        return []
    in_mst = [False] * n
    dist   = [float("inf")] * n
    parent = [-1] * n
    dist[0] = 0
    edges: List[Tuple[int, int]] = []
    for _ in range(n):
        u = -1; best = float("inf")
        for i in range(n):
            if not in_mst[i] and dist[i] < best:
                best = dist[i]; u = i
        if u == -1:
            break
        in_mst[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u))
        ux, uy = points[u]
        for v in range(n):
            if in_mst[v]: continue
            vx, vy = points[v]
            w = (ux - vx)**2 + (uy - vy)**2
            if w < dist[v]:
                dist[v] = w; parent[v] = u
    return edges

def decorate_rooms(surface: pygame.Surface, settings: DungeonSettings, rng: random.Random,
                   rooms: Sequence[pygame.Rect], room_contents: Dict[int, List[str]],
                   symbols: SymbolLibrary) -> None:
    if not rooms:
        return
    ts = settings.tilesize
    indexed_rooms = list(enumerate(rooms, start=1))

    # Colonnes dans grandes salles
    large_rooms = [(i, r) for i, r in indexed_rooms if r.width >= 12*ts and r.height >= 12*ts]
    if large_rooms and settings.column_room_ratio > 0:
        target = max(1 if settings.column_room_ratio > 0 else 0,
                     min(len(large_rooms), round(settings.column_room_ratio * len(large_rooms))))
        if target:
            for idx, rect in rng.sample(large_rooms, target):
                inner = rect.inflate(-ts * 2, -ts * 2)
                if inner.width <= ts or inner.height <= ts:
                    continue
                placements = rng.randint(2, 4)
                for _ in range(placements):
                    tx = rng.randint(inner.left // ts, (inner.right - 1) // ts)
                    ty = rng.randint(inner.top  // ts, (inner.bottom - 1) // ts)
                    cx = tx * ts + ts // 2; cy = ty * ts + ts // 2
                    symbols.draw(surface, "archway", cx, cy, scale=0.5)
                room_contents[idx].append("columns")

    def marker(label: str, symbol: str, ratio: float) -> None:
        if ratio <= 0:
            return
        target = max(1 if ratio > 0 else 0, min(len(indexed_rooms), round(ratio * len(indexed_rooms))))
        for idx, rect in rng.sample(indexed_rooms, target):
            symbols.draw(surface, symbol, rect.centerx, rect.centery, scale=0.6)
            room_contents[idx].append(label)

    marker("treasure", "portcullis", settings.treasure_room_ratio)
    marker("trap",     "trapped",    settings.trap_room_ratio)

# --- Marching Squares & murs --------------------------------------------------
def generate_grid_from_rooms(rooms: Sequence[pygame.Rect], corridors: Sequence[pygame.Rect],
                             settings: DungeonSettings) -> List[List[int]]:
    ts = settings.tilesize
    width_tiles  = math.ceil(settings.canvas_width  / ts)
    height_tiles = math.ceil(settings.canvas_height / ts)
    grid = [[1 for _ in range(width_tiles)] for _ in range(height_tiles)]

    def fill_rect(rect: pygame.Rect) -> None:
        tx0 = max(0, rect.left // ts);         tx1 = min(width_tiles,  math.ceil(rect.right  / ts))
        ty0 = max(0, rect.top  // ts);         ty1 = min(height_tiles, math.ceil(rect.bottom / ts))
        for ty in range(ty0, ty1):
            for tx in range(tx0, tx1):
                grid[ty][tx] = 0

    for r in rooms:     fill_rect(r)
    for c in corridors: fill_rect(c)
    return grid

def marching_squares(grid: Sequence[Sequence[int]], settings: DungeonSettings) -> List[List[Tuple[int, int]]]:
    h = len(grid); w = len(grid[0]) if h else 0; ts = settings.tilesize
    polys: List[List[Tuple[int, int]]] = []
    for y in range(h - 1):
        for x in range(w - 1):
            state = 0
            if grid[y][x] == 0:         state |= 1
            if grid[y][x + 1] == 0:     state |= 2
            if grid[y + 1][x + 1] == 0: state |= 4
            if grid[y + 1][x] == 0:     state |= 8
            px = x * ts; py = y * ts
            if state in (1, 14):  polys.append([(px, py), (px + ts, py)])
            if state in (2, 13):  polys.append([(px + ts, py), (px + ts, py + ts)])
            if state in (4, 11):  polys.append([(px, py + ts), (px + ts, py + ts)])
            if state in (8, 7):   polys.append([(px, py), (px, py + ts)])
    return polys

def draw_walls(surface: pygame.Surface, polys: Sequence[Sequence[Tuple[int, int]]],
               settings: DungeonSettings, rng: random.Random,
               color: Tuple[int, int, int] = INK_BROWN) -> None:
    if not polys:
        return
    wall_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    outer = wall_thickness(settings); inner = max(1, outer // 2)

    snap = max(1, settings.tilesize // 4)
    merged = hp._merge_wall_segments(polys, snap=snap)
    cleaned = [p for p in merged if hp._poly_length(p) >= settings.tilesize * 1.1]

    for poly in cleaned:
        closed = (len(poly) > 2 and poly[0] == poly[-1])
        pygame.draw.lines(wall_surface, (*color, 235), closed, poly, outer)
        pygame.draw.lines(wall_surface, (*color, 255), closed, poly, inner)

    surface.blit(wall_surface, (0, 0))

# --- Export -------------------------------------------------------------------
def surface_to_png_bytes(surface: pygame.Surface) -> bytes:
    w, h = surface.get_size()
    data  = pygame.image.tostring(surface, "RGBA")
    image = Image.frombytes("RGBA", (w, h), data)
    buf   = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def generate_dungeon(settings: DungeonSettings, symbols: SymbolLibrary) -> DungeonResult:
    """Génère un donjon et compose toutes les couches dans le bon ordre (style parchemin)."""
    ensure_pygame_ready()
    rng = random.Random(settings.seed)

    size = (settings.canvas_width, settings.canvas_height)
    layers = make_layers(size)

    # 1) Fond parchemin
    draw_background_layer(layers["background"], settings.parchment_path)

    # 2) Génération des salles et des arêtes MST
    rooms = generate_rooms(settings, rng)
    centers = [r.center for r in rooms]
    edges = prim_mst(centers)

    room_contents: Dict[int, List[str]] = {i: [] for i in range(1, len(rooms) + 1)}
    corridor_inners: List[pygame.Rect] = []

    # 3) Corridors + portes
    for i, j in edges:
        inners = corridor_between_layers(layers["fills"], layers["walls"], settings, rng, rooms[i], rooms[j])
        corridor_inners.extend(inners)

        place_doors_for_connection_layers(
            layers["symbols"], rooms[i], inners, room_contents, i + 1,
            symbols, settings.door_distribution, rng
        )
        place_doors_for_connection_layers(
            layers["symbols"], rooms[j], inners, room_contents, j + 1,
            symbols, settings.door_distribution, rng
        )

    # 4) Salles
    for idx, room in enumerate(rooms, start=1):
        draw_room_fill(layers["fills"], layers["walls"], room, idx, rng)

    # 5) Murs globaux via marching squares
    grid = generate_grid_from_rooms(rooms, corridor_inners, settings)
    wall_polys = marching_squares(grid, settings)
    draw_global_walls(layers["walls"], wall_polys, thickness=max(3, settings.tilesize // 3))

    # 6) Escaliers
    if rooms:
        if len(rooms) >= 2:
            up_i, down_i = rng.sample(range(len(rooms)), 2)
            symbols.draw(layers["symbols"], "stairs_up", rooms[up_i].centerx, rooms[up_i].centery, scale=0.6)
            symbols.draw(layers["symbols"], "stairs_down", rooms[down_i].centerx, rooms[down_i].centery, scale=0.6)
            room_contents[up_i + 1].append("stairs up")
            room_contents[down_i + 1].append("stairs down")
        else:
            symbols.draw(layers["symbols"], "stairs_up", rooms[0].centerx, rooms[0].centery, scale=0.6)
            room_contents[1].append("stairs up")

    # 7) Quadrillage
    draw_grid_layer(layers["grid"], settings.tilesize)

    # 8) Vignette (fx)
    apply_vignette_fx(layers["fx"], strength=settings.vignette_strength)

    # 9) Composition finale
    composite = pygame.Surface(size).convert_alpha()
    composite.blit(layers["background"], (0, 0))                # parchemin
    composite.blit(layers["fills"], (0, 0))                     # salles/couloirs
    composite.blit(layers["walls"], (0, 0))                     # murs
    composite.blit(layers["grid"], (0, 0))                      # quadrillage
    composite.blit(layers["symbols"], (0, 0))                   # portes, escaliers
    composite.blit(layers["fx"], (0, 0), special_flags=pygame.BLEND_RGBA_SUB)  # vignette assombrissante

    # 10) Export
    image_bytes = surface_to_png_bytes(composite)
    filename = f'dungeon_{datetime.datetime.now():%Y%m%d_%H%M%S}.png'
    legend_lines = draw_legend(composite, settings, room_contents)

    return DungeonResult(
        image_bytes=image_bytes,
        filename=filename,
        legend_lines=legend_lines,
        room_contents=room_contents,
        settings=settings,
        seed_used=settings.seed,
    )


@st.cache_resource(show_spinner=False)
def get_symbol_library(folder: str) -> SymbolLibrary:
    ensure_pygame_ready()
    return SymbolLibrary(folder=folder)

# --- UI Streamlit -------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Dungeon Generator", layout="wide")
    st.title("Dungeon Generator")

    st.sidebar.header("Layout")
    canvas_width  = st.sidebar.number_input("Canvas width (px)",  min_value=600, max_value=3000, value=1400, step=50)
    canvas_height = st.sidebar.number_input("Canvas height (px)", min_value=600, max_value=3000, value=900,  step=50)
    tilesize      = st.sidebar.slider("Tile size (px)", min_value=8, max_value=40, value=15)
    vignette_str  = st.sidebar.slider("Vignette", 0.0, 1.0, 0.35, 0.05)

    st.sidebar.header("Rooms")
    n_rooms         = st.sidebar.slider("Number of rooms", min_value=5, max_value=80, value=28)
    min_room_width  = st.sidebar.slider("Minimum room width (tiles)", 3, 30, 6)
    max_room_width  = st.sidebar.slider("Maximum room width (tiles)", min_room_width, 40, 18)
    min_room_height = st.sidebar.slider("Minimum room height (tiles)", 3, 30, 6)
    max_room_height = st.sidebar.slider("Maximum room height (tiles)", min_room_height, 40, 16)
    margin_tiles    = st.sidebar.slider("Map margin (tiles)", 0, 12, 2)
    min_room_spacing= st.sidebar.slider("Room spacing (tiles)", 0, 6, 1)
    allow_overlap   = st.sidebar.checkbox("Allow room overlap", value=False)

    st.sidebar.header("Corridors & Doors")
    corridor_width_range = st.sidebar.slider("Corridor width (tiles)", 1, 6, (1, 2))
    locked_pct  = st.sidebar.slider("Locked doors %",  0, 100, 15)
    secret_pct  = st.sidebar.slider("Secret doors %",  0, 100, 10)
    trapped_pct = st.sidebar.slider("Trapped doors %", 0, 100, 5)
    special_total = locked_pct + secret_pct + trapped_pct
    door_pct = max(0.0, 100.0 - special_total)
    if special_total > 100:
        st.sidebar.warning("Door percentages exceed 100%. They will be renormalized.")
    st.sidebar.caption(f"Normal doors auto-adjust to {door_pct:.1f}% before normalization.")

    st.sidebar.header("Decor")
    trap_ratio_pct    = st.sidebar.slider("Rooms with traps %",     0, 100, 12)
    treasure_ratio_pct= st.sidebar.slider("Rooms with treasure %",  0, 100, 10)
    columns_ratio_pct = st.sidebar.slider("Large rooms with columns %", 0, 100, 18)

    st.sidebar.header("Output")
    parchment_enabled = st.sidebar.checkbox("Use parchment texture", value=True)
    save_to_disk      = st.sidebar.checkbox("Save PNG to disk", value=False)
    output_directory  = st.sidebar.text_input("Output directory", value=".")

    st.sidebar.header("Randomness")
    seed_input   = st.sidebar.text_input("Seed", value="", placeholder="Leave blank for random")
    roll_seed    = st.sidebar.button("Roll new seed")
    if "auto_seed" not in st.session_state:
        st.session_state["auto_seed"] = random.randrange(1_000_000_000)
    if roll_seed:
        st.session_state["auto_seed"] = random.randrange(1_000_000_000)
    seed_text = seed_input.strip()
    if seed_text:
        try:
            seed_value: Optional[Union[int, str]] = int(seed_text)
        except ValueError:
            seed_value = seed_text
    else:
        seed_value = st.session_state["auto_seed"]

    auto_regenerate   = st.sidebar.checkbox("Auto regenerate on change", value=True)
    generate_clicked  = st.sidebar.button("Generate dungeon", type="primary")
    trigger_generation= auto_regenerate or generate_clicked

    door_distribution = normalize_door_percentages({
        "door": door_pct,
        "locked": locked_pct,
        "secret": secret_pct,
        "trapped": trapped_pct,
    })

    parchment_path = "images/parchment.jpg" if parchment_enabled else None

    settings = DungeonSettings(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        tilesize=int(tilesize),
        vignette_strength=float(vignette_str),
        n_rooms=int(n_rooms),
        min_room_width=int(min_room_width),
        max_room_width=int(max_room_width),
        min_room_height=int(min_room_height),
        max_room_height=int(max_room_height),
        margin_tiles=int(margin_tiles),
        min_room_spacing=int(min_room_spacing),
        allow_overlap=bool(allow_overlap),
        corridor_width_tiles=(int(corridor_width_range[0]), int(corridor_width_range[1])),
        door_distribution=door_distribution,
        trap_room_ratio=trap_ratio_pct / 100.0,
        treasure_room_ratio=treasure_ratio_pct / 100.0,
        column_room_ratio=columns_ratio_pct / 100.0,
        parchment_path=parchment_path,
        image_folder="images",
        export_with_timestamp=True,
        write_to_disk=bool(save_to_disk),
        output_directory=output_directory,
        seed=seed_value,
    )

    result: Optional[DungeonResult] = None
    if trigger_generation:
        symbols = get_symbol_library(settings.image_folder)
        result = generate_dungeon(settings, symbols)
        st.session_state["last_result"] = result
    elif "last_result" in st.session_state:
        result = st.session_state["last_result"]

    if result:
        st.sidebar.metric("Rooms generated", len(result.room_contents))
        st.sidebar.caption(f"Seed used: {result.seed_used}")

        st.caption(f"Seed: {result.seed_used}")
        st.image(result.image_bytes, caption=result.filename, use_container_width=True)
        st.download_button("Download PNG", data=result.image_bytes, file_name=result.filename,
                           mime="image/png", use_container_width=True)

        if result.settings.write_to_disk:
            saved_path = os.path.abspath(os.path.join(result.settings.output_directory, result.filename))
            st.success(f"Saved a copy to {saved_path}")

        if result.legend_lines:
            st.subheader("Legend")
            st.markdown("\n".join(f"- {line}" for line in result.legend_lines))

        st.subheader("Door distribution")
        st.write({label: f"{weight * 100:.1f}%" for label, weight in result.settings.door_distribution})
    else:
        st.info("Adjust settings to your liking, then click Generate dungeon.")

if __name__ == "__main__":
    main()
