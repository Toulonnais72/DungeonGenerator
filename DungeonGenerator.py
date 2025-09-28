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

INK_BROWN = (92, 68, 42)
GRID_COLOR = (212, 196, 164)
ROOM_FILL = (238, 226, 204)
ROOM_GRID_COLOR = (204, 182, 148)
HATCH_COLOR = (168, 148, 118)
BACKGROUND_COLOR = (236, 219, 191)
VIGNETTE_COLOR = (150, 128, 100)

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
            ("door", 0.7),
            ("locked", 0.15),
            ("secret", 0.1),
            ("trapped", 0.05),
        )
    )
    trap_room_ratio: float = 0.12
    treasure_room_ratio: float = 0.1
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
    total = sum(max(0.0, raw.get(key, 0.0)) for key in order)
    if total <= 0:
        return (
            ("door", 1.0),
            ("locked", 0.0),
            ("secret", 0.0),
            ("trapped", 0.0),
        )
    return tuple((key, max(0.0, raw.get(key, 0.0)) / total) for key in order)


def weighted_choice(rng: random.Random, weighted_items: Sequence[Tuple[str, float]]) -> str:
    if not weighted_items:
        raise ValueError("weighted_items must not be empty")
    total = sum(weight for _, weight in weighted_items)
    if total <= 0:
        return weighted_items[0][0]
    r = rng.random() * total
    cumulative = 0.0
    for item, weight in weighted_items:
        cumulative += weight
        if r <= cumulative:
            return item
    return weighted_items[-1][0]


def draw_background(surface: pygame.Surface, settings: DungeonSettings, rng: random.Random) -> None:
    width, height = surface.get_size()

    # 1. Crée une base neutre
    surface.fill(BACKGROUND_COLOR)

    # 2. Charge et applique le parchemin si dispo
    if settings.parchment_path and os.path.exists(settings.parchment_path):
        texture = pygame.image.load(settings.parchment_path).convert()
        texture = pygame.transform.smoothscale(texture, (width, height))

        # 🔑 BLEND_RGBA_MULT au lieu de blit direct → on garde le fond parcheminé
        surface.blit(texture, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # 3. Ajoute un grain léger
    grain_tile = pygame.Surface((128, 128), pygame.SRCALPHA)
    for _ in range(300):  # un peu moins pour laisser respirer le parchemin
        gx = rng.randrange(128)
        gy = rng.randrange(128)
        tint = rng.randint(-12, 12)
        alpha = rng.randint(12, 28)  # plus transparent
        color = (
            max(120, min(255, BACKGROUND_COLOR[0] + tint)),
            max(108, min(255, BACKGROUND_COLOR[1] + tint)),
            max(92, min(255, BACKGROUND_COLOR[2] + tint)),
            alpha,
        )
        grain_tile.set_at((gx, gy), color)
    for y in range(0, height, 128):
        for x in range(0, width, 128):
            surface.blit(grain_tile, (x, y), special_flags=pygame.BLEND_RGBA_MULT)

    # 4. Ajoute quelques taches (mais plus transparentes)
    blot_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    max_radius = int(max(width, height) * 0.22)
    min_radius = max(settings.tilesize * 5, int(max_radius * 0.35))
    for _ in range(15):  # un peu moins que 22
        radius = rng.randint(min_radius, max_radius)
        cx = rng.randint(-radius, width + radius)
        cy = rng.randint(-radius, height + radius)
        alpha = rng.randint(12, 28)  # plus discret
        tint = rng.randint(-10, 10)
        color = (
            max(120, min(255, BACKGROUND_COLOR[0] + tint)),
            max(108, min(255, BACKGROUND_COLOR[1] + tint)),
            max(90, min(255, BACKGROUND_COLOR[2] + tint)),
            alpha,
        )
        pygame.draw.circle(blot_surface, color, (cx, cy), radius)
    surface.blit(blot_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)


def hatch_background(surface: pygame.Surface, settings: DungeonSettings) -> None:
    width, height = surface.get_size()
    step = max(8, settings.tilesize)
    tile_size = step * 2
    hatch_tile = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
    color = (*HATCH_COLOR, 30)
    stride = max(2, step // 2)
    for offset in range(-tile_size, tile_size * 2, stride):
        pygame.draw.line(hatch_tile, color, (offset, 0), (0, offset), 1)
        pygame.draw.line(hatch_tile, color, (tile_size, offset), (offset, tile_size), 1)
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            surface.blit(hatch_tile, (x, y))


def apply_floor_texture(surface: pygame.Surface, settings: DungeonSettings, rng: random.Random) -> None:
    width, height = surface.get_size()
    texture = pygame.Surface((width, height), pygame.SRCALPHA)
    cell = max(12, settings.tilesize * 2)
    for y in range(0, height + cell, cell):
        for x in range(0, width + cell, cell):
            tint = rng.randint(-10, 10)
            alpha = rng.randint(8, 16)
            color = (
                max(70, min(255, BACKGROUND_COLOR[0] + tint)),
                max(64, min(255, BACKGROUND_COLOR[1] + tint)),
                max(55, min(255, BACKGROUND_COLOR[2] + tint)),
                alpha,
            )
            pygame.draw.rect(texture, color, pygame.Rect(x, y, cell, cell))
    surface.blit(texture, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)


def apply_vignette(surface: pygame.Surface, strength: float = 0.3) -> None:
    width, height = surface.get_size()
    base = 256
    gradient = pygame.Surface((base, base), pygame.SRCALPHA)
    center = base / 2
    max_dist = math.hypot(center, center)

    # 🎨 couleur chaude pour les bords
    warm_color = (210, 110, 55)

    for y in range(base):
        for x in range(base):
            dist = math.hypot(x - center, y - center)
            norm = min(1.0, dist / max_dist)

            # Alpha fort aux bords, nul au centre
            alpha = int((norm ** 2.5) * 180 * strength)

            gradient.set_at((x, y), (*warm_color, alpha))

    vignette = pygame.transform.smoothscale(gradient, (width, height))

    # 🔑 utiliser BLEND_RGBA_SUB pour assombrir au lieu d'effacer
    surface.blit(vignette, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)





def draw_grid(surface: pygame.Surface, settings: DungeonSettings) -> None:
    width, height = surface.get_size()
    ts = max(4, settings.tilesize)
    grid_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    color = (*GRID_COLOR, 70)
    for xpix in range(0, width + 1, ts):
        pygame.draw.line(grid_surface, color, (xpix, 0), (xpix, height))
    for ypix in range(0, height + 1, ts):
        pygame.draw.line(grid_surface, color, (0, ypix), (width, ypix))
    surface.blit(grid_surface, (0, 0))


def draw_room_grid(surface: pygame.Surface, rect: pygame.Rect, settings: DungeonSettings) -> None:
    ts = settings.tilesize
    if rect.width <= ts or rect.height <= ts:
        return
    grid_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    color = (*ROOM_GRID_COLOR, 90)
    for xpix in range(ts, rect.width, ts):
        pygame.draw.line(grid_surface, color, (xpix, 0), (xpix, rect.height))
    for ypix in range(ts, rect.height, ts):
        pygame.draw.line(grid_surface, color, (0, ypix), (rect.width, ypix))
    surface.blit(grid_surface, rect.topleft)


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
        tint = rng.randint(-12, 12)
        color = (
            max(150, min(255, ROOM_FILL[0] + tint)),
            max(140, min(255, ROOM_FILL[1] + tint)),
            max(120, min(255, ROOM_FILL[2] + tint)),
            alpha,
        )
        pygame.draw.circle(shading, color, (cx, cy), radius)
    surface.blit(shading, rect.topleft, special_flags=pygame.BLEND_RGBA_MULT)



def draw_room(surface: pygame.Surface, rect: pygame.Rect, number: Optional[int], settings: DungeonSettings, rng: random.Random) -> None:
    wall = wall_thickness(settings)
    base_rect = rect.copy()
    pygame.draw.rect(surface, ROOM_FILL, base_rect)
    inner = base_rect.inflate(-2 * wall, -2 * wall)
    if inner.width <= 0 or inner.height <= 0:
        inner = base_rect.inflate(-wall, -wall)
    if inner.width <= 0 or inner.height <= 0:
        inner = base_rect.copy()
    apply_floor_shading(surface, inner, rng, intensity=0.6)
    draw_room_grid(surface, inner, settings)
    pygame.draw.rect(surface, INK_BROWN, base_rect, wall)
    if number is not None and inner.width > 0 and inner.height > 0:
        font = FONT_STORE.get(16, bold=False)
        img = font.render(str(number), True, INK_BROWN)
        surface.blit(img, (inner.centerx - img.get_width() // 2, inner.centery - img.get_height() // 2))


def draw_corridor_rect(surface: pygame.Surface, rect: pygame.Rect, settings: DungeonSettings, rng: random.Random) -> pygame.Rect:
    wall = max(2, wall_thickness(settings) - 2)
    pygame.draw.rect(surface, ROOM_FILL, rect)
    inner = rect.inflate(-2 * wall, -2 * wall)
    if inner.width <= 0 or inner.height <= 0:
        inner = rect.inflate(-wall, -wall)
    if inner.width <= 0 or inner.height <= 0:
        inner = rect.copy()
    apply_floor_shading(surface, inner, rng, intensity=0.45)
    draw_room_grid(surface, inner, settings)
    pygame.draw.rect(surface, INK_BROWN, rect, wall)
    return inner


def rects_intersect(a: pygame.Rect, b: pygame.Rect, spacing_px: int = 0) -> bool:
    if spacing_px <= 0:
        return a.colliderect(b)
    expanded = pygame.Rect(b.x - spacing_px, b.y - spacing_px, b.width + 2 * spacing_px, b.height + 2 * spacing_px)
    return a.colliderect(expanded)


def generate_rooms(settings: DungeonSettings, rng: random.Random) -> List[pygame.Rect]:
    rooms: List[pygame.Rect] = []
    width_tiles = settings.canvas_width // settings.tilesize
    height_tiles = settings.canvas_height // settings.tilesize
    max_w_tiles = width_tiles - 2 * settings.margin_tiles
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
        rect = pygame.Rect(
            x_t * settings.tilesize,
            y_t * settings.tilesize,
            w_t * settings.tilesize,
            h_t * settings.tilesize,
        )
        if not settings.allow_overlap and any(rects_intersect(rect, existing, spacing_px) for existing in rooms):
            continue
        rooms.append(rect)
    if rooms:
        minx = min(r.left for r in rooms)
        miny = min(r.top for r in rooms)
        maxx = max(r.right for r in rooms)
        maxy = max(r.bottom for r in rooms)
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
    dist = [float("inf")] * n
    parent = [-1] * n
    dist[0] = 0
    edges: List[Tuple[int, int]] = []
    for _ in range(n):
        u = -1
        best = float("inf")
        for i in range(n):
            if not in_mst[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1:
            break
        in_mst[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u))
        ux, uy = points[u]
        for v in range(n):
            if in_mst[v]:
                continue
            vx, vy = points[v]
            w = (ux - vx) ** 2 + (uy - vy) ** 2
            if w < dist[v]:
                dist[v] = w
                parent[v] = u
    return edges


def corridor_between(
    surface: pygame.Surface,
    settings: DungeonSettings,
    rng: random.Random,
    rect_a: pygame.Rect,
    rect_b: pygame.Rect,
) -> List[pygame.Rect]:
    ax, ay = rect_a.center
    bx, by = rect_b.center
    lo, hi = settings.corridor_width_range()
    corridor_w_tiles = rng.randint(lo, hi)
    corridor_width_px = max(1, corridor_w_tiles) * settings.tilesize

    def snap(value: int) -> int:
        return (value // settings.tilesize) * settings.tilesize + settings.tilesize // 2

    midx = snap(bx)
    ay_snap = snap(ay)
    by_snap = snap(by)
    corridor_rects: List[pygame.Rect] = []

    x1 = min(ax, midx)
    x2 = max(ax, midx)
    hrect = pygame.Rect(
        x1 - corridor_width_px // 2,
        ay_snap - corridor_width_px // 2,
        (x2 - x1) + corridor_width_px,
        corridor_width_px,
    )
    corridor_rects.append(draw_corridor_rect(surface, hrect, settings, rng))

    y1 = min(ay_snap, by_snap)
    y2 = max(ay_snap, by_snap)
    vrect = pygame.Rect(
        midx - corridor_width_px // 2,
        y1 - corridor_width_px // 2,
        corridor_width_px,
        (y2 - y1) + corridor_width_px,
    )
    corridor_rects.append(draw_corridor_rect(surface, vrect, settings, rng))
    return corridor_rects


def place_doors_for_connection(
    surface: pygame.Surface,
    room_rect: pygame.Rect,
    corridor_inners: Sequence[pygame.Rect],
    room_contents: Dict[int, List[str]],
    room_idx: int,
    symbols: SymbolLibrary,
    door_distribution: Sequence[Tuple[str, float]],
    rng: random.Random,
    settings: DungeonSettings,
) -> None:
    best: Optional[Tuple[pygame.Rect, pygame.Rect]] = None
    best_area = -1
    for corridor in corridor_inners:
        if not room_rect.colliderect(corridor):
            continue
        clip = room_rect.clip(corridor)
        area = clip.width * clip.height
        if area <= 0:
            continue
        if area > best_area:
            best_area = area
            best = (clip, corridor)
    if not best:
        return
    clip, corridor = best
    door_symbol = weighted_choice(rng, door_distribution)
    if corridor.width >= corridor.height:
        cy = int(max(room_rect.top, min(room_rect.bottom, corridor.centery)))
        cx = room_rect.left if corridor.centerx < room_rect.centerx else room_rect.right
    else:
        cx = int(max(room_rect.left, min(room_rect.right, corridor.centerx)))
        cy = room_rect.top if corridor.centery < room_rect.centery else room_rect.bottom
    symbols.draw(surface, door_symbol, cx, cy, scale=0.7)
    room_contents.setdefault(room_idx, []).append(DOOR_LABELS.get(door_symbol, door_symbol))


def decorate_rooms(
    surface: pygame.Surface,
    settings: DungeonSettings,
    rng: random.Random,
    rooms: Sequence[pygame.Rect],
    room_contents: Dict[int, List[str]],
    symbols: SymbolLibrary,
) -> None:
    if not rooms:
        return
    tilesize = settings.tilesize
    indexed_rooms = list(enumerate(rooms, start=1))

    large_rooms = [
        (idx, rect)
        for idx, rect in indexed_rooms
        if rect.width >= 12 * tilesize and rect.height >= 12 * tilesize
    ]
    if large_rooms and settings.column_room_ratio > 0:
        target = max(0, round(settings.column_room_ratio * len(large_rooms)))
        if target == 0 and settings.column_room_ratio > 0:
            target = 1
        target = min(target, len(large_rooms))
        if target:
            for idx, rect in rng.sample(large_rooms, target):
                inner = rect.inflate(-tilesize * 2, -tilesize * 2)
                if inner.width <= tilesize or inner.height <= tilesize:
                    continue
                placements = rng.randint(2, 4)
                for _ in range(placements):
                    min_tx = inner.left // tilesize
                    max_tx = max(min_tx, (inner.right - 1) // tilesize)
                    min_ty = inner.top // tilesize
                    max_ty = max(min_ty, (inner.bottom - 1) // tilesize)
                    tx = rng.randint(min_tx, max_tx)
                    ty = rng.randint(min_ty, max_ty)
                    cx = tx * tilesize + tilesize // 2
                    cy = ty * tilesize + tilesize // 2
                    symbols.draw(surface, "archway", cx, cy, scale=0.5)
                room_contents[idx].append("columns")

    def apply_marker(label: str, symbol_name: str, ratio: float) -> None:
        if ratio <= 0:
            return
        target = max(0, round(ratio * len(indexed_rooms)))
        if target == 0 and ratio > 0:
            target = 1
        target = min(target, len(indexed_rooms))
        if target == 0:
            return
        for idx, rect in rng.sample(indexed_rooms, target):
            symbols.draw(surface, symbol_name, rect.centerx, rect.centery, scale=0.6)
            room_contents[idx].append(label)

    apply_marker("treasure", "portcullis", settings.treasure_room_ratio)
    apply_marker("trap", "trapped", settings.trap_room_ratio)


def draw_legend(surface: pygame.Surface, settings: DungeonSettings, room_contents: Dict[int, List[str]]) -> List[str]:
    lines: List[str] = []
    for idx in sorted(room_contents.keys()):
        items = room_contents[idx]
        if items:
            lines.append(f"Room {idx}: {', '.join(items)}")
    return lines


def generate_grid_from_rooms(
    rooms: Sequence[pygame.Rect],
    corridors: Sequence[pygame.Rect],
    settings: DungeonSettings,
) -> List[List[int]]:
    tilesize = settings.tilesize
    width_tiles = math.ceil(settings.canvas_width / tilesize)
    height_tiles = math.ceil(settings.canvas_height / tilesize)
    grid = [[1 for _ in range(width_tiles)] for _ in range(height_tiles)]

    def fill_rect(rect: pygame.Rect) -> None:
        tx_start = max(0, rect.left // tilesize)
        tx_end = min(width_tiles, math.ceil(rect.right / tilesize))
        ty_start = max(0, rect.top // tilesize)
        ty_end = min(height_tiles, math.ceil(rect.bottom / tilesize))
        for ty in range(ty_start, ty_end):
            for tx in range(tx_start, tx_end):
                grid[ty][tx] = 0

    for rect in rooms:
        fill_rect(rect)
    for rect in corridors:
        fill_rect(rect)
    return grid


def marching_squares(grid: Sequence[Sequence[int]], settings: DungeonSettings) -> List[List[Tuple[int, int]]]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    tilesize = settings.tilesize
    polys: List[List[Tuple[int, int]]] = []
    for y in range(height - 1):
        for x in range(width - 1):
            state = 0
            if grid[y][x] == 0:
                state |= 1
            if grid[y][x + 1] == 0:
                state |= 2
            if grid[y + 1][x + 1] == 0:
                state |= 4
            if grid[y + 1][x] == 0:
                state |= 8
            px = x * tilesize
            py = y * tilesize
            if state in (1, 14):
                polys.append([(px, py), (px + tilesize, py)])
            if state in (2, 13):
                polys.append([(px + tilesize, py), (px + tilesize, py + tilesize)])
            if state in (4, 11):
                polys.append([(px, py + tilesize), (px + tilesize, py + tilesize)])
            if state in (8, 7):
                polys.append([(px, py), (px, py + tilesize)])
    return polys


def draw_walls(
    surface: pygame.Surface,
    polys: Sequence[Sequence[Tuple[int, int]]],
    settings: DungeonSettings,
    rng: random.Random,
    color: Tuple[int, int, int] = INK_BROWN,
) -> None:
    if not polys:
        return

    wall_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    outer = wall_thickness(settings)
    inner = max(1, outer // 2)

    # 1) Fusionne les segments voisins
    snap = max(1, settings.tilesize // 4)
    merged = hp._merge_wall_segments(polys, snap=snap)

    # 2) Filtre les petits bouts (souvent les traits parasites)
    min_len = settings.tilesize * 1.1  # ≈ 1 tuile
    cleaned = [p for p in merged if hp._poly_length(p) >= min_len]

    # 3) Dessin propre
    for poly in cleaned:
        closed = (len(poly) > 2 and poly[0] == poly[-1])
        pygame.draw.lines(wall_surface, (*color, 235), closed, poly, outer)
        pygame.draw.lines(wall_surface, (*color, 255), closed, poly, inner)

    surface.blit(wall_surface, (0, 0))


def surface_to_png_bytes(surface: pygame.Surface) -> bytes:
    width, height = surface.get_size()
    data = pygame.image.tostring(surface, "RGBA")
    image = Image.frombytes("RGBA", (width, height), data)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_dungeon(settings: DungeonSettings, symbols: SymbolLibrary) -> DungeonResult:
    ensure_pygame_ready()
    rng = random.Random(settings.seed)
    surface = pygame.Surface((settings.canvas_width, settings.canvas_height)).convert_alpha()
    draw_background(surface, settings, rng)
    hatch_background(surface, settings)
    apply_floor_texture(surface, settings, rng)
    draw_grid(surface, settings)

    rooms = generate_rooms(settings, rng)
    centers = [room.center for room in rooms]
    edges = prim_mst(centers)
    room_contents: Dict[int, List[str]] = {i: [] for i in range(1, len(rooms) + 1)}
    corridor_floors: List[pygame.Rect] = []

    for i, j in edges:
        corridor_inners = corridor_between(surface, settings, rng, rooms[i], rooms[j])
        corridor_floors.extend(corridor_inners)
        place_doors_for_connection(
            surface,
            rooms[i],
            corridor_inners,
            room_contents,
            i + 1,
            symbols,
            settings.door_distribution,
            rng,
            settings,
        )
        place_doors_for_connection(
            surface,
            rooms[j],
            corridor_inners,
            room_contents,
            j + 1,
            symbols,
            settings.door_distribution,
            rng,
            settings,
        )

    for idx, room in enumerate(rooms, start=1):
        draw_room(surface, room, idx, settings, rng)

    if rooms:
        if len(rooms) >= 2:
            up_idx, down_idx = rng.sample(range(len(rooms)), 2)
            up_room, down_room = rooms[up_idx], rooms[down_idx]
            symbols.draw(surface, "stairs_up", up_room.centerx, up_room.centery, scale=0.6)
            symbols.draw(surface, "stairs_down", down_room.centerx, down_room.centery, scale=0.6)
            room_contents[up_idx + 1].append("stairs up")
            room_contents[down_idx + 1].append("stairs down")
        else:
            symbols.draw(surface, "stairs_up", rooms[0].centerx, rooms[0].centery, scale=0.6)
            room_contents[1].append("stairs up")

    decorate_rooms(surface, settings, rng, rooms, room_contents, symbols)
    apply_vignette(surface, strength=settings.vignette_strength)
    legend_lines = draw_legend(surface, settings, room_contents)
    filename = "dungeon.png"
    if settings.export_with_timestamp:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dungeon_{stamp}.png"

    if settings.write_to_disk:
        output_path = os.path.join(settings.output_directory, filename)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pygame.image.save(surface, output_path)

    image_bytes = surface_to_png_bytes(surface)
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


def main() -> None:
    st.set_page_config(page_title="Dungeon Generator", layout="wide")
    st.title("Dungeon Generator")

    st.sidebar.header("Layout")
    canvas_width = st.sidebar.number_input("Canvas width (px)", min_value=600, max_value=3000, value=1400, step=50)
    canvas_height = st.sidebar.number_input("Canvas height (px)", min_value=600, max_value=3000, value=900, step=50)
    tilesize = st.sidebar.slider("Tile size (px)", min_value=8, max_value=40, value=15)
    vignette_str = st.sidebar.slider("Vignette", 0.0, 1.0, 0.35, 0.05)


    st.sidebar.header("Rooms")
    n_rooms = st.sidebar.slider("Number of rooms", min_value=5, max_value=80, value=28)
    min_room_width = st.sidebar.slider("Minimum room width (tiles)", min_value=3, max_value=30, value=6)
    max_room_width = st.sidebar.slider("Maximum room width (tiles)", min_value=min_room_width, max_value=40, value=18)
    min_room_height = st.sidebar.slider("Minimum room height (tiles)", min_value=3, max_value=30, value=6)
    max_room_height = st.sidebar.slider("Maximum room height (tiles)", min_value=min_room_height, max_value=40, value=16)
    margin_tiles = st.sidebar.slider("Map margin (tiles)", min_value=0, max_value=12, value=2)
    min_room_spacing = st.sidebar.slider("Room spacing (tiles)", min_value=0, max_value=6, value=1)
    allow_overlap = st.sidebar.checkbox("Allow room overlap", value=False)

    st.sidebar.header("Corridors & Doors")
    corridor_width_range = st.sidebar.slider("Corridor width (tiles)", min_value=1, max_value=6, value=(1, 2))
    locked_pct = st.sidebar.slider("Locked doors %", min_value=0, max_value=100, value=15)
    secret_pct = st.sidebar.slider("Secret doors %", min_value=0, max_value=100, value=10)
    trapped_pct = st.sidebar.slider("Trapped doors %", min_value=0, max_value=100, value=5)
    special_total = locked_pct + secret_pct + trapped_pct
    door_pct = max(0.0, 100.0 - special_total)
    if special_total > 100:
        st.sidebar.warning("Door percentages exceed 100%. They will be renormalized.")
    st.sidebar.caption(f"Normal doors auto-adjust to {door_pct:.1f}% before normalization.")

    st.sidebar.header("Decor")
    trap_ratio_pct = st.sidebar.slider("Rooms with traps %", min_value=0, max_value=100, value=12)
    treasure_ratio_pct = st.sidebar.slider("Rooms with treasure %", min_value=0, max_value=100, value=10)
    columns_ratio_pct = st.sidebar.slider("Large rooms with columns %", min_value=0, max_value=100, value=18)

    st.sidebar.header("Output")
    parchment_enabled = st.sidebar.checkbox("Use parchment texture", value=True)
    save_to_disk = st.sidebar.checkbox("Save PNG to disk", value=False)
    output_directory = st.sidebar.text_input("Output directory", value=".", help="Relative path for saved PNG (if enabled).")

    st.sidebar.header("Randomness")
    seed_input = st.sidebar.text_input("Seed", value="", placeholder="Leave blank for random")
    roll_seed = st.sidebar.button("Roll new seed")
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

    auto_regenerate = st.sidebar.checkbox("Auto regenerate on change", value=True)
    generate_clicked = st.sidebar.button("Generate dungeon", type="primary")
    trigger_generation = auto_regenerate or generate_clicked

    door_distribution = normalize_door_percentages(
        {
            "door": door_pct,
            "locked": locked_pct,
            "secret": secret_pct,
            "trapped": trapped_pct,
        }
    )

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
        st.download_button(
            "Download PNG",
            data=result.image_bytes,
            file_name=result.filename,
            mime="image/png",
            use_container_width=True,
        )

        if result.settings.write_to_disk:
            saved_path = os.path.abspath(os.path.join(result.settings.output_directory, result.filename))
            st.success(f"Saved a copy to {saved_path}")

        if result.legend_lines:
            st.subheader("Legend")
            st.markdown("\n".join(f"- {line}" for line in result.legend_lines))

        with st.expander("Room contents data"):
            st.json(result.room_contents)

        st.subheader("Door distribution")
        st.write({label: f"{weight * 100:.1f}%" for label, weight in result.settings.door_distribution})
    else:
        st.info("Adjust settings to your liking, then click Generate dungeon.")


if __name__ == "__main__":
    main()



