"""RoadSentinel – Procedural Road-Defect Texture Generator
===========================================================

Generates physically-plausible PBR texture maps (diffuse, normal, ORM) for
road surfaces with realistic damage:

  * Dry potholes – irregular shape, depth-cued shading, edge breakup, cracks
  * Water-filled potholes – turbid water, specular highlights, rim detail
  * Hairline, longitudinal and transverse cracks
  * Interconnected / alligator cracking around deteriorated zones
  * Asphalt wear, faded markings, patch repairs, minor rutting
  * Configurable severity levels and physically-meaningful parameters
  * Wheel-path-biased placement with optional clustering
  * Machine-readable ground-truth JSON for every placed defect

IMPORTANT – texture / normal-map approach
-----------------------------------------
This generator does NOT modify CARLA mesh geometry.  All "depth" effects are
achieved via:
  - Physically-motivated shading of the diffuse map (shadow / AO gradients)
  - Normal maps that encode the 3-D surface slope of the depression
  - ORM maps (Occlusion / Roughness / Metallic) for PBR material response

The depth_m field is the *intended* real-world depth and drives how dark and
how steeply the normal map slopes; it is NOT a geometric vertex displacement.
True mesh displacement would require CARLA's Python API to swap / deform
road mesh assets – that is a separate integration step.

Architecture
------------
Pure Pillow + NumPy – no CARLA import – so the module can be tested and
iterated entirely outside CARLA before integration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Noise & Utility
# ---------------------------------------------------------------------------

def fractal_noise(
    h: int,
    w: int,
    octaves: int = 5,
    persistence: float = 0.55,
    base_freq: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Fractal / fBm noise in [0, 1] via successive bicubic-upsampled grids."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), dtype=np.float64)
    amp, total = 1.0, 0.0
    freq = max(2, int(base_freq))
    for _ in range(max(1, octaves)):
        gh = min(max(2, int(freq)), h)
        gw = min(max(2, int(freq * w / max(h, 1))), w)
        grid = rng.random((gh, gw))
        im = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        im = im.resize((w, h), Image.Resampling.BICUBIC)
        out += np.asarray(im, dtype=np.float64) / 255.0 * amp
        total += amp
        amp *= persistence
        freq = int(freq * 2)
    return np.clip(out / max(total, 1e-9), 0.0, 1.0)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ---------------------------------------------------------------------------
# Shape Primitives
# ---------------------------------------------------------------------------

def jagged_polygon(
    size: int,
    base_frac: float,
    irregularity: float,
    seed: int,
    n_points: int = 80,
) -> List[Tuple[float, float]]:
    """Generate an irregular closed polygon mimicking a pothole outline."""
    rng = np.random.default_rng(seed)
    c = size / 2.0
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    radius = np.ones(n_points)
    for k in range(1, 6):
        radius += (
            irregularity
            * rng.uniform(0.25, 0.85)
            / (k ** 1.30)
            * np.sin(k * angles + rng.uniform(0, 2 * np.pi))
        )

    for _ in range(int(rng.integers(5, 9))):
        a = rng.uniform(0, 2 * np.pi)
        bw = rng.uniform(0.30, 1.00)
        d = np.angle(np.exp(1j * (angles - a)))
        f = np.clip(1 - np.abs(d) / bw, 0, 1)
        radius -= rng.uniform(0.08, 0.30) * smoothstep(f)

    for _ in range(int(rng.integers(2, 5))):
        a = rng.uniform(0, 2 * np.pi)
        bw = rng.uniform(0.15, 0.50)
        d = np.angle(np.exp(1j * (angles - a)))
        f = np.clip(1 - np.abs(d) / bw, 0, 1)
        radius += rng.uniform(0.05, 0.18) * smoothstep(f)

    radius = np.clip(radius + rng.normal(0.0, 0.020, n_points), 0.38, 1.50)
    r = radius * base_frac * size
    return [(c + rr * np.cos(a), c + rr * np.sin(a)) for rr, a in zip(r, angles)]


def rasterize_mask(size: int, pts: List[Tuple[float, float]]) -> np.ndarray:
    """Render polygon to a soft-edged alpha mask in [0, 1]."""
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).polygon(pts, fill=255)
    img = img.filter(ImageFilter.GaussianBlur(max(0.7, size / 400.0)))
    return np.asarray(img, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# Asphalt Base Texture
# ---------------------------------------------------------------------------

def asphalt_base(h: int, w: int, seed: int) -> np.ndarray:
    """Return H x W x 3 float64 representing intact asphalt (~dark grey)."""
    coarse = fractal_noise(h, w, 5, 0.55, max(2, h // 10), seed)
    fine   = fractal_noise(h, w, 6, 0.55, max(2, h // 3),  seed + 1)
    patch  = fractal_noise(h, w, 3, 0.55, 4,                seed + 2)
    stone  = fractal_noise(h, w, 4, 0.50, max(2, h // 5),  seed + 3)
    base_l = 63.0 + coarse * 20.0 - (fine - 0.5) * 18.0 + (patch - 0.5) * 10.0
    r = np.clip(base_l + stone * 4.5,  18, 235)
    g = np.clip(base_l * 0.985,         18, 235)
    b = np.clip(base_l * 0.950,         18, 235)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Normal Map
# ---------------------------------------------------------------------------

def height_to_normal(height: np.ndarray, strength: float = 2.8) -> np.ndarray:
    """Convert a [0,1] heightmap to an RGB normal map (OpenGL convention)."""
    h64 = height.astype(np.float64)
    gy, gx = np.gradient(h64)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h64)
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack([nx / mag, ny / mag, nz / mag], axis=-1) * 0.5 + 0.5
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Crack Generation
# ---------------------------------------------------------------------------

def draw_cracks(
    size: int,
    poly_pts: List[Tuple[float, float]],
    seed: int,
    n_cracks: int = 12,
    surrounding: bool = False,
    crack_type: str = "radial",
) -> np.ndarray:
    """
    Draw cracks originating from polygon boundary or scattered across area.

    crack_type:
      radial        - emanate outward from pothole edge
      longitudinal  - roughly parallel to road axis
      transverse    - roughly perpendicular to road axis
      alligator     - dense polygon cracking pattern (fatigue failure)
    """
    rng = np.random.default_rng(seed + (1300 if surrounding else 300))
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    c = size / 2.0

    if crack_type == "alligator":
        n_cells = int(rng.integers(18, 45))
        pts_x = rng.uniform(size * 0.05, size * 0.95, n_cells)
        pts_y = rng.uniform(size * 0.05, size * 0.95, n_cells)
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                dist = math.hypot(pts_x[i] - pts_x[j], pts_y[i] - pts_y[j])
                if dist < size * 0.22 and rng.random() < 0.55:
                    mid_jitter_x = rng.normal(0, size * 0.025)
                    mid_jitter_y = rng.normal(0, size * 0.025)
                    mx = (pts_x[i] + pts_x[j]) / 2 + mid_jitter_x
                    my = (pts_y[i] + pts_y[j]) / 2 + mid_jitter_y
                    w = max(1, int(rng.uniform(0.8, 2.0)))
                    draw.line(
                        [(pts_x[i], pts_y[i]), (mx, my), (pts_x[j], pts_y[j])],
                        fill=int(rng.uniform(160, 230)),
                        width=w,
                    )
        arr = np.asarray(img, dtype=np.float64) / 255.0
        return np.clip(arr, 0, 1)

    if crack_type in ("longitudinal", "transverse"):
        base_angle = 0.0 if crack_type == "longitudinal" else math.pi / 2
        for _ in range(n_cracks):
            x = rng.uniform(0.05, 0.95) * size
            y = rng.uniform(0.05, 0.95) * size
            length = rng.uniform(0.12, 0.55) * size
            direction = base_angle + rng.normal(0, 0.12)
            steps = max(6, int(length / 4.5))
            width = rng.uniform(0.6, 2.2)
            px, py = x, y
            for s in range(steps):
                direction += rng.normal(0, 0.06)
                step = length / steps
                nx2 = px + np.cos(direction) * step
                ny2 = py + np.sin(direction) * step
                ww = max(0.4, width * (1 - s / max(steps - 1, 1)) ** 1.2)
                draw.line(
                    [(px, py), (nx2, ny2)],
                    fill=int(rng.uniform(140, 220)),
                    width=max(1, int(round(ww))),
                )
                px, py = nx2, ny2
                if rng.random() < 0.14 and s < steps - 3:
                    ba = direction + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2)
                    bx, by = px, py
                    bs = max(2, int(steps * rng.uniform(0.15, 0.35)))
                    for bi in range(bs):
                        ba += rng.uniform(-0.18, 0.18)
                        nbx = bx + np.cos(ba) * step
                        nby = by + np.sin(ba) * step
                        bww = max(0.4, ww * 0.5 * (1 - bi / bs))
                        draw.line(
                            [(bx, by), (nbx, nby)],
                            fill=int(rng.uniform(120, 200)),
                            width=max(1, int(round(bww))),
                        )
                        bx, by = nbx, nby
        arr = np.asarray(img, dtype=np.float64) / 255.0
        return np.clip(arr, 0, 1)

    # Default: radial cracks from polygon boundary
    n_chosen = min(n_cracks, len(poly_pts))
    chosen = rng.choice(len(poly_pts), size=n_chosen, replace=False)
    for idx in chosen:
        ox, oy = poly_pts[idx]
        angle = math.atan2(oy - c, ox - c) + rng.uniform(-0.40, 0.40)
        total_len = size * rng.uniform(0.09, 0.30 if surrounding else 0.34)
        step = rng.uniform(3, 5) if surrounding else rng.uniform(4, 7)
        width = rng.uniform(0.5, 1.8) if surrounding else rng.uniform(1.2, 3.8)
        steps = max(4, int(total_len / step))
        x, y = ox, oy
        for s in range(steps):
            angle += rng.uniform(-0.24, 0.24)
            nx2 = x + np.cos(angle) * step
            ny2 = y + np.sin(angle) * step
            ww = max(0.4, width * (1 - s / max(steps - 1, 1)) ** 1.4)
            draw.line(
                [(x, y), (nx2, ny2)],
                fill=240,
                width=max(1, int(round(ww))),
            )
            x, y = nx2, ny2
            if rng.random() < (0.10 if surrounding else 0.18) and s < steps - 3:
                ba = angle + rng.choice([-1, 1]) * rng.uniform(0.50, 1.15)
                bx, by = x, y
                bs = max(2, int(steps * rng.uniform(0.18, 0.40)))
                for bi in range(bs):
                    ba += rng.uniform(-0.26, 0.26)
                    nbx = bx + np.cos(ba) * step
                    nby = by + np.sin(ba) * step
                    bww = max(0.4, ww * 0.55 * (1 - bi / bs))
                    draw.line(
                        [(bx, by), (nbx, nby)],
                        fill=220,
                        width=max(1, int(round(bww))),
                    )
                    bx, by = nbx, nby

    arr = np.asarray(img, dtype=np.float64) / 255.0
    if surrounding:
        blurred = Image.fromarray((arr * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(0.40)
        )
        arr = np.asarray(blurred, dtype=np.float64) / 255.0
    return np.clip(arr, 0, 1)


def draw_hairline_cracks(size: int, seed: int, density: float = 0.3) -> np.ndarray:
    """Very fine hairline cracks scattered across the surface."""
    rng = np.random.default_rng(seed + 7700)
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    n = max(1, int(density * 35))
    for _ in range(n):
        x = rng.uniform(0.02, 0.98) * size
        y = rng.uniform(0.02, 0.98) * size
        angle = rng.uniform(0, math.pi)
        length = rng.uniform(0.04, 0.18) * size
        steps = max(4, int(length / 3))
        for s in range(steps):
            angle += rng.normal(0, 0.05)
            nx2 = x + np.cos(angle) * (length / steps)
            ny2 = y + np.sin(angle) * (length / steps)
            draw.line([(x, y), (nx2, ny2)], fill=int(rng.uniform(100, 180)), width=1)
            x, y = nx2, ny2
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return np.clip(arr * density, 0, 1)


# ---------------------------------------------------------------------------
# Edge Debris / Aggregate Chunks
# ---------------------------------------------------------------------------

def debris_chunks(
    size: int,
    poly_pts: List[Tuple[float, float]],
    seed: int,
    n: int = 10,
) -> np.ndarray:
    """Scatter broken asphalt chunks near the pothole rim."""
    rng = np.random.default_rng(seed + 400)
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for _ in range(n):
        ox, oy = poly_pts[rng.integers(0, len(poly_pts))]
        a = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(2, 20)
        r = rng.uniform(3, 11)
        px = ox + np.cos(a) * dist
        py = oy + np.sin(a) * dist
        n_pts = int(rng.integers(5, 9))
        phase = rng.uniform(0, 2 * np.pi)
        pts = []
        for k in range(n_pts):
            ang = phase + 2 * np.pi * k / n_pts
            rr = r * rng.uniform(0.60, 1.30)
            pts.append((px + rr * np.cos(ang), py + rr * np.sin(ang)))
        draw.polygon(pts, fill=int(rng.uniform(110, 225)))
    return np.asarray(img, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PotholeSpec:
    """
    Physically-meaningful specification for a single pothole.

    Parameters
    ----------
    seed : int
        RNG seed (ensures reproducibility).
    diameter_m : float
        Pothole opening diameter in metres (min 0.08 m).
    depth_m : float
        Maximum depth from road surface to pothole floor, in metres (min 1 mm).
        Drives shading intensity and normal-map slope only.
        Does NOT displace CARLA mesh vertices (see module docstring).
    irregularity : float [0, 1]
        0 = near-circular, 1 = highly jagged outline.
    is_water : bool
        Whether the pothole is water-filled.
    turbidity : float [0, 1]
        Water turbidity: 0 = clear blue, 1 = muddy brown.
    water_level_m : float or None
        Depth of water inside the depression in metres.
        Defaults to ~55% of depth_m when is_water=True.
    edge_breakup : float [0, 1]
        How much the rim is shattered / spalled.
    surrounding_cracks : float [0, 1]
        Intensity of radial cracks propagating from the rim outward.
    crack_pattern : str
        "radial" | "longitudinal" | "transverse" | "alligator"
    severity : str
        Human-readable severity label. Auto-assigned from depth_m if empty.
    """

    seed: int
    diameter_m: float
    depth_m: float
    irregularity: float = 0.55
    is_water: bool = False
    turbidity: float = 0.40
    water_level_m: Optional[float] = None
    edge_breakup: float = 0.55
    surrounding_cracks: float = 0.35
    crack_pattern: str = "radial"
    severity: str = ""

    def __post_init__(self) -> None:
        self.diameter_m = max(0.08, float(self.diameter_m))
        self.depth_m = max(0.001, float(self.depth_m))
        self.irregularity = float(np.clip(self.irregularity, 0.0, 1.0))
        self.turbidity = float(np.clip(self.turbidity, 0.0, 1.0))
        self.edge_breakup = float(np.clip(self.edge_breakup, 0.0, 1.0))
        self.surrounding_cracks = float(np.clip(self.surrounding_cracks, 0.0, 1.0))
        if self.water_level_m is None:
            self.water_level_m = self.depth_m * 0.55 if self.is_water else 0.0
        self.water_level_m = float(np.clip(self.water_level_m, 0.0, self.depth_m))
        if not self.severity:
            d = self.depth_m
            if d < 0.030:
                self.severity = "low"
            elif d < 0.070:
                self.severity = "medium"
            elif d < 0.130:
                self.severity = "high"
            else:
                self.severity = "critical"


@dataclass
class RoadWearSpec:
    """
    Specification for overall road surface deterioration.

    Parameters
    ----------
    seed : int
    wear_level : float [0,1]      General surface wear / abrasion.
    crack_density : float [0,1]   Density of longitudinal/transverse cracks.
    patchiness : float [0,1]      Extent of patch repairs.
    rutting : float [0,1]         Wheel-track rutting in normal map.
    faded_surface : float [0,1]   Binder oxidation / greying.
    hairline_density : float [0,1] Fine hairline crack density.
    alligator_zone : float [0,1]  Fraction showing alligator cracking.
    """

    seed: int
    wear_level: float = 0.25
    crack_density: float = 0.20
    patchiness: float = 0.15
    rutting: float = 0.00
    faded_surface: float = 0.10
    hairline_density: float = 0.10
    alligator_zone: float = 0.00


# ---------------------------------------------------------------------------
# Pothole Texture Synthesis
# ---------------------------------------------------------------------------

def make_pothole(spec: PotholeSpec, tex_px: int = 512):
    """
    Synthesize a single pothole tile.

    Returns
    -------
    diffuse : H x W x 3  uint8
    normal  : H x W x 3  uint8  (OpenGL tangent-space normal map)
    orm     : H x W x 4  uint8  (Occlusion, Roughness, Metallic, padding)
    mask    : H x W      float64  soft alpha in [0, 1]
    poly    : list of (x, y) polygon boundary points
    """
    size = int(tex_px)
    rng = np.random.default_rng(spec.seed)

    poly = jagged_polygon(size, 0.32, spec.irregularity, spec.seed)
    mask = rasterize_mask(size, poly)

    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2.0
    radial = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (size / 2.0)

    n1 = fractal_noise(size, size, 4, 0.55, 3, spec.seed + 50)
    n2 = fractal_noise(size, size, 4, 0.55, 7, spec.seed + 51)
    field = np.clip(0.84 + 0.22 * (n1 - 0.5) + 0.13 * (n2 - 0.5), 0.60, 1.12)
    bowl_exp = rng.uniform(1.60, 2.50)
    bowl = np.clip(1.0 - radial, 0.0, 1.0) ** bowl_exp
    bowl = np.clip(bowl * field, 0.0, 1.0)

    low_freq = fractal_noise(size, size, 3, 0.55, 3, spec.seed + 100)
    h_field = np.clip(bowl * (0.80 + 0.28 * low_freq), 0.0, 1.0) * mask

    base = asphalt_base(size, size, spec.seed)

    inner_cracks = draw_cracks(
        size, poly, spec.seed,
        n_cracks=int(6 + spec.irregularity * 9),
        surrounding=False,
        crack_type=spec.crack_pattern,
    )
    outer_cracks = (
        draw_cracks(
            size, poly, spec.seed,
            n_cracks=int(4 + spec.surrounding_cracks * 20),
            surrounding=True,
            crack_type="radial",
        )
        * spec.surrounding_cracks
    )
    chunks = debris_chunks(size, poly, spec.seed, int(4 + spec.edge_breakup * 12))
    aggregate = fractal_noise(size, size, 6, 0.55, max(2, size // 4), spec.seed + 77)

    sun = np.array([-0.55, -0.83])
    sun /= np.linalg.norm(sun)
    relx = (xx - c) / (size / 2.0)
    rely = (yy - c) / (size / 2.0)
    shadow = np.clip(-(relx * sun[0] + rely * sun[1]), 0.0, 1.0)
    lit    = np.clip(  relx * sun[0] + rely * sun[1],  0.0, 1.0)

    ao = np.clip(h_field * 1.60 + shadow * mask * 0.72 + inner_cracks * 0.20, 0.0, 1.0)
    rim_band = np.clip(mask * 1.25 - mask, 0.0, 1.0)

    diffuse = base.copy().astype(np.float64)
    strength_eb = 0.55 + 0.70 * spec.edge_breakup
    diffuse += (rim_band * lit * strength_eb)[..., None] * np.array([32.0, 28.0, 22.0])
    diffuse -= (rim_band * shadow * (0.68 + 0.62 * spec.edge_breakup))[..., None] * np.array([22.0, 22.0, 20.0])

    if not spec.is_water:
        depth_t = np.clip(h_field * 1.40, 0.0, 1.0)[..., None]
        interior = (
            np.array([60.0, 49.0, 41.0]) * (1.0 - depth_t)
            + np.array([22.0, 17.0, 15.0]) * depth_t
        )
        interior *= 0.80 + 0.32 * (1.0 - ao[..., None])
        interior += (aggregate[..., None] - 0.5) * 24.0
        diffuse = diffuse * (1.0 - mask[..., None]) + interior * mask[..., None]
        diffuse *= 1.0 - inner_cracks[..., None] * 0.60
        diffuse *= 1.0 - outer_cracks[..., None] * 0.30
        diffuse = np.where(
            (chunks > 0)[..., None] & (mask[..., None] < 0.5),
            diffuse * 0.36 + np.array([42.0, 38.0, 33.0]) * 0.64,
            diffuse,
        )
        rough = np.clip(
            0.58 + mask * 0.32 + (aggregate - 0.5) * 0.22 + inner_cracks * 0.06,
            0.12, 0.97,
        )
        metal = np.zeros_like(mask)
        normal_height = h_field

    else:
        clean = np.array([46.0, 72.0, 86.0])
        murky = np.array([55.0, 51.0, 34.0])
        water_color = clean * (1.0 - spec.turbidity) + murky * spec.turbidity

        diffuse += rim_band[..., None] * (np.array([28.0, 29.0, 28.0]) - diffuse) * 0.90

        fill_frac = float(spec.water_level_m / max(spec.depth_m, 1e-9))
        edge = np.clip(0.15 + fill_frac * 0.92, 0.15, 1.0)
        water_mask = mask * np.clip((edge - radial) / max(edge, 1e-9), 0.0, 1.0) ** 1.65

        variation = fractal_noise(size, size, 3, 0.55, 4, spec.seed + 900)
        ripple    = fractal_noise(size, size, 5, 0.50, 8, spec.seed + 901)
        water_rgb = water_color * (0.88 + 0.18 * (variation[..., None] - 0.5))

        diffuse = (
            diffuse * (1.0 - water_mask[..., None])
            + water_rgb * water_mask[..., None]
        )

        rdx = relx - sun[0] * 0.50
        rdy = rely - sun[1] * 0.50
        spec_term = (
            np.clip(1.0 - np.sqrt(rdx * rdx + rdy * rdy) * 2.5, 0.0, 1.0) ** 2.2
            * water_mask
            * (0.72 + 0.38 * variation)
        )
        ripple_spec = (
            np.clip(ripple - 0.60, 0.0, 0.4) / 0.4
            * water_mask
            * 0.30
        )
        diffuse += (spec_term + ripple_spec)[..., None] * np.array([130.0, 138.0, 144.0])
        diffuse *= 1.0 - outer_cracks[..., None] * 0.28

        rough = np.clip(
            0.90 - water_mask * 0.76 + aggregate * 0.05,
            0.05, 0.97,
        )
        metal = water_mask * 0.03
        normal_height = h_field.copy()
        normal_height *= 1.0 - water_mask * 0.80

    diffuse = np.clip(diffuse, 0.0, 255.0).astype(np.uint8)
    normal  = height_to_normal(normal_height, strength=2.80)
    orm = np.stack(
        [
            np.clip((1.0 - ao) * 255.0, 0, 255),
            np.clip(rough * 255.0, 0, 255),
            np.clip(metal * 255.0, 0, 255),
            np.zeros_like(ao),
        ],
        axis=-1,
    ).astype(np.uint8)

    return diffuse, normal, orm, mask, poly


# ---------------------------------------------------------------------------
# Road-Wide Deterioration Layer
# ---------------------------------------------------------------------------

def make_road_condition_layer(
    h: int,
    w: int,
    spec: RoadWearSpec,
):
    """
    Compute per-pixel wear, crack, deformation, fade and patch layers.

    Returns
    -------
    wear        : [0,1] surface abrasion
    cracks      : [0,1] crack map
    deformation : [0,1] rut depth field
    fade        : [0,1] oxidation whitening
    patch       : [0,1] repair patch map
    hairlines   : [0,1] hairline crack map
    """
    rng = np.random.default_rng(spec.seed + 20)

    coarse = fractal_noise(h, w, 4, 0.55, 4,               spec.seed + 10)
    medium = fractal_noise(h, w, 5, 0.55, max(3, h // 12), spec.seed + 11)
    fine   = fractal_noise(h, w, 4, 0.55, max(3, h // 3),  spec.seed + 12)

    wear = np.clip(
        spec.wear_level * (0.35 * coarse + 0.40 * medium + 0.25 * fine) / 0.65,
        0.0, 1.0,
    )

    crack_img = Image.new("L", (w, h), 0)
    d_draw = ImageDraw.Draw(crack_img)
    n_cracks = max(1, int(8 + 52 * spec.crack_density))
    for _ in range(n_cracks):
        x  = rng.uniform(0.04, 0.96) * w
        y  = rng.uniform(0.0, 1.0) * h
        length = rng.uniform(0.06, 0.48) * h
        base_dir = rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.20)
        steps = max(5, int(length / 4.5))
        wd = rng.uniform(0.5, 1.6) * (0.65 + spec.crack_density)
        px2, py2 = x, y
        for _ in range(steps):
            base_dir += rng.normal(0, 0.07)
            step = length / steps
            nx2 = px2 + np.cos(base_dir) * step
            ny2 = py2 + np.sin(base_dir) * step
            d_draw.line(
                [(px2, py2), (nx2, ny2)],
                fill=int(95 + 130 * spec.crack_density),
                width=max(1, int(wd)),
            )
            px2, py2 = nx2, ny2
    cracks = (
        np.asarray(crack_img.filter(ImageFilter.GaussianBlur(0.30)), dtype=np.float64)
        / 255.0
        * spec.crack_density
    )

    patch_raw = fractal_noise(h, w, 3, 0.55, 4, spec.seed + 30)
    patch = np.clip(patch_raw * 1.30 - 0.28, 0.0, 1.0) * spec.patchiness

    _, xx_r = np.mgrid[0:h, 0:w]
    xn = xx_r / max(w - 1, 1)
    tracks = 0.5 * (
        np.exp(-((xn - 0.25) / 0.055) ** 2)
        + np.exp(-((xn - 0.75) / 0.055) ** 2)
    )
    deformation = np.clip(tracks * spec.rutting * (0.55 + 0.45 * medium), 0.0, 1.0)

    fade = np.clip((coarse * 0.72 + fine * 0.28) * spec.faded_surface, 0.0, 1.0)

    hairlines = draw_hairline_cracks(max(h, w), spec.seed, spec.hairline_density)
    hairlines = hairlines[:h, :w]

    return wear, cracks, deformation, fade, patch, hairlines


# ---------------------------------------------------------------------------
# Scatter Planning
# ---------------------------------------------------------------------------

def scatter_plan(
    length_m: float,
    width_m: float,
    n_dry: int = 6,
    n_water: int = 2,
    wheel_path_frac: Tuple[float, float] = (0.22, 0.78),
    size_range_m: Tuple[float, float] = (0.30, 1.60),
    depth_range_m: Tuple[float, float] = (0.035, 0.15),
    cluster_prob: float = 0.30,
    crack_patterns: Optional[List[str]] = None,
    seed: int = 0,
) -> List[Tuple[float, float, "PotholeSpec"]]:
    """
    Generate a wheel-path-biased placement plan for potholes.

    Returns list of (along_m, across_m, PotholeSpec).
    """
    if crack_patterns is None:
        crack_patterns = ["radial", "longitudinal", "transverse", "alligator"]

    rng = np.random.default_rng(seed)
    kinds = [False] * int(n_dry) + [True] * int(n_water)
    rng.shuffle(kinds)
    out = []
    i = 0
    while i < len(kinds):
        water = bool(kinds[i])
        along  = float(rng.uniform(0.06, 0.94) * length_m)
        across = float(rng.choice(list(wheel_path_frac))) * width_m + float(
            rng.normal(0, width_m * 0.055)
        )
        across = float(np.clip(across, 0.05 * width_m, 0.95 * width_m))
        diameter = float(
            np.clip(rng.exponential(0.42) + size_range_m[0], *size_range_m)
        )
        depth = min(float(rng.uniform(*depth_range_m)), diameter * 0.22)
        water_lvl = float(depth * rng.uniform(0.35, 0.82)) if water else 0.0
        cp = str(rng.choice(crack_patterns))
        s = PotholeSpec(
            seed=int(rng.integers(0, 1_000_000)),
            diameter_m=diameter,
            depth_m=depth,
            irregularity=float(rng.uniform(0.38, 0.85)),
            is_water=water,
            turbidity=float(rng.uniform(0.05, 0.68)),
            water_level_m=water_lvl,
            edge_breakup=float(rng.uniform(0.30, 0.92)),
            surrounding_cracks=float(rng.uniform(0.18, 0.92)),
            crack_pattern=cp,
        )
        out.append((along, across, s))
        i += 1

        if rng.random() < cluster_prob and i < len(kinds):
            water2  = bool(kinds[i])
            along2  = along  + float(rng.uniform(0.55, 2.30))
            across2 = across + float(rng.uniform(-0.60, 0.60))
            across2 = float(np.clip(across2, 0.05 * width_m, 0.95 * width_m))
            d2   = float(np.clip(
                rng.exponential(0.28) + size_range_m[0] * 0.60,
                size_range_m[0] * 0.55, size_range_m[1],
            ))
            dep2 = min(float(rng.uniform(*depth_range_m) * 0.80), d2 * 0.22)
            wl2  = float(dep2 * rng.uniform(0.35, 0.80)) if water2 else 0.0
            cp2  = str(rng.choice(crack_patterns))
            s2 = PotholeSpec(
                seed=int(rng.integers(0, 1_000_000)),
                diameter_m=d2,
                depth_m=dep2,
                irregularity=float(rng.uniform(0.42, 0.85)),
                is_water=water2,
                turbidity=float(rng.uniform(0.05, 0.68)),
                water_level_m=wl2,
                edge_breakup=float(rng.uniform(0.30, 0.92)),
                surrounding_cracks=float(rng.uniform(0.18, 0.92)),
                crack_pattern=cp2,
            )
            out.append((along2, across2, s2))
            i += 1

    return out


# ---------------------------------------------------------------------------
# Full Road Patch Compositor
# ---------------------------------------------------------------------------

def compose_road_patch(
    length_m: float,
    width_m: float,
    px_per_m: float,
    placements: List[Tuple[float, float, "PotholeSpec"]],
    seed: int = 0,
    tex_px: int = 512,
    road_wear: Optional[RoadWearSpec] = None,
):
    """
    Composite all potholes and wear layers onto a road patch canvas.

    Parameters
    ----------
    length_m   : road segment length in metres
    width_m    : road segment width in metres
    px_per_m   : canvas resolution (pixels per metre)
    placements : list from scatter_plan()
    seed       : canvas RNG seed
    tex_px     : per-pothole tile resolution
    road_wear  : optional RoadWearSpec

    Returns
    -------
    diffuse  : H x W x 3  uint8
    normal   : H x W x 3  uint8
    orm      : H x W x 4  uint8
    placed   : list of ground-truth dicts
    meta     : full ground-truth dict
    """
    W = max(8, int(width_m  * px_per_m))
    L = max(8, int(length_m * px_per_m))

    diffuse = asphalt_base(L, W, seed).astype(np.float64)
    normal  = np.zeros((L, W, 3), dtype=np.float64)
    normal[..., 2] = 255.0
    orm = np.zeros((L, W, 4), dtype=np.float64)
    orm[..., 0] = 255.0
    orm[..., 1] = int(0.60 * 255)

    road_meta: Dict = {}

    if road_wear is not None:
        wear, cracks, deformation, fade, patch, hairlines = make_road_condition_layer(
            L, W, road_wear
        )
        diffuse *= 1.0 - wear[..., None] * 0.18
        lum = diffuse.mean(axis=-1, keepdims=True)
        diffuse = diffuse * (1.0 - fade[..., None] * 0.32) + lum * (fade[..., None] * 0.32)
        diffuse *= 1.0 - cracks[..., None] * 0.75
        diffuse *= 1.0 - hairlines[..., None] * 0.35
        diffuse += patch[..., None] * np.array([13.0, 12.0, 11.0])

        gy_rut, gx_rut = np.gradient(deformation.astype(float))
        nx_rut = -gx_rut * 4.2
        ny_rut = -gy_rut * 4.2
        nz_rut = np.ones_like(deformation)
        mag_rut = np.sqrt(nx_rut ** 2 + ny_rut ** 2 + nz_rut ** 2)
        local_n = np.stack([nx_rut / mag_rut, ny_rut / mag_rut, nz_rut / mag_rut], axis=-1)
        local_n = np.clip((local_n * 0.5 + 0.5) * 255.0, 0.0, 255.0)
        blend = deformation[..., None] * 0.50
        normal = normal * (1.0 - blend) + local_n * blend

        road_meta = {
            "wear_level":       float(road_wear.wear_level),
            "crack_density":    float(road_wear.crack_density),
            "patchiness":       float(road_wear.patchiness),
            "rutting":          float(road_wear.rutting),
            "faded_surface":    float(road_wear.faded_surface),
            "hairline_density": float(road_wear.hairline_density),
            "alligator_zone":   float(road_wear.alligator_zone),
        }

    placed: List[Dict] = []
    for along, across, spec in placements:
        d_tile, n_tile, o_tile, m_tile, _ = make_pothole(spec, tex_px)

        size_px = max(8, int(spec.diameter_m * px_per_m / 0.64))
        d_tile = np.asarray(
            Image.fromarray(d_tile).resize((size_px, size_px), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
        n_tile = np.asarray(
            Image.fromarray(n_tile).resize((size_px, size_px), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
        o_tile = np.asarray(
            Image.fromarray(o_tile).resize((size_px, size_px), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
        m_tile = np.asarray(
            Image.fromarray((m_tile * 255).astype(np.uint8)).resize(
                (size_px, size_px), Image.Resampling.LANCZOS
            ),
            dtype=np.float64,
        ) / 255.0

        cx = int(across * px_per_m)
        cy = int(along  * px_per_m)
        x0, y0 = cx - size_px // 2, cy - size_px // 2
        xs0 = max(0, x0);       xs1 = min(W, x0 + size_px)
        ys0 = max(0, y0);       ys1 = min(L, y0 + size_px)
        if xs1 <= xs0 or ys1 <= ys0:
            continue

        sx0 = xs0 - x0;  sy0 = ys0 - y0
        sx1 = sx0 + (xs1 - xs0)
        sy1 = sy0 + (ys1 - ys0)
        alpha = m_tile[sy0:sy1, sx0:sx1][..., None]

        diffuse[ys0:ys1, xs0:xs1] = (
            diffuse[ys0:ys1, xs0:xs1] * (1.0 - alpha)
            + d_tile[sy0:sy1, sx0:sx1] * alpha
        )
        normal[ys0:ys1, xs0:xs1] = (
            normal[ys0:ys1, xs0:xs1] * (1.0 - alpha)
            + n_tile[sy0:sy1, sx0:sx1] * alpha
        )
        orm[ys0:ys1, xs0:xs1] = (
            orm[ys0:ys1, xs0:xs1] * (1.0 - alpha)
            + o_tile[sy0:sy1, sx0:sx1] * alpha
        )

        area_proxy   = math.pi * (spec.diameter_m / 2.0) ** 2 * 0.72
        volume_proxy = area_proxy * spec.depth_m * 0.42
        placed.append(
            {
                "u":                  float(np.clip(across / width_m,  0, 1)),
                "v":                  float(np.clip(along  / length_m, 0, 1)),
                "x_across_m":         float(across),
                "y_along_m":          float(along),
                "diameter_m":         float(spec.diameter_m),
                "depth_m":            float(spec.depth_m),
                "is_water":           bool(spec.is_water),
                "water_level_m":      float(spec.water_level_m),
                "turbidity":          float(spec.turbidity),
                "irregularity":       float(spec.irregularity),
                "edge_breakup":       float(spec.edge_breakup),
                "surrounding_cracks": float(spec.surrounding_cracks),
                "crack_pattern":      str(spec.crack_pattern),
                "severity":           str(spec.severity),
                "area_proxy_m2":      float(area_proxy),
                "volume_proxy_m3":    float(volume_proxy),
                "_note": (
                    "depth_m drives texture shading and normal-map slope only; "
                    "no CARLA mesh/geometry displacement is performed."
                ),
            }
        )

    meta: Dict = {
        "road": {
            "length_m":     float(length_m),
            "width_m":      float(width_m),
            "px_per_m":     float(px_per_m),
            "canvas_px_W":  int(W),
            "canvas_px_H":  int(L),
            **road_meta,
        },
        "defects":             placed,
        "total_potholes":      len(placed),
        "water_filled_count":  sum(1 for p in placed if p["is_water"]),
        "dry_count":           sum(1 for p in placed if not p["is_water"]),
        "seed":                int(seed),
    }

    return (
        np.clip(diffuse, 0, 255).astype(np.uint8),
        np.clip(normal,  0, 255).astype(np.uint8),
        np.clip(orm,     0, 255).astype(np.uint8),
        placed,
        meta,
    )


# ---------------------------------------------------------------------------
# Standalone Preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Output directory: always relative to this script's own location.
    # Never writes to /tmp, /home/<someone-else>, or any hardcoded path.
    _HERE = Path(__file__).resolve().parent
    OUT   = _HERE / "road_defects_preview"
    OUT.mkdir(parents=True, exist_ok=True)

    print("RoadSentinel - Texture Generator Preview")
    print(f"Output directory: {OUT}")
    print()

    # --- 1. Individual pothole tiles ---
    demo_specs = [
        PotholeSpec(seed=1,  diameter_m=0.80, depth_m=0.09,  irregularity=0.65,
                    is_water=False, edge_breakup=0.70, surrounding_cracks=0.60,
                    crack_pattern="radial"),
        PotholeSpec(seed=2,  diameter_m=1.20, depth_m=0.14,  irregularity=0.78,
                    is_water=True,  turbidity=0.30, water_level_m=0.08,
                    edge_breakup=0.55, surrounding_cracks=0.80, crack_pattern="radial"),
        PotholeSpec(seed=3,  diameter_m=0.45, depth_m=0.05,  irregularity=0.40,
                    is_water=False, edge_breakup=0.35, surrounding_cracks=0.25,
                    crack_pattern="longitudinal"),
        PotholeSpec(seed=4,  diameter_m=1.50, depth_m=0.16,  irregularity=0.85,
                    is_water=True,  turbidity=0.72, water_level_m=0.12,
                    edge_breakup=0.88, surrounding_cracks=0.90, crack_pattern="alligator"),
        PotholeSpec(seed=5,  diameter_m=0.60, depth_m=0.07,  irregularity=0.55,
                    is_water=False, edge_breakup=0.50, surrounding_cracks=0.50,
                    crack_pattern="transverse"),
    ]
    tile_labels = [
        "dry_medium",
        "water_clear_large",
        "dry_small_longitudinal",
        "water_turbid_critical",
        "dry_transverse",
    ]

    print("Generating individual pothole tiles ...")
    for spec, label in zip(demo_specs, tile_labels):
        diff, norm, orm_tile, _, _ = make_pothole(spec, tex_px=512)
        Image.fromarray(diff).save(OUT / f"tile_{label}_diffuse.png")
        Image.fromarray(norm).save(OUT / f"tile_{label}_normal.png")
        Image.fromarray(orm_tile[..., :3]).save(OUT / f"tile_{label}_orm_rgb.png")
        print(
            f"  [{spec.severity.upper():8s}] {label}  "
            f"diam={spec.diameter_m:.2f} m  depth={spec.depth_m * 100:.1f} cm  "
            f"water={'YES' if spec.is_water else ' NO '}"
        )

    # --- 2. Full road-patch composite ---
    print()
    print("Generating full road-patch composite ...")

    wear = RoadWearSpec(
        seed=42,
        wear_level=0.38,
        crack_density=0.30,
        patchiness=0.20,
        rutting=0.25,
        faded_surface=0.18,
        hairline_density=0.20,
        alligator_zone=0.10,
    )

    plan = scatter_plan(
        length_m=30.0,
        width_m=14.0,
        n_dry=6,
        n_water=3,
        seed=7,
        cluster_prob=0.35,
        crack_patterns=["radial", "longitudinal", "transverse", "alligator"],
    )

    diffuse, normal, orm_out, placed, meta = compose_road_patch(
        length_m=30.0,
        width_m=14.0,
        px_per_m=24,
        placements=plan,
        seed=7,
        tex_px=512,
        road_wear=wear,
    )

    Image.fromarray(diffuse).save(OUT / "_preview_diffuse.png")
    Image.fromarray(normal).save(OUT / "_preview_normal.png")
    Image.fromarray(orm_out[..., :3]).save(OUT / "_preview_orm_rgb.png")

    with open(OUT / "_preview_ground_truth.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"\n{len(placed)} potholes placed on 30 m x 14 m road patch:")
    print(f"  Dry   : {meta['dry_count']}")
    print(f"  Water : {meta['water_filled_count']}")
    print()
    for idx, p in enumerate(placed, 1):
        kind = "WATER" if p["is_water"] else "DRY  "
        print(
            f"  [{idx:2d}] {kind} | {p['severity']:8s} | "
            f"diam={p['diameter_m']:.2f} m | depth={p['depth_m'] * 100:.1f} cm | "
            f"water={p['water_level_m'] * 100:.1f} cm | "
            f"cracks={p['crack_pattern']:12s} | "
            f"u={p['u']:.2f}  v={p['v']:.2f}"
        )

    print()
    print(f"Saved to: {OUT}/")
    print("  _preview_diffuse.png")
    print("  _preview_normal.png")
    print("  _preview_orm_rgb.png")
    print("  _preview_ground_truth.json")
    print("  tile_*.png  (individual pothole tiles x5 x3 maps each)")
    print()
    print("IMPORTANT: depth_m drives texture shading and normal-map slope only.")
    print("No CARLA mesh / geometry displacement is performed by this module.")
