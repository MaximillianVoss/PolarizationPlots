# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "polarization_app" / "assets"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _lerp(first: int, second: int, factor: float) -> int:
    return int(round(first + (second - first) * factor))


def _hex_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha


def _gradient_tile(size: int, radius: int) -> Image.Image:
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    margin = int(size * 0.055)
    mask_draw.rounded_rectangle((margin, margin, size - margin, size - margin), radius=radius, fill=255)

    gradient = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pixels = gradient.load()
    top = (252, 253, 255)
    bottom = (232, 236, 242)
    for y in range(size):
        factor = y / max(1, size - 1)
        for x in range(size):
            pixels[x, y] = (
                _lerp(top[0], bottom[0], factor),
                _lerp(top[1], bottom[1], factor),
                _lerp(top[2], bottom[2], factor),
                255,
            )

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_mask = mask.filter(ImageFilter.GaussianBlur(int(size * 0.025)))
    shadow.alpha_composite(Image.new("RGBA", (size, size), (22, 32, 46, 52)), (0, int(size * 0.018)))
    shadow.putalpha(shadow_mask)
    tile.alpha_composite(shadow)
    tile.alpha_composite(Image.composite(gradient, Image.new("RGBA", (size, size), (0, 0, 0, 0)), mask))

    border = ImageDraw.Draw(tile)
    border.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=radius,
        outline=(210, 216, 226, 180),
        width=max(1, size // 128),
    )
    return tile


def _project(point: tuple[float, float, float], size: int) -> tuple[float, float]:
    x, y, z = point
    origin_x = size * 0.25
    origin_y = size * 0.70
    scale = size * 0.205
    skew_x = size * 0.115
    skew_y = -size * 0.115
    return origin_x + x * scale + z * skew_x, origin_y - y * scale + z * skew_y


def _draw_lattice(draw: ImageDraw.ImageDraw, size: int) -> None:
    depth = [0.0, 1.0]
    coords = [(x, y, z) for z in depth for x in range(3) for y in range(3)]
    points = {coord: _project(coord, size) for coord in coords}
    rod_color = (93, 105, 120, 178)
    rod_width = max(2, size // 80)

    for z in depth:
        for y in range(3):
            for x in range(2):
                draw.line((points[(x, y, z)], points[(x + 1, y, z)]), fill=rod_color, width=rod_width)
        for x in range(3):
            for y in range(2):
                draw.line((points[(x, y, z)], points[(x, y + 1, z)]), fill=rod_color, width=rod_width)
    for x in range(3):
        for y in range(3):
            draw.line((points[(x, y, 0.0)], points[(x, y, 1.0)]), fill=rod_color, width=rod_width)

    node_radius = max(4, size // 34)
    for coord in sorted(coords, key=lambda item: item[2]):
        x, y = points[coord]
        _draw_sphere(draw, x, y, node_radius, "#27313d", "#8893a2")


def _draw_sphere(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int, base: str, highlight: str) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=_hex_to_rgba(base), outline=(20, 27, 36, 180))
    h = max(1, radius // 2)
    draw.ellipse((x - radius * 0.45, y - radius * 0.55, x - radius * 0.45 + h, y - radius * 0.55 + h), fill=_hex_to_rgba(highlight, 210))


def _trajectory_points(size: int) -> list[tuple[float, float]]:
    return [
        (size * 0.19, size * 0.79),
        (size * 0.32, size * 0.69),
        (size * 0.49, size * 0.55),
        (size * 0.66, size * 0.39),
        (size * 0.82, size * 0.25),
    ]


def _draw_bezier(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], width: int, fill: tuple[int, int, int, int]) -> None:
    samples = []
    for index in range(len(points) - 1):
        p0 = points[max(0, index - 1)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(len(points) - 1, index + 2)]
        for step in range(18):
            t = step / 18.0
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            samples.append((x, y))
    samples.append(points[-1])
    draw.line(samples, fill=fill, width=width, joint="curve")


def _draw_arrow_head(draw: ImageDraw.ImageDraw, tip: tuple[float, float], angle: float, size: int, fill: tuple[int, int, int, int]) -> None:
    left = (
        tip[0] - math.cos(angle - math.pi / 6.0) * size,
        tip[1] - math.sin(angle - math.pi / 6.0) * size,
    )
    right = (
        tip[0] - math.cos(angle + math.pi / 6.0) * size,
        tip[1] - math.sin(angle + math.pi / 6.0) * size,
    )
    draw.polygon([tip, left, right], fill=fill)


def _draw_trajectory(draw: ImageDraw.ImageDraw, size: int) -> None:
    points = _trajectory_points(size)
    width = max(7, size // 28)
    shadow_points = [(x + size * 0.008, y + size * 0.014) for x, y in points]
    _draw_bezier(draw, shadow_points, width + max(2, size // 96), (16, 24, 35, 52))
    _draw_bezier(draw, points, width, _hex_to_rgba("#168447"))

    p_prev = points[-2]
    p_tip = points[-1]
    angle = math.atan2(p_tip[1] - p_prev[1], p_tip[0] - p_prev[0])
    _draw_arrow_head(draw, p_tip, angle, max(18, size // 11), _hex_to_rgba("#168447"))

    electron_radius = max(11, size // 19)
    _draw_sphere(draw, points[0][0], points[0][1], electron_radius, "#159947", "#9df5b6")


def _draw_spin_symbol(draw: ImageDraw.ImageDraw, size: int) -> None:
    center = (size * 0.72, size * 0.64)
    radius = size * 0.112
    arc_width = max(3, size // 64)
    bbox = (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
    draw.arc(bbox, start=35, end=315, fill=(57, 70, 86, 230), width=arc_width)
    arc_angle = math.radians(35)
    tip = (center[0] + math.cos(arc_angle) * radius, center[1] + math.sin(arc_angle) * radius)
    _draw_arrow_head(draw, tip, arc_angle + math.pi / 2.0, max(8, size // 30), (57, 70, 86, 230))

    arrow_width = max(5, size // 42)
    up_start = (center[0] - size * 0.025, center[1] + size * 0.045)
    up_end = (center[0] - size * 0.025, center[1] - size * 0.095)
    down_start = (center[0] + size * 0.04, center[1] - size * 0.045)
    down_end = (center[0] + size * 0.04, center[1] + size * 0.105)
    draw.line((up_start, up_end), fill=_hex_to_rgba("#2563eb"), width=arrow_width)
    _draw_arrow_head(draw, up_end, -math.pi / 2.0, max(9, size // 28), _hex_to_rgba("#2563eb"))
    draw.line((down_start, down_end), fill=_hex_to_rgba("#dc2626"), width=arrow_width)
    _draw_arrow_head(draw, down_end, math.pi / 2.0, max(9, size // 28), _hex_to_rgba("#dc2626"))


def render_icon(size: int) -> Image.Image:
    scale = 4
    canvas_size = size * scale
    image = _gradient_tile(canvas_size, radius=int(canvas_size * 0.18))
    layer = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    _draw_lattice(draw, canvas_size)
    _draw_trajectory(draw, canvas_size)
    _draw_spin_symbol(draw, canvas_size)
    image.alpha_composite(layer)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    source = render_icon(1024)
    source.save(ASSET_DIR / "app_icon.png")
    for size in ICON_SIZES:
        render_icon(size).save(ASSET_DIR / f"app_icon_{size}.png")
    source.save(ASSET_DIR / "app_icon.ico", sizes=[(size, size) for size in ICON_SIZES])


if __name__ == "__main__":
    main()
