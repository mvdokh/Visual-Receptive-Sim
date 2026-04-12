#!/usr/bin/env python3
"""
Render rgc_simulator_icon.svg into PyInstaller assets: app.ico (all hosts) and app.icns (macOS only).

Requires CairoSVG + Pillow (see packaging/requirements-build.txt).
On macOS, also requires /usr/bin/iconutil for .icns.

Fallback: if CairoSVG is unavailable, tries rsvg-convert (e.g. apt install librsvg2-bin).
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _svg_to_png_cairo(svg_path: Path, size: int) -> bytes:
    import cairosvg

    return cairosvg.svg2png(url=str(svg_path), output_width=size, output_height=size)


def _svg_to_png_rsvg(svg_path: Path, size: int) -> bytes:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        raise FileNotFoundError("rsvg-convert not found")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        subprocess.run(
            [
                rsvg,
                "-w",
                str(size),
                "-h",
                str(size),
                "-o",
                str(out),
                str(svg_path),
            ],
            check=True,
            capture_output=True,
        )
        return out.read_bytes()
    finally:
        out.unlink(missing_ok=True)


def _load_rgba_png(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGBA")


def render_square(svg_path: Path, size: int):
    try:
        png = _svg_to_png_cairo(svg_path, size)
    except Exception:
        png = _svg_to_png_rsvg(svg_path, size)
    return _load_rgba_png(png)


def write_ico(svg_path: Path, ico_path: Path) -> None:
    from PIL import Image

    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [render_square(svg_path, s) for s in sizes]
    ico_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:],
    )


def write_icns(svg_path: Path, icns_path: Path) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("app.icns can only be built on macOS (iconutil).")
    iconutil = shutil.which("iconutil")
    if not iconutil:
        raise RuntimeError("iconutil not found (expected on macOS).")

    specs = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]
    out_dir = icns_path.parent
    iconset = out_dir / "AppIcon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)

    for name, sz in specs:
        render_square(svg_path, sz).save(iconset / name, format="PNG")

    subprocess.run(
        [iconutil, "-c", "icns", str(iconset), "-o", str(icns_path)],
        check=True,
    )
    shutil.rmtree(iconset, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build app.ico / app.icns from SVG.")
    p.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="Path to source SVG (default: repo root rgc_simulator_icon.svg)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for app.ico and app.icns",
    )
    p.add_argument(
        "--icns",
        action="store_true",
        help="Also build app.icns (macOS only; no-op elsewhere)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    svg = args.svg or (root / "rgc_simulator_icon.svg")
    if not svg.is_file():
        raise SystemExit(f"SVG not found: {svg}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ico_path = args.out_dir / "app.ico"
    write_ico(svg, ico_path)
    print(f"Wrote {ico_path}")

    if args.icns:
        icns_path = args.out_dir / "app.icns"
        write_icns(svg, icns_path)
        print(f"Wrote {icns_path}")


if __name__ == "__main__":
    main()
