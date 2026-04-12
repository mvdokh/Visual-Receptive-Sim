from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


APP_NAME = "RGC-Circuit-Simulator"
ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "main.py"
PACKAGING_DIR = ROOT / "packaging"
DIST_ROOT = PACKAGING_DIR / "dist"
BUILD_ROOT = PACKAGING_DIR / "build"
SPECS_ROOT = PACKAGING_DIR / "specs"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def _detect_target() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    raise RuntimeError(f"Unsupported platform: {system}")


def _add_data_arg(src: Path, dest_name: str, target: str) -> str:
    sep = ";" if target == "windows" else ":"
    return f"{src}{sep}{dest_name}"


def _clean(paths: list[Path]) -> None:
    for p in paths:
        if p.exists():
            shutil.rmtree(p)


def _build_with_pyinstaller(target: str, clean: bool) -> Path:
    target_dist = DIST_ROOT / target
    target_build = BUILD_ROOT / target
    target_specs = SPECS_ROOT / target

    if clean:
        _clean([target_dist, target_build, target_specs])

    target_dist.mkdir(parents=True, exist_ok=True)
    target_build.mkdir(parents=True, exist_ok=True)
    target_specs.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--windowed",
        "--name",
        APP_NAME,
        "--distpath",
        str(target_dist),
        "--workpath",
        str(target_build),
        "--specpath",
        str(target_specs),
        "--add-data",
        _add_data_arg(ROOT / "data", "data", target),
        "--add-data",
        _add_data_arg(ROOT / "rgbtolms", "rgbtolms", target),
        str(ENTRYPOINT),
    ]

    if target in {"windows", "linux"}:
        cmd.insert(3, "--onefile")

    _run(cmd)
    return target_dist


def _package_macos(dist_dir: Path, skip_dmg: bool) -> None:
    app_bundle = dist_dir / f"{APP_NAME}.app"
    if not app_bundle.exists():
        raise RuntimeError(f"Expected app bundle not found: {app_bundle}")

    if skip_dmg:
        print(f"Created macOS app bundle: {app_bundle}")
        return

    dmg_path = dist_dir / f"{APP_NAME}.dmg"
    staging_dir = dist_dir / "_dmg_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(app_bundle, staging_dir / app_bundle.name)

    _run(
        [
            "hdiutil",
            "create",
            "-volname",
            APP_NAME,
            "-srcfolder",
            str(staging_dir),
            "-ov",
            "-format",
            "UDZO",
            str(dmg_path),
        ]
    )
    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"Created DMG: {dmg_path}")


def _package_linux(dist_dir: Path) -> None:
    binary_path = dist_dir / APP_NAME
    if not binary_path.exists():
        raise RuntimeError(f"Expected Linux binary not found: {binary_path}")

    archive_path = dist_dir / f"{APP_NAME}-linux.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(binary_path, arcname=APP_NAME)
    print(f"Created Linux archive: {archive_path}")


def _package_windows(dist_dir: Path) -> None:
    exe_path = dist_dir / f"{APP_NAME}.exe"
    if not exe_path.exists():
        raise RuntimeError(f"Expected Windows executable not found: {exe_path}")
    print(f"Created Windows executable: {exe_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build desktop binaries with PyInstaller. "
            "Run on the native OS you want to package."
        )
    )
    parser.add_argument(
        "--target",
        choices=["auto", "macos", "windows", "linux"],
        default="auto",
        help="Target platform. Default: auto-detect from current OS.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean packaging/build outputs for the selected target before building.",
    )
    parser.add_argument(
        "--skip-dmg",
        action="store_true",
        help="macOS only: skip DMG creation and keep only the .app bundle.",
    )
    args = parser.parse_args()

    target = _detect_target() if args.target == "auto" else args.target
    current = _detect_target()
    if target != current:
        raise RuntimeError(
            f"Cross-compilation is not configured (current={current}, target={target}). "
            "Run this script on the target OS or use the GitHub Actions workflow."
        )

    dist_dir = _build_with_pyinstaller(target=target, clean=args.clean)

    if target == "macos":
        _package_macos(dist_dir, skip_dmg=args.skip_dmg)
    elif target == "windows":
        _package_windows(dist_dir)
    elif target == "linux":
        _package_linux(dist_dir)

    print(f"Build complete. Artifacts are in: {dist_dir}")


if __name__ == "__main__":
    main()
