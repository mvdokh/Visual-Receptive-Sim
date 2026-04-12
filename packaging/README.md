Packaging

This directory contains everything needed to build native executables for the simulator.

Artifacts by OS:
- macOS: .app and .dmg
- Windows: .exe
- Linux: native binary and .tar.gz archive

Quick start

1. Create/activate a Python environment.
2. Install runtime + build dependencies:

pip install -r requirements.txt -r packaging/requirements-build.txt

3. Run the build script:

python packaging/build.py --clean

Outputs are written to packaging/dist/<platform>/.

Examples

- macOS app only (skip DMG):
python packaging/build.py --clean --skip-dmg

- Explicit target (must match current OS):
python packaging/build.py --target macos --clean

Notes

- Cross-compiling across OSes is not enabled in this script.

Automatic Releases On Push

The workflow .github/workflows/build-binaries.yml runs automatically on pushes to main/master and will:
- build macOS, Windows, and Linux binaries,
- create a GitHub prerelease,
- upload release files (.dmg, .exe, .tar.gz).

Manual trigger is also available via Actions -> Build Binaries And Release -> Run workflow.
