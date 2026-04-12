# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/martindokholyan/Desktop/Repos/Visual-Receptive-Sim/main.py'],
    pathex=[],
    binaries=[],
    datas=[('/Users/martindokholyan/Desktop/Repos/Visual-Receptive-Sim/data', 'data'), ('/Users/martindokholyan/Desktop/Repos/Visual-Receptive-Sim/rgbtolms', 'rgbtolms')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RGC-Circuit-Simulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RGC-Circuit-Simulator',
)
app = BUNDLE(
    coll,
    name='RGC-Circuit-Simulator.app',
    icon=None,
    bundle_identifier=None,
)
