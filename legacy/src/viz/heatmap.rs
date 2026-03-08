//! 2D layer heatmaps with colormaps. Maps grid data → egui ColorImage.

use eframe::egui;
use colorgrad::turbo;

use crate::circuit::{ConeType, Retina};
use crate::stimulus::{ColorStimulus, N_WAVELENGTH_BANDS, WAVELENGTH_MIN_NM, WAVELENGTH_STEP_NM};

/// Colormap for layer visualization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colormap {
    /// RF: blue=negative, black=zero, red=positive
    Biphasic,
    /// RGC output: black → yellow → white (spiking activity)
    FiringRate,
    /// Cone absorption: spectral colors
    Spectral,
    /// Color opponent signals
    Diverging,
    /// Grayscale
    Grayscale,
}

/// Convert flat f32 grid to ColorImage using colormap.
pub fn layer_to_image(
    data: &[f32],
    width: usize,
    height: usize,
    colormap: Colormap,
    max_val: f32,
) -> egui::ColorImage {
    let grad = turbo();
    let max_v = max_val.max(1e-6);
    let pixels: Vec<egui::Color32> = data
        .iter()
        .map(|&v| {
            let n = (v / max_v).clamp(0.0, 1.0);
            match colormap {
                Colormap::Grayscale | Colormap::FiringRate => {
                    let b = (n * 255.0) as u8;
                    egui::Color32::from_rgb(b, b, b)
                }
                Colormap::Biphasic => {
                    let c = grad.at(n as f64).to_rgba8();
                    egui::Color32::from_rgb(c[0], c[1], c[2])
                }
                Colormap::Spectral | Colormap::Diverging => {
                    let c = grad.at(n as f64).to_rgba8();
                    egui::Color32::from_rgb(c[0], c[1], c[2])
                }
            }
        })
        .collect();
    egui::ColorImage {
        size: [width, height],
        pixels,
    }
}

/// Stimulus as RGB (LMS → display colors).
pub fn stimulus_rgb(stimulus: &ColorStimulus, w: usize, h: usize) -> Vec<egui::Color32> {
    let mut out = vec![egui::Color32::from_rgb(20, 20, 20); w * h];
    let sw = stimulus.width;
    let sh = stimulus.height;
    for iy in 0..h {
        for ix in 0..w {
            let sy = (iy * sh / h).min(sh - 1);
            let sx = (ix * sw / w).min(sw - 1);
            let (mut l, mut m, mut s) = (0.0f32, 0.0f32, 0.0f32);
            for b in 0..N_WAVELENGTH_BANDS {
                let lam = WAVELENGTH_MIN_NM + b as f32 * WAVELENGTH_STEP_NM;
                let i = stimulus.spectra[[sy, sx, b]];
                l += i * crate::circuit::cones::spectral_sensitivity(ConeType::L, lam);
                m += i * crate::circuit::cones::spectral_sensitivity(ConeType::M, lam);
                s += i * crate::circuit::cones::spectral_sensitivity(ConeType::S, lam);
            }
            let r = (l * 2.0 - m).max(0.0).min(1.0);
            let g = m.max(0.0).min(1.0);
            let b = s.max(0.0).min(1.0);
            out[iy * w + ix] =
                egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8);
        }
    }
    out
}

/// Cone layer (L, M, or S) as grayscale heatmap.
pub fn layer_cones(
    retina: &Retina,
    w: usize,
    h: usize,
    cone_type: ConeType,
) -> Vec<egui::Color32> {
    let mut grid = vec![0.0f32; w * h];
    let n = (retina.cones.len() as f32).sqrt() as usize;
    let cell_w = w / n.max(1);
    let cell_h = h / n.max(1);
    for c in &retina.cones {
        if c.cone_type != cone_type {
            continue;
        }
        let v = c.effective_activation;
        let px = (c.x * w as f32).clamp(0.0, (w - 1) as f32) as usize;
        let py = (c.y * h as f32).clamp(0.0, (h - 1) as f32) as usize;
        let x0 = (px.saturating_sub(cell_w / 2)).min(w.saturating_sub(cell_w));
        let y0 = (py.saturating_sub(cell_h / 2)).min(h.saturating_sub(cell_h));
        for dy in 0..cell_h {
            for dx in 0..cell_w {
                let ix = x0 + dx;
                let iy = y0 + dy;
                if ix < w && iy < h {
                    grid[iy * w + ix] = v;
                }
            }
        }
    }
    let max_v = grid.iter().fold(1e-6f32, |a, &b| a.max(b));
    grid.iter()
        .map(|&v| {
            let b = (v / max_v).clamp(0.0, 1.0) as u8;
            egui::Color32::from_rgb(b, b, b)
        })
        .collect()
}

/// Horizontal cell layer.
pub fn layer_horizontal(retina: &Retina, w: usize, h: usize) -> Vec<egui::Color32> {
    let mut grid = vec![0.0f32; w * h];
    let n = (retina.horizontals.len() as f32).sqrt() as usize;
    let cw = w / n.max(1);
    let ch = h / n.max(1);
    for hc in &retina.horizontals {
        let v = hc.activation;
        let px = (hc.x * w as f32).clamp(0.0, (w - 1) as f32) as usize;
        let py = (hc.y * h as f32).clamp(0.0, (h - 1) as f32) as usize;
        let x0 = (px.saturating_sub(cw / 2)).min(w.saturating_sub(cw));
        let y0 = (py.saturating_sub(ch / 2)).min(h.saturating_sub(ch));
        for dy in 0..ch {
            for dx in 0..cw {
                let ix = x0 + dx;
                let iy = y0 + dy;
                if ix < w && iy < h {
                    grid[iy * w + ix] = v;
                }
            }
        }
    }
    let max_v = grid.iter().fold(1e-6f32, |a, &b| a.max(b));
    grid.iter()
        .map(|&v| {
            let b = (v / max_v).clamp(0.0, 1.0) as u8;
            egui::Color32::from_rgb(b, b, b)
        })
        .collect()
}

/// Bipolar layer (ON or OFF).
pub fn layer_bipolar(retina: &Retina, w: usize, h: usize, on: bool) -> Vec<egui::Color32> {
    let mut grid = vec![0.0f32; w * h];
    let n = (retina.bipolars.len() / 2) as f32;
    let n_xy = n.sqrt() as usize;
    let cw = w / n_xy.max(1);
    let ch = h / n_xy.max(1);
    for b in &retina.bipolars {
        if b.is_on() != on {
            continue;
        }
        let v = b.effective_activation;
        let px = (b.x * w as f32).clamp(0.0, (w - 1) as f32) as usize;
        let py = (b.y * h as f32).clamp(0.0, (h - 1) as f32) as usize;
        let x0 = (px.saturating_sub(cw / 2)).min(w.saturating_sub(cw));
        let y0 = (py.saturating_sub(ch / 2)).min(h.saturating_sub(ch));
        for dy in 0..ch {
            for dx in 0..cw {
                let ix = x0 + dx;
                let iy = y0 + dy;
                if ix < w && iy < h {
                    grid[iy * w + ix] = v;
                }
            }
        }
    }
    let max_v = grid.iter().fold(1e-6f32, |a, &b| a.max(b));
    grid.iter()
        .map(|&v| {
            let b = (v / max_v).clamp(0.0, 1.0) as u8;
            egui::Color32::from_rgb(b, b, b)
        })
        .collect()
}

/// Amacrine layer.
pub fn layer_amacrine(retina: &Retina, w: usize, h: usize) -> Vec<egui::Color32> {
    let mut grid = vec![0.0f32; w * h];
    let n = (retina.amacrines.len() as f32).sqrt() as usize;
    let cw = w / n.max(1);
    let ch = h / n.max(1);
    for a in &retina.amacrines {
        let v = a.activation;
        let px = (a.x * w as f32).clamp(0.0, (w - 1) as f32) as usize;
        let py = (a.y * h as f32).clamp(0.0, (h - 1) as f32) as usize;
        let x0 = (px.saturating_sub(cw / 2)).min(w.saturating_sub(cw));
        let y0 = (py.saturating_sub(ch / 2)).min(h.saturating_sub(ch));
        for dy in 0..ch {
            for dx in 0..cw {
                let ix = x0 + dx;
                let iy = y0 + dy;
                if ix < w && iy < h {
                    grid[iy * w + ix] = v;
                }
            }
        }
    }
    let max_v = grid.iter().fold(1e-6f32, |a, &b| a.max(b));
    grid.iter()
        .map(|&v| {
            let b = (v / max_v).clamp(0.0, 1.0) as u8;
            egui::Color32::from_rgb(b, b, b)
        })
        .collect()
}

/// RGC firing rate as grayscale.
pub fn layer_rgc(retina: &Retina, w: usize, h: usize) -> Vec<egui::Color32> {
    let max_r: f32 = retina.rgcs.iter().map(|r| r.firing_rate).fold(1e-6, f32::max);
    let mut grid = vec![0.0f32; w * h];
    let n = (retina.rgcs.len() / 2) as f32;
    let n_xy = n.sqrt() as usize;
    let cw = w / n_xy.max(1);
    let ch = h / n_xy.max(1);
    for r in &retina.rgcs {
        let v = r.firing_rate / max_r;
        let px = (r.x * w as f32).clamp(0.0, (w - 1) as f32) as usize;
        let py = (r.y * h as f32).clamp(0.0, (h - 1) as f32) as usize;
        let x0 = (px.saturating_sub(cw / 2)).min(w.saturating_sub(cw));
        let y0 = (py.saturating_sub(ch / 2)).min(h.saturating_sub(ch));
        for dy in 0..ch {
            for dx in 0..cw {
                let ix = x0 + dx;
                let iy = y0 + dy;
                if ix < w && iy < h {
                    grid[iy * w + ix] = v;
                }
            }
        }
    }
    grid.iter()
        .map(|&v| {
            let b = (v.clamp(0.0, 1.0) * 255.0) as u8;
            egui::Color32::from_rgb(b, b, b)
        })
        .collect()
}

/// RGC firing rate as Turbo colormap (blue low, red high).
pub fn layer_rgc_heatmap(retina: &Retina, w: usize, h: usize) -> Vec<egui::Color32> {
    let grad = turbo();
    let max_r: f32 = retina.rgcs.iter().map(|r| r.firing_rate).fold(1e-6, f32::max);
    let mut grid = vec![0.0f32; w * h];
    let n = (retina.rgcs.len() / 2) as f32;
    let n_xy = n.sqrt() as usize;
    let cw = w / n_xy.max(1);
    let ch = h / n_xy.max(1);
    for r in &retina.rgcs {
        let v = if max_r > 0.0 {
            r.firing_rate / max_r
        } else {
            0.0
        };
        let px = (r.x * w as f32).clamp(0.0, (w - 1) as f32) as usize;
        let py = (r.y * h as f32).clamp(0.0, (h - 1) as f32) as usize;
        let x0 = (px.saturating_sub(cw / 2)).min(w.saturating_sub(cw));
        let y0 = (py.saturating_sub(ch / 2)).min(h.saturating_sub(ch));
        for dy in 0..ch {
            for dx in 0..cw {
                let ix = x0 + dx;
                let iy = y0 + dy;
                if ix < w && iy < h {
                    grid[iy * w + ix] = v;
                }
            }
        }
    }
    grid.iter()
        .map(|&v| {
            let c = grad.at(v.clamp(0.0, 1.0) as f64).to_rgba8();
            egui::Color32::from_rgb(c[0], c[1], c[2])
        })
        .collect()
}
