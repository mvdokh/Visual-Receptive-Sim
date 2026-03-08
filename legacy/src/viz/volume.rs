//! 3D stack: isometric projection of retinal layers.

use eframe::egui;
use colorgrad::turbo;

use crate::circuit::{ConeType, Retina};
use crate::stimulus::{
    ColorStimulus, N_WAVELENGTH_BANDS, WAVELENGTH_MIN_NM, WAVELENGTH_STEP_NM,
};

/// Draw 3D-style stacked layers: each layer as a horizontal band of "cells" (dots) colored by activation.
pub fn draw_3d_stack(
    ui: &mut egui::Ui,
    retina: &Retina,
    stimulus: &ColorStimulus,
    available: egui::Vec2,
) {
    const N: usize = 16;
    let strip_h = 32.0;
    let label_w = 72.0;
    let grid_w = (available.x - label_w).max(1.0);
    let grid_h = strip_h * 0.9;
    let cell_w = grid_w / N as f32;
    let cell_h = grid_h / N as f32;
    let rad = (cell_w.min(cell_h) * 0.4).max(1.0);
    let grad = turbo();

    let stim_lum_at = |ix: usize, iy: usize| -> f32 {
        let sx = (ix * stimulus.width / N).min(stimulus.width.saturating_sub(1));
        let sy = (iy * stimulus.height / N).min(stimulus.height.saturating_sub(1));
        let (mut l, mut m, mut s) = (0.0f32, 0.0f32, 0.0f32);
        for b in 0..N_WAVELENGTH_BANDS {
            let lam = WAVELENGTH_MIN_NM + b as f32 * WAVELENGTH_STEP_NM;
            let i = stimulus.spectra[[sy, sx, b]];
            l += i * crate::circuit::cones::spectral_sensitivity(ConeType::L, lam);
            m += i * crate::circuit::cones::spectral_sensitivity(ConeType::M, lam);
            s += i * crate::circuit::cones::spectral_sensitivity(ConeType::S, lam);
        }
        (l + m + s) / 3.0
    };

    let labels = [
        "Stimulus",
        "Cones L",
        "Cones M",
        "Cones S",
        "Horizontal",
        "Bp ON",
        "Bp OFF",
        "Amacrine",
        "RGC",
    ];
    let cone_n = (retina.cones.len() as f32).sqrt() as usize;

    for (layer, &label) in labels.iter().enumerate() {
        ui.horizontal(|ui| {
            ui.set_width(label_w);
            ui.label(egui::RichText::new(label).small());
            let grid_rect = ui
                .allocate_rect(
                    egui::Rect::from_min_size(ui.cursor().min, egui::Vec2::new(grid_w, grid_h)),
                    egui::Sense::hover(),
                )
                .rect;
            let painter = ui.painter();
            for iy in 0..N {
                for ix in 0..N {
                    let cx = grid_rect.min.x + (ix as f32 + 0.5) * cell_w;
                    let cy = grid_rect.min.y + (iy as f32 + 0.5) * cell_h;
                    let v = match layer {
                        0 => stim_lum_at(ix.min(stimulus.width.saturating_sub(1)), iy.min(stimulus.height.saturating_sub(1))),
                        1 => retina
                            .cones
                            .get(iy * cone_n + ix)
                            .map(|c| {
                                if c.cone_type == ConeType::L {
                                    c.effective_activation
                                } else {
                                    0.0
                                }
                            })
                            .unwrap_or(0.0),
                        2 => retina
                            .cones
                            .get(iy * cone_n + ix)
                            .map(|c| {
                                if c.cone_type == ConeType::M {
                                    c.effective_activation
                                } else {
                                    0.0
                                }
                            })
                            .unwrap_or(0.0),
                        3 => retina
                            .cones
                            .get(iy * cone_n + ix)
                            .map(|c| {
                                if c.cone_type == ConeType::S {
                                    c.effective_activation
                                } else {
                                    0.0
                                }
                            })
                            .unwrap_or(0.0),
                        4 => retina
                            .horizontals
                            .get(iy * cone_n + ix)
                            .map(|h| h.activation)
                            .unwrap_or(0.0),
                        5 => retina
                            .bipolars
                            .get((iy.min(15) * 16 + ix.min(15)) * 4 + 0)
                            .map(|b| b.effective_activation)
                            .unwrap_or(0.0),
                        6 => retina
                            .bipolars
                            .get((iy.min(15) * 16 + ix.min(15)) * 4 + 1)
                            .map(|b| b.effective_activation)
                            .unwrap_or(0.0),
                        7 => retina
                            .amacrines
                            .get(iy * cone_n + ix)
                            .map(|a| a.activation)
                            .unwrap_or(0.0),
                        8 => {
                            let n_xy = (retina.rgcs.len() / 6) as f32;
                            let n_xy = n_xy.sqrt() as usize;
                            if n_xy == 0 {
                                0.0
                            } else {
                                let base = (iy.min(n_xy - 1) * n_xy + ix.min(n_xy - 1)) * 6;
                                let sum: f32 = (0..6)
                                    .filter_map(|k| retina.rgcs.get(base + k))
                                    .map(|r| r.firing_rate)
                                    .sum();
                                (sum / 6.0 / 100.0).min(1.0)
                            }
                        }
                        _ => 0.0,
                    };
                    let v = v.min(1.0).max(0.0);
                    let (r, g, b) = if layer == 8 {
                        let c = grad.at(v as f64).to_rgba8();
                        (c[0], c[1], c[2])
                    } else {
                        let b = (v * 255.0) as u8;
                        (b, b, b)
                    };
                    painter.circle_filled(
                        egui::Pos2::new(cx, cy),
                        rad,
                        egui::Color32::from_rgb(r, g, b),
                    );
                }
            }
        });
    }
}
