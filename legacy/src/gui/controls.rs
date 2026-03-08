//! Left panel controls: view layer, circuit params, stimulus settings, statistics.

use eframe::egui;

use crate::circuit::{ConeType, RGCType};
use crate::stimulus::StimulusType;

use super::state::{ViewLayer, DRAWN_MASK_SIZE};
use super::CircuitApp;

/// Show the left control panel content (view layer, circuit sliders, stimulus, statistics).
pub fn show(ui: &mut egui::Ui, app: &mut CircuitApp) {
    ui.vertical_centered(|ui| ui.heading("View"));
    ui.separator();
    for layer in ViewLayer::ALL {
        if ui
            .selectable_label(app.view_layer == layer, layer.label())
            .clicked()
        {
            app.view_layer = layer;
            app.dirty = true;
        }
    }

    if app.view_layer == ViewLayer::RGC || app.view_layer == ViewLayer::RGCHeatmap {
        ui.add_space(8.0);
        let (mean, max, min) = {
            let r: Vec<f32> = app.retina.rgcs.iter().map(|x| x.firing_rate).collect();
            let n = r.len() as f32;
            let mean = r.iter().sum::<f32>() / n.max(1.0);
            let max = r.iter().copied().fold(0.0f32, f32::max);
            let min = r.iter().copied().fold(f32::MAX, f32::min);
            (mean, max, min)
        };
        ui.label(egui::RichText::new("Firing (sp/s)").small());
        ui.label(format!("Mean: {:.1}  Max: {:.1}  Min: {:.1}", mean, max, min));
        ui.add_space(4.0);
    }

    ui.add_space(12.0);
    ui.heading("Circuit");
    ui.separator();
    if ui.add(egui::Slider::new(&mut app.retina.horizontal_feedback_strength, 0.0..=2.0).text("H feedback α")).changed() { app.dirty = true; }
    if ui.add(egui::Slider::new(&mut app.retina.amacrine_inhibition_strength, 0.0..=1.0).text("Amacrine γ")).changed() { app.dirty = true; }
    if ui.add(egui::Slider::new(&mut app.retina.rgc_r_max, 1.0..=200.0).text("RGC r_max")).changed() { app.dirty = true; }
    if ui.add(egui::Slider::new(&mut app.retina.rgc_x_half, 0.0..=2.0).text("RGC x_half")).changed() { app.dirty = true; }
    if ui.add(egui::Slider::new(&mut app.retina.rgc_slope, 0.5..=10.0).text("RGC slope")).changed() { app.dirty = true; }
    ui.label("Temporal (realism)");
    if ui.add(egui::Slider::new(&mut app.retina.cone_tau_s, 0.0..=0.15).text("Cone τ (s)")).changed() { app.dirty = true; }
    if ui.add(egui::Slider::new(&mut app.retina.rgc_tau_s, 0.0..=0.15).text("RGC τ (s)")).changed() { app.dirty = true; }

    ui.add_space(12.0);
    ui.heading("Stimulus");
    ui.separator();
    if ui.add(egui::Slider::new(&mut app.resolution_preset, 0..=2).text("Res")).changed() { app.dirty = true; }
    ui.add_space(8.0);

    let mut type_idx = 0u8;
    match &app.stimulus_type {
        StimulusType::MonochromaticSpot { .. } => type_idx = 0,
        StimulusType::WhiteLight { .. } => type_idx = 1,
        StimulusType::BlinkingSpot { .. } => type_idx = 2,
        StimulusType::PulsingSpot { .. } => type_idx = 3,
        StimulusType::MovingSpot { .. } => type_idx = 4,
        StimulusType::DrawnShape { .. } => type_idx = 5,
        StimulusType::UniformLMS { .. } => type_idx = 6,
        StimulusType::FullFieldWavelength { .. } => type_idx = 7,
        StimulusType::DualSpot { .. } => type_idx = 8,
        StimulusType::MultiSpot { .. } => type_idx = 9,
        _ => {}
    }
    let types = [
        "Spot (static)", "White spot", "Blinking spot", "Pulsing spot", "Moving spot",
        "Draw shape", "Uniform field", "Full field λ", "Dual spot", "Multi spot (4)",
    ];
    egui::ComboBox::from_label("Type")
        .selected_text(types[type_idx as usize].to_string())
        .show_ui(ui, |ui| {
            for (i, t) in types.iter().enumerate() {
                if ui.selectable_label(type_idx == i as u8, *t).clicked() {
                    app.dirty = true;
                    match i {
                        0 => app.stimulus_type = StimulusType::MonochromaticSpot { x: 0.5, y: 0.5, r: 0.2, wavelength_nm: 550.0 },
                        1 => app.stimulus_type = StimulusType::WhiteLight { x: 0.5, y: 0.5, r: 0.2 },
                        2 => app.stimulus_type = StimulusType::BlinkingSpot { x: 0.5, y: 0.5, r: 0.2, wavelength_nm: 550.0, period_s: 0.5, duty: 0.5 },
                        3 => app.stimulus_type = StimulusType::PulsingSpot { x: 0.5, y: 0.5, r: 0.2, wavelength_nm: 550.0, period_s: 1.0 },
                        4 => app.stimulus_type = StimulusType::MovingSpot { x0: 0.2, y0: 0.5, x1: 0.8, y1: 0.5, r: 0.15, wavelength_nm: 550.0, period_s: 2.0 },
                        5 => app.stimulus_type = StimulusType::DrawnShape {
                            mask: vec![0.0; DRAWN_MASK_SIZE * DRAWN_MASK_SIZE],
                            width: DRAWN_MASK_SIZE,
                            height: DRAWN_MASK_SIZE,
                            wavelength_nm: 550.0,
                        },
                        6 => app.stimulus_type = StimulusType::UniformLMS { l: 0.5, m: 0.5, s: 0.5 },
                        7 => app.stimulus_type = StimulusType::FullFieldWavelength { wavelength_nm: 550.0 },
                        8 => app.stimulus_type = StimulusType::DualSpot {
                            x1: 0.35, y1: 0.5, r1: 0.15, wavelength_nm1: 450.0,
                            x2: 0.65, y2: 0.5, r2: 0.15, wavelength_nm2: 620.0,
                        },
                        9 => app.stimulus_type = StimulusType::MultiSpot {
                            spots: [(0.3, 0.4, 0.12, 420.0), (0.7, 0.4, 0.12, 530.0), (0.3, 0.6, 0.12, 560.0), (0.7, 0.6, 0.12, 650.0)],
                        },
                        _ => {}
                    }
                    ui.close_menu();
                }
            }
        });

    match &mut app.stimulus_type {
        StimulusType::MonochromaticSpot { x, y, r, wavelength_nm } => {
            if ui.add(egui::Slider::new(x, 0.0..=1.0).text("X")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(y, 0.0..=1.0).text("Y")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(r, 0.05..=0.5).text("Radius")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(wavelength_nm, 380.0..=700.0).text("λ (nm)")).changed() { app.dirty = true; }
        }
        StimulusType::WhiteLight { x, y, r } => {
            if ui.add(egui::Slider::new(x, 0.0..=1.0).text("X")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(y, 0.0..=1.0).text("Y")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(r, 0.05..=0.5).text("Radius")).changed() { app.dirty = true; }
        }
        StimulusType::BlinkingSpot { x, y, r, wavelength_nm, period_s, duty } => {
            if ui.add(egui::Slider::new(x, 0.0..=1.0).text("X")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(y, 0.0..=1.0).text("Y")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(r, 0.05..=0.5).text("Radius")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(wavelength_nm, 380.0..=700.0).text("λ (nm)")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(period_s, 0.1..=2.0).text("Period (s)")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(duty, 0.1..=0.9).text("Duty")).changed() { app.dirty = true; }
        }
        StimulusType::PulsingSpot { x, y, r, wavelength_nm, period_s } => {
            if ui.add(egui::Slider::new(x, 0.0..=1.0).text("X")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(y, 0.0..=1.0).text("Y")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(r, 0.05..=0.5).text("Radius")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(wavelength_nm, 380.0..=700.0).text("λ (nm)")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(period_s, 0.2..=3.0).text("Period (s)")).changed() { app.dirty = true; }
        }
        StimulusType::MovingSpot { x0, y0, x1, y1, r, wavelength_nm, period_s } => {
            if ui.add(egui::Slider::new(x0, 0.0..=1.0).text("X0")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(y0, 0.0..=1.0).text("Y0")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(x1, 0.0..=1.0).text("X1")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(y1, 0.0..=1.0).text("Y1")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(r, 0.05..=0.4).text("Radius")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(wavelength_nm, 380.0..=700.0).text("λ (nm)")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(period_s, 0.5..=5.0).text("Period (s)")).changed() { app.dirty = true; }
        }
        StimulusType::DrawnShape { mask, wavelength_nm, .. } => {
            app.drawn_mask = mask.clone();
            if ui.add(egui::Slider::new(wavelength_nm, 380.0..=700.0).text("λ (nm)")).changed() { app.dirty = true; }
            ui.add(egui::Slider::new(&mut app.drawn_mask_brush, 0.1..=1.0).text("Brush"));
            ui.label("Draw below (click/drag)");
        }
        StimulusType::UniformLMS { l, m, s } => {
            if ui.add(egui::Slider::new(l, 0.0..=1.0).text("L")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(m, 0.0..=1.0).text("M")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(s, 0.0..=1.0).text("S")).changed() { app.dirty = true; }
        }
        StimulusType::FullFieldWavelength { wavelength_nm } => {
            if ui.add(egui::Slider::new(wavelength_nm, 380.0..=700.0).text("λ (nm)")).changed() { app.dirty = true; }
        }
        StimulusType::DualSpot { x1, y1, r1, wavelength_nm1, x2, y2, r2, wavelength_nm2 } => {
            if ui.add(egui::Slider::new(x1, 0.0..=1.0).text("X1")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(y1, 0.0..=1.0).text("Y1")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(r1, 0.05..=0.4).text("R1")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(wavelength_nm1, 380.0..=700.0).text("λ1")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(x2, 0.0..=1.0).text("X2")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(y2, 0.0..=1.0).text("Y2")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(r2, 0.05..=0.4).text("R2")).changed() { app.dirty = true; }
            if ui.add(egui::Slider::new(wavelength_nm2, 380.0..=700.0).text("λ2")).changed() { app.dirty = true; }
        }
        StimulusType::MultiSpot { spots } => {
            for (i, (x, y, r, wl)) in spots.iter_mut().enumerate() {
                ui.collapsing(format!("Spot {}", i + 1), |ui| {
                    if ui.add(egui::Slider::new(x, 0.0..=1.0).text("X")).changed() { app.dirty = true; }
                    if ui.add(egui::Slider::new(y, 0.0..=1.0).text("Y")).changed() { app.dirty = true; }
                    if ui.add(egui::Slider::new(r, 0.0..=0.4).text("R")).changed() { app.dirty = true; }
                    if ui.add(egui::Slider::new(wl, 380.0..=700.0).text("λ")).changed() { app.dirty = true; }
                });
            }
        }
        _ => {}
    }

    ui.add_space(8.0);
    egui::CollapsingHeader::new("Statistics")
        .default_open(true)
        .show(ui, |ui| {
            let r = &app.retina;
            let l_cone: f32 = r.cones.iter().filter(|c| c.cone_type == ConeType::L).map(|c| c.effective_activation).sum::<f32>();
            let m_cone: f32 = r.cones.iter().filter(|c| c.cone_type == ConeType::M).map(|c| c.effective_activation).sum::<f32>();
            let s_cone: f32 = r.cones.iter().filter(|c| c.cone_type == ConeType::S).map(|c| c.effective_activation).sum::<f32>();
            let n_l = r.cones.iter().filter(|c| c.cone_type == ConeType::L).count().max(1);
            let n_m = r.cones.iter().filter(|c| c.cone_type == ConeType::M).count().max(1);
            let n_s = r.cones.iter().filter(|c| c.cone_type == ConeType::S).count().max(1);
            ui.label("Cones (mean act):");
            ui.label(format!("  L: {:.3}  M: {:.3}  S: {:.3}", l_cone / n_l as f32, m_cone / n_m as f32, s_cone / n_s as f32));
            let on_act: f32 = r.bipolars.iter().filter(|b| b.is_on()).map(|b| b.effective_activation).sum();
            let off_act: f32 = r.bipolars.iter().filter(|b| !b.is_on()).map(|b| b.effective_activation).sum();
            let n_on = r.bipolars.iter().filter(|b| b.is_on()).count().max(1);
            let n_off = r.bipolars.iter().filter(|b| !b.is_on()).count().max(1);
            ui.label("Bipolars (mean):");
            ui.label(format!("  ON: {:.3}  OFF: {:.3}", on_act / n_on as f32, off_act / n_off as f32));
            let h_mean: f32 = r.horizontals.iter().map(|h| h.activation).sum::<f32>() / r.horizontals.len().max(1) as f32;
            let a_mean: f32 = r.amacrines.iter().map(|a| a.activation).sum::<f32>() / r.amacrines.len().max(1) as f32;
            ui.label(format!("H mean: {:.3}  A mean: {:.3}", h_mean, a_mean));
            ui.label("RGC by type (sp/s):");
            for rgc_type in [RGCType::MidgetON_L, RGCType::MidgetOFF_L, RGCType::MidgetON_M, RGCType::MidgetOFF_M, RGCType::ParasolON, RGCType::ParasolOFF] {
                let cells: Vec<f32> = r.rgcs.iter().filter(|x| x.rgc_type == rgc_type).map(|x| x.firing_rate).collect();
                let (mean, max, min) = if cells.is_empty() {
                    (0.0f32, 0.0, 0.0)
                } else {
                    let mean = cells.iter().sum::<f32>() / cells.len() as f32;
                    let max = cells.iter().copied().fold(0.0f32, f32::max);
                    let min = cells.iter().copied().fold(f32::MAX, f32::min);
                    (mean, max, min)
                };
                ui.label(format!("  {:?}: n={} mean={:.1} max={:.1} min={:.1}", rgc_type, cells.len(), mean, max, min));
            }
        });
}
