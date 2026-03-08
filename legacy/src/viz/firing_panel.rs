//! Firing & Statistics panel: per-cell-type bars, signal flow trace, histogram.

use eframe::egui;

use crate::circuit::{ConeType, Retina, RGCType};

/// Show the Firing & Statistics panel.
pub fn show(ui: &mut egui::Ui, retina: &Retina, selected_rgc_index: Option<usize>) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.heading("Firing by cell type");
        ui.add_space(4.0);

        // Per-cell-type mean firing (horizontal bar chart)
        let rgc_types = [
            RGCType::MidgetON_L,
            RGCType::MidgetOFF_L,
            RGCType::MidgetON_M,
            RGCType::MidgetOFF_M,
            RGCType::ParasolON,
            RGCType::ParasolOFF,
        ];
        let max_firing = 150.0f32; // typical max for bar scale
        for rgc_type in rgc_types {
            let cells: Vec<f32> = retina
                .rgcs
                .iter()
                .filter(|x| x.rgc_type == rgc_type)
                .map(|x| x.firing_rate)
                .collect();
            let mean = if cells.is_empty() {
                0.0
            } else {
                cells.iter().sum::<f32>() / cells.len() as f32
            };
            let bar_w = (mean / max_firing).min(1.0) * 120.0;
            let name = format!("{:?}", rgc_type);
            ui.horizontal(|ui| {
                ui.set_width(140.0);
                ui.label(egui::RichText::new(name).small());
                ui.add_space(4.0);
                let rect = ui.allocate_rect(
                    egui::Rect::from_min_size(ui.cursor().min, egui::Vec2::new(120.0, 12.0)),
                    egui::Sense::hover(),
                )
                .rect;
                let painter = ui.painter();
                painter.rect_filled(
                    egui::Rect::from_min_size(rect.min, egui::Vec2::new(120.0, 12.0)),
                    2.0,
                    egui::Color32::from_rgb(40, 40, 40),
                );
                painter.rect_filled(
                    egui::Rect::from_min_size(rect.min, egui::Vec2::new(bar_w.max(1.0), 12.0)),
                    2.0,
                    egui::Color32::from_rgb(100, 180, 255),
                );
                ui.add_space(130.0);
                ui.label(format!("{:.1} sp/s", mean));
            });
        }

        ui.add_space(12.0);
        ui.separator();
        ui.heading("Signal flow (selected cell)");
        ui.add_space(4.0);

        if let Some(idx) = selected_rgc_index {
            if let Some(rgc) = retina.rgcs.get(idx) {
                ui.label(format!("{:?} at ({:.2}, {:.2})", rgc.rgc_type, rgc.x, rgc.y));
                ui.label(format!("Generator: {:.3}", rgc.activation));
                ui.label(format!("Firing rate: {:.1} sp/s", rgc.firing_rate));
                ui.add_space(4.0);
            }
        } else {
            ui.label(egui::RichText::new("Click an RGC in the main view to see its signal flow.").small());
        }

        ui.add_space(12.0);
        ui.separator();
        ui.heading("Population summary");
        ui.add_space(4.0);

        // Cones
        let l_cone: f32 = retina
            .cones
            .iter()
            .filter(|c| c.cone_type == ConeType::L)
            .map(|c| c.effective_activation)
            .sum();
        let m_cone: f32 = retina
            .cones
            .iter()
            .filter(|c| c.cone_type == ConeType::M)
            .map(|c| c.effective_activation)
            .sum();
        let s_cone: f32 = retina
            .cones
            .iter()
            .filter(|c| c.cone_type == ConeType::S)
            .map(|c| c.effective_activation)
            .sum();
        let n_l = retina.cones.iter().filter(|c| c.cone_type == ConeType::L).count().max(1);
        let n_m = retina.cones.iter().filter(|c| c.cone_type == ConeType::M).count().max(1);
        let n_s = retina.cones.iter().filter(|c| c.cone_type == ConeType::S).count().max(1);
        ui.label("Cones (mean act):");
        ui.label(format!(
            "  L: {:.3}  M: {:.3}  S: {:.3}",
            l_cone / n_l as f32,
            m_cone / n_m as f32,
            s_cone / n_s as f32
        ));

        // Bipolars
        let on_act: f32 = retina
            .bipolars
            .iter()
            .filter(|b| b.is_on())
            .map(|b| b.effective_activation)
            .sum();
        let off_act: f32 = retina
            .bipolars
            .iter()
            .filter(|b| !b.is_on())
            .map(|b| b.effective_activation)
            .sum();
        let n_on = retina.bipolars.iter().filter(|b| b.is_on()).count().max(1);
        let n_off = retina.bipolars.iter().filter(|b| !b.is_on()).count().max(1);
        ui.label("Bipolars (mean):");
        ui.label(format!(
            "  ON: {:.3}  OFF: {:.3}",
            on_act / n_on as f32,
            off_act / n_off as f32
        ));

        let h_mean: f32 = retina
            .horizontals
            .iter()
            .map(|h| h.activation)
            .sum::<f32>()
            / retina.horizontals.len().max(1) as f32;
        let a_mean: f32 = retina
            .amacrines
            .iter()
            .map(|a| a.activation)
            .sum::<f32>()
            / retina.amacrines.len().max(1) as f32;
        ui.label(format!("H mean: {:.3}  A mean: {:.3}", h_mean, a_mean));

        // Color opponent
        let l_minus_m = (l_cone / n_l as f32) - (m_cone / n_m as f32);
        let s_minus_lm = (s_cone / n_s as f32) - ((l_cone / n_l as f32 + m_cone / n_m as f32) / 2.0);
        let lum = (l_cone / n_l as f32 + m_cone / n_m as f32) / 2.0;
        ui.add_space(8.0);
        ui.label("Color opponent:");
        ui.label(format!("  L−M: {:+.3}  S−(L+M): {:+.3}", l_minus_m, s_minus_lm));
        ui.label(format!("  Luminance (L+M): {:.3}", lum));
    });
}
