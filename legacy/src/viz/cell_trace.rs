//! RF emergence trace: click RGC → sweep probe → show receptive field heatmap.
//! Full probe sweep: 32×32 positions, small spot, record RGC firing (emergent from circuit).

use eframe::egui;

use crate::circuit::retina::simulate_frame;
use crate::circuit::Retina;
use crate::stimulus::{render_stimulus, StimulusType};

const RF_GRID_SIZE: usize = 32;
const PROBE_RADIUS: f32 = 0.04; // Small spot for RF mapping
const PROBE_WAVELENGTH_NM: f32 = 550.0;

/// State for RF trace computation.
#[derive(Default)]
pub struct CellTraceState {
    /// Index of selected RGC, if any.
    pub selected_rgc: Option<usize>,
    /// Cached RF heatmap (32×32) from probe sweep. None until computed.
    pub rf_grid: Option<Vec<f32>>,
    /// Whether RF computation is in progress.
    pub computing: bool,
}

/// Sweep a small spot across 32×32 positions, run simulation at each, record selected RGC firing.
/// Uses tau=0 for instant response so each probe gives a clean RF sample.
fn compute_rf_sweep(retina: &Retina, rgc_index: usize, stimulus_size: (usize, usize)) -> Vec<f32> {
    let (sw, sh) = stimulus_size;
    let mut grid = vec![0.0f32; RF_GRID_SIZE * RF_GRID_SIZE];
    let mut work = retina.clone();
    work.cone_tau_s = 0.0;
    work.rgc_tau_s = 0.0;
    let dt_s = 0.02f32;
    for iy in 0..RF_GRID_SIZE {
        for ix in 0..RF_GRID_SIZE {
            let x = (ix as f32 + 0.5) / RF_GRID_SIZE as f32;
            let y = (iy as f32 + 0.5) / RF_GRID_SIZE as f32;
            let stim = StimulusType::MonochromaticSpot {
                x,
                y,
                r: PROBE_RADIUS,
                wavelength_nm: PROBE_WAVELENGTH_NM,
            };
            let stimulus = render_stimulus(&stim, sw, sh, 0.0);
            simulate_frame(&mut work, &stimulus, dt_s);
            let firing = work
                .rgcs
                .get(rgc_index)
                .map(|r| r.firing_rate)
                .unwrap_or(0.0);
            grid[iy * RF_GRID_SIZE + ix] = firing;
        }
    }
    grid
}

/// Show RF trace panel. When selected_rgc is set, displays "Computing RF..." or the RF heatmap.
pub fn show(ui: &mut egui::Ui, retina: &Retina, state: &mut CellTraceState) {
    ui.heading("Receptive Field");
    ui.add_space(4.0);

    if let Some(idx) = state.selected_rgc {
        if let Some(rgc) = retina.rgcs.get(idx) {
            ui.label(format!("{:?} at ({:.2}, {:.2})", rgc.rgc_type, rgc.x, rgc.y));
            ui.label(format!("Firing: {:.1} sp/s", rgc.firing_rate));
            ui.add_space(8.0);

            if state.computing {
                ui.label("Computing RF... (sweep probe across 32×32 grid)");
            } else if let Some(ref grid) = state.rf_grid {
                // Draw 32×32 RF heatmap
                let sz = 128.0f32;
                let (rect, _) = ui.allocate_exact_size(
                    egui::Vec2::new(sz, sz),
                    egui::Sense::hover(),
                );
                let max_v = grid.iter().fold(1e-6f32, |a, &b| a.max(b));
                let n = RF_GRID_SIZE;
                let cell_sz = sz / n as f32;
                let painter = ui.painter();
                for iy in 0..n {
                    for ix in 0..n {
                        let v = grid.get(iy * n + ix).copied().unwrap_or(0.0) / max_v;
                        let v = v.clamp(0.0, 1.0);
                        let gray = (v * 255.0) as u8;
                        let x = rect.min.x + ix as f32 * cell_sz;
                        let y = rect.min.y + iy as f32 * cell_sz;
                        painter.rect_filled(
                            egui::Rect::from_min_size(
                                egui::Pos2::new(x, y),
                                egui::Vec2::new(cell_sz + 1.0, cell_sz + 1.0),
                            ),
                            0.0,
                            egui::Color32::from_rgb(gray, gray, gray),
                        );
                    }
                }
                ui.label("RF from probe sweep (emergent from circuit)");
            } else {
                ui.label("Click 'Compute RF' to sweep a small probe and measure response.");
                if ui.button("Compute RF").clicked() {
                    state.computing = true;
                    state.rf_grid = Some(compute_rf_sweep(retina, idx, (256, 256)));
                    state.computing = false;
                }
            }
        }
    } else {
        ui.label("Click an RGC in the main view to trace its receptive field.");
    }
}
