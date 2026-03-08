//! GUI: central viewer, throttled updates, time-varying stimuli, draw canvas, firing heatmap.

mod controls;
mod layout;
mod state;

use std::time::Instant;

use eframe::egui;
use colorgrad::turbo;

use crate::circuit::{Retina, ConeType};
use crate::circuit::retina::simulate_frame;
use crate::stimulus::{
    render_stimulus, N_WAVELENGTH_BANDS, StimulusType, WAVELENGTH_MIN_NM, WAVELENGTH_STEP_NM,
};
use crate::viz::{build_layer_stack, draw_3d_stack, heatmap, show_firing_panel};

pub use state::{ViewLayer, DRAWN_MASK_SIZE, LAYER_THUMB_SIZE, REPAINT_INTERVAL_MS};
use state::{VIEW_SIZE_HIGH, VIEW_SIZE_MED, VIEW_SIZE_LOW};

pub struct CircuitApp {
    pub retina: Retina,
    pub stimulus_type: StimulusType,
    pub stimulus_size: (usize, usize),
    pub view_layer: ViewLayer,
    viewer_image: egui::ColorImage,
    /// Simulation time for blinking/pulsing/moving (seconds)
    sim_time_s: f32,
    last_frame: Option<Instant>,
    /// Only run sim + rebuild when something changed (avoids lag on slider drag)
    dirty: bool,
    /// User-drawn mask for DrawnShape (DRAWN_MASK_SIZE^2)
    drawn_mask: Vec<f32>,
    drawn_mask_brush: f32,
    firing_history: Vec<f32>,
    firing_history_max_len: usize,
    layer_stack_images: Vec<egui::ColorImage>,
    /// 0=Low(256), 1=Med(384), 2=High(512) — reduces lag at lower resolution
    pub resolution_preset: u8,
    /// Selected RGC index for signal flow trace in Firing panel
    pub selected_rgc_index: Option<usize>,
}

#[allow(dead_code)]
impl CircuitApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let retina = Retina::build_demo_grid(16, (10, 5, 1));
        let stimulus_type = StimulusType::MonochromaticSpot {
            x: 0.5,
            y: 0.5,
            r: 0.2,
            wavelength_nm: 550.0,
        };
        let drawn_mask = vec![0.0; DRAWN_MASK_SIZE * DRAWN_MASK_SIZE];
        CircuitApp {
            stimulus_type,
            stimulus_size: (256, 256),
            retina,
            view_layer: ViewLayer::Stimulus,
            viewer_image: egui::ColorImage {
                size: [VIEW_SIZE_MED, VIEW_SIZE_MED],
                pixels: vec![egui::Color32::GRAY; VIEW_SIZE_MED * VIEW_SIZE_MED],
            },
            sim_time_s: 0.0,
            last_frame: None,
            dirty: true,
            drawn_mask,
            drawn_mask_brush: 0.5,
            firing_history: Vec::with_capacity(200),
            firing_history_max_len: 200,
            layer_stack_images: vec![],
            resolution_preset: 1, // Med default
            selected_rgc_index: None,
        }
    }

    fn view_size(&self) -> usize {
        match self.resolution_preset {
            0 => VIEW_SIZE_LOW,
            2 => VIEW_SIZE_HIGH,
            _ => VIEW_SIZE_MED,
        }
    }

    fn advance_time(&mut self, dt_s: f32) {
        self.sim_time_s += dt_s;
    }

    fn run_simulation(&mut self, dt_s: f32) {
        let (sw, sh) = self.stimulus_size;
        let stimulus = render_stimulus(&self.stimulus_type, sw, sh, self.sim_time_s);
        simulate_frame(&mut self.retina, &stimulus, dt_s);
    }

    fn build_viewer_image(&mut self) {
        // Simulation already runs every frame in update(); just rebuild the texture from current retina state
        // Update firing history for plot
        let mean_firing: f32 = self.retina.rgcs.iter().map(|r| r.firing_rate).sum::<f32>()
            / self.retina.rgcs.len().max(1) as f32;
        self.firing_history.push(mean_firing);
        if self.firing_history.len() > self.firing_history_max_len {
            self.firing_history.remove(0);
        }

        let w = self.view_size();
        let h = self.view_size();
        let (sw, sh) = self.stimulus_size;
        let stimulus = render_stimulus(&self.stimulus_type, sw, sh, self.sim_time_s);

        let _thumb = LAYER_THUMB_SIZE;
        if self.view_layer == ViewLayer::AllLayers || self.view_layer == ViewLayer::Stack3D {
            let stack = build_layer_stack(&self.retina, &stimulus);
            self.layer_stack_images = stack.into_iter().map(|(_, img)| img).collect();
            self.viewer_image = self.layer_stack_images.first().cloned().unwrap_or_else(|| egui::ColorImage { size: [w, h], pixels: vec![egui::Color32::GRAY; w * h] });
        }
        if self.view_layer != ViewLayer::AllLayers && self.view_layer != ViewLayer::Stack3D {
            let pixels: Vec<egui::Color32> = match self.view_layer {
                ViewLayer::Stimulus => heatmap::stimulus_rgb(&stimulus, w, h),
                ViewLayer::ConesL => heatmap::layer_cones(&self.retina, w, h, ConeType::L),
                ViewLayer::ConesM => heatmap::layer_cones(&self.retina, w, h, ConeType::M),
                ViewLayer::ConesS => heatmap::layer_cones(&self.retina, w, h, ConeType::S),
                ViewLayer::Horizontal => heatmap::layer_horizontal(&self.retina, w, h),
                ViewLayer::BipolarOn => heatmap::layer_bipolar(&self.retina, w, h, true),
                ViewLayer::BipolarOff => heatmap::layer_bipolar(&self.retina, w, h, false),
                ViewLayer::Amacrine => heatmap::layer_amacrine(&self.retina, w, h),
                ViewLayer::RGC => heatmap::layer_rgc(&self.retina, w, h),
                ViewLayer::RGCHeatmap => heatmap::layer_rgc_heatmap(&self.retina, w, h),
                ViewLayer::AllLayers | ViewLayer::Stack3D => unreachable!(),
            };
            self.viewer_image = egui::ColorImage { size: [w, h], pixels };
        }
        self.dirty = false;
    }

    #[allow(dead_code)]
    fn layer_cones(
        retina: &Retina,
        w: usize,
        h: usize,
        cone_type: ConeType,
        to_byte: impl Fn(f32) -> u8,
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
                let b = to_byte(v / max_v);
                egui::Color32::from_rgb(b, b, b)
            })
            .collect()
    }

    fn layer_horizontal(
        retina: &Retina,
        w: usize,
        h: usize,
        to_byte: impl Fn(f32) -> u8,
    ) -> Vec<egui::Color32> {
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
                let b = to_byte(v / max_v);
                egui::Color32::from_rgb(b, b, b)
            })
            .collect()
    }

    fn layer_bipolar(
        retina: &Retina,
        w: usize,
        h: usize,
        on: bool,
        to_byte: impl Fn(f32) -> u8,
    ) -> Vec<egui::Color32> {
        let mut grid = vec![0.0f32; w * h];
        let n = (retina.bipolars.len() / 2) as f32;
        let n_xy = (n.sqrt()) as usize;
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
                let b = to_byte(v / max_v);
                egui::Color32::from_rgb(b, b, b)
            })
            .collect()
    }

    fn layer_amacrine(
        retina: &Retina,
        w: usize,
        h: usize,
        to_byte: impl Fn(f32) -> u8,
    ) -> Vec<egui::Color32> {
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
                let b = to_byte(v / max_v);
                egui::Color32::from_rgb(b, b, b)
            })
            .collect()
    }

    fn layer_rgc(
        retina: &Retina,
        w: usize,
        h: usize,
        to_byte: impl Fn(f32) -> u8,
    ) -> Vec<egui::Color32> {
        let mut grid = vec![0.0f32; w * h];
        let n = (retina.rgcs.len() / 2) as f32;
        let n_xy = (n.sqrt()) as usize;
        let cw = w / n_xy.max(1);
        let ch = h / n_xy.max(1);
        let max_r: f32 = retina.rgcs.iter().map(|r| r.firing_rate).fold(1e-6, f32::max);
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
                let b = to_byte(v);
                egui::Color32::from_rgb(b, b, b)
            })
            .collect()
    }

    /// RGC firing rate as color heatmap (Turbo: blue low, red high)
    fn layer_rgc_heatmap(retina: &Retina, w: usize, h: usize) -> Vec<egui::Color32> {
        let grad = turbo();
        let max_r: f32 = retina.rgcs.iter().map(|r| r.firing_rate).fold(1e-6, f32::max);
        let mut grid = vec![0.0f32; w * h];
        let n = (retina.rgcs.len() / 2) as f32;
        let n_xy = (n.sqrt()) as usize;
        let cw = w / n_xy.max(1);
        let ch = h / n_xy.max(1);
        for r in &retina.rgcs {
            let v = if max_r > 0.0 { r.firing_rate / max_r } else { 0.0 };
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
                let c = grad.at(v as f64).to_rgba8();
                egui::Color32::from_rgb(c[0], c[1], c[2])
            })
            .collect()
    }

    /// Draw 3D-style stacked layers: each layer as a horizontal band of "cells" (dots) colored by activation.
    fn draw_3d_stack(
        ui: &mut egui::Ui,
        retina: &Retina,
        stimulus: &crate::stimulus::ColorStimulus,
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
            let sx = (ix * stimulus.width / N).min(stimulus.width - 1);
            let sy = (iy * stimulus.height / N).min(stimulus.height - 1);
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

        let labels = ["Stimulus", "Cones L", "Cones M", "Cones S", "Horizontal", "Bp ON", "Bp OFF", "Amacrine", "RGC"];
        let cone_n = (retina.cones.len() as f32).sqrt() as usize;

        for (layer, &label) in labels.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.set_width(label_w);
                ui.label(egui::RichText::new(label).small());
                let grid_rect = ui.allocate_rect(
                    egui::Rect::from_min_size(ui.cursor().min, egui::Vec2::new(grid_w, grid_h)),
                    egui::Sense::hover(),
                ).rect;
                let painter = ui.painter();
                for iy in 0..N {
                    for ix in 0..N {
                        let cx = grid_rect.min.x + (ix as f32 + 0.5) * cell_w;
                        let cy = grid_rect.min.y + (iy as f32 + 0.5) * cell_h;
                        let v = match layer {
                            0 => stim_lum_at(ix.min(stimulus.width), iy.min(stimulus.height)),
                            1 => retina.cones.get(iy * cone_n + ix).map(|c| if c.cone_type == ConeType::L { c.effective_activation } else { 0.0 }).unwrap_or(0.0),
                            2 => retina.cones.get(iy * cone_n + ix).map(|c| if c.cone_type == ConeType::M { c.effective_activation } else { 0.0 }).unwrap_or(0.0),
                            3 => retina.cones.get(iy * cone_n + ix).map(|c| if c.cone_type == ConeType::S { c.effective_activation } else { 0.0 }).unwrap_or(0.0),
                            4 => retina.horizontals.get(iy * cone_n + ix).map(|h| h.activation).unwrap_or(0.0),
                            5 => retina.bipolars.get((iy.min(15) * 16 + ix.min(15)) * 4 + 0).map(|b| b.effective_activation).unwrap_or(0.0),
                            6 => retina.bipolars.get((iy.min(15) * 16 + ix.min(15)) * 4 + 1).map(|b| b.effective_activation).unwrap_or(0.0),
                            7 => retina.amacrines.get(iy * cone_n + ix).map(|a| a.activation).unwrap_or(0.0),
                            8 => {
                                let n_xy = (retina.rgcs.len() / 6) as f32;
                                let n_xy = n_xy.sqrt() as usize;
                                if n_xy == 0 { 0.0 } else {
                                    let base = (iy.min(n_xy - 1) * n_xy + ix.min(n_xy - 1)) * 6;
                                    let sum: f32 = (0..6).filter_map(|k| retina.rgcs.get(base + k)).map(|r| r.firing_rate).sum();
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
                        painter.circle_filled(egui::Pos2::new(cx, cy), rad, egui::Color32::from_rgb(r, g, b));
                    }
                }
            });
        }
    }
}

impl eframe::App for CircuitApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let time_varying = self.stimulus_type.is_time_varying();
        let dt_s = self.last_frame.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.016);
        self.last_frame = Some(std::time::Instant::now());

        // Always run simulation every frame so Firing & Statistics panel shows live stimulus response
        self.advance_time(dt_s);
        self.run_simulation(dt_s);

        // Rebuild viewer image when dirty or time-varying (throttle texture rebuild for performance)
        if self.dirty || time_varying {
            self.build_viewer_image();
        }

        if time_varying {
            ctx.request_repaint_after(std::time::Duration::from_millis(REPAINT_INTERVAL_MS));
        } else {
            // Request repaint periodically so stats stay live when user changes stimulus
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        // ─── Firing & Statistics window (separate) ───
        egui::Window::new("Firing & Statistics")
            .default_pos([400.0, 20.0])
            .default_size([280.0, 400.0])
            .resizable(true)
            .scroll2([false, true])
            .show(ctx, |ui| {
                show_firing_panel(ui, &self.retina, self.selected_rgc_index);
            });

        // ─── Left panel (scrollable) ───
        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(layout::CONTROLS_PANEL_WIDTH)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    controls::show(ui, self);
                });
            });

        // ─── Draw canvas for DrawnShape ───
        if matches!(&self.stimulus_type, StimulusType::DrawnShape { .. }) {
            egui::Window::new("Draw stimulus")
                .default_width(280.0)
                .show(ctx, |ui| {
                    let sz = 180.0;
                    let (rect, response) = ui.allocate_exact_size(egui::Vec2::new(sz, sz), egui::Sense::click_and_drag());
                    let mut mask = match &mut self.stimulus_type {
                        StimulusType::DrawnShape { mask, .. } => mask.clone(),
                        _ => vec![0.0; DRAWN_MASK_SIZE * DRAWN_MASK_SIZE],
                    };
                    if let Some(pos) = response.interact_pointer_pos() {
                        if response.dragged() || response.clicked() {
                            let x = ((pos.x - rect.min.x) / rect.width() * DRAWN_MASK_SIZE as f32)
                                .clamp(0.0, (DRAWN_MASK_SIZE - 1) as f32) as usize;
                            let y = ((pos.y - rect.min.y) / rect.height() * DRAWN_MASK_SIZE as f32)
                                .clamp(0.0, (DRAWN_MASK_SIZE - 1) as f32) as usize;
                            let rad = 3;
                            for dy in -(rad as i32)..=rad as i32 {
                                for dx in -(rad as i32)..=rad as i32 {
                                    let ix = (x as i32 + dx).clamp(0, (DRAWN_MASK_SIZE - 1) as i32) as usize;
                                    let iy = (y as i32 + dy).clamp(0, (DRAWN_MASK_SIZE - 1) as i32) as usize;
                                    if ix < DRAWN_MASK_SIZE && iy < DRAWN_MASK_SIZE {
                                        let idx = iy * DRAWN_MASK_SIZE + ix;
                                        if idx < mask.len() {
                                            mask[idx] = (mask[idx] + self.drawn_mask_brush).min(1.0);
                                        }
                                    }
                                }
                            }
                            self.dirty = true;
                        }
                    }
                    if let StimulusType::DrawnShape { mask: m, .. } = &mut self.stimulus_type {
                        *m = mask;
                    }
                    // Draw preview in the same rect so the image is what you click
                    let preview_mask = match &self.stimulus_type {
                        StimulusType::DrawnShape { mask, .. } => mask,
                        _ => &self.drawn_mask,
                    };
                    let preview: Vec<egui::Color32> = preview_mask
                        .iter()
                        .map(|&v| {
                            let b = (v * 255.0) as u8;
                            egui::Color32::from_rgb(b, b, b)
                        })
                        .collect();
                    let img = egui::ColorImage {
                        size: [DRAWN_MASK_SIZE, DRAWN_MASK_SIZE],
                        pixels: preview,
                    };
                    let tex = ui.ctx().load_texture("draw_preview", img, egui::TextureOptions::default());
                    ui.put(rect, egui::Image::new((tex.id(), egui::Vec2::new(sz, sz))));
                    if ui.button("Clear").clicked() {
                        if let StimulusType::DrawnShape { mask: m, .. } = &mut self.stimulus_type {
                            m.fill(0.0);
                            self.dirty = true;
                        }
                    }
                });
        }

        // ─── Top bar ───
        egui::TopBottomPanel::top("top_bar")
            .exact_height(28.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("RGC Circuit Simulator").size(14.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(self.view_layer.label());
                    });
                });
            });

        // ─── Central viewer ───
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(18, 18, 18)))
            .show(ctx, |ui| {
                let available = ui.available_size();

                if self.view_layer == ViewLayer::AllLayers && !self.layer_stack_images.is_empty() {
                    let labels = ["Stimulus", "Cones L", "Cones M", "Cones S", "Horizontal", "Bipolar ON", "Bipolar OFF", "Amacrine", "RGC", "RGC heat"];
                    let thumb_sz = LAYER_THUMB_SIZE as f32;
                    let cols = 5;
                    let opts = egui::TextureOptions::default();
                    egui::ScrollArea::both().show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            for (row, chunk) in self.layer_stack_images.chunks(cols).enumerate() {
                                ui.horizontal(|ui| {
                                    for (col, img) in chunk.iter().enumerate() {
                                        let label = labels.get(row * cols + col).copied().unwrap_or("");
                                        ui.vertical(|ui| {
                                            ui.label(label);
                                            let id = format!("layer_thumb_{}_{}", row, col);
                                            let tex = ui.ctx().load_texture(id, img.clone(), opts);
                                            ui.image((tex.id(), egui::Vec2::new(thumb_sz, thumb_sz)));
                                        });
                                    }
                                });
                            }
                        });
                    });
                } else if self.view_layer == ViewLayer::Stack3D {
                    let (sw, sh) = self.stimulus_size;
                    let stimulus = render_stimulus(&self.stimulus_type, sw, sh, self.sim_time_s);
                    draw_3d_stack(ui, &self.retina, &stimulus, available);
                } else {
                    let tex = ui.ctx().load_texture(
                        "main_viewer",
                        self.viewer_image.clone(),
                        egui::TextureOptions::default(),
                    );
                    let vsz = self.view_size() as f32;
                    let size = (available.x.min(available.y)).min(vsz);
                    let size = egui::Vec2::new(size, size);
                    ui.allocate_ui_with_layout(
                        egui::Vec2::new(available.x, available.y),
                        egui::Layout::centered_and_justified(egui::Direction::TopDown),
                        |ui| {
                            ui.image((tex.id(), size));
                        },
                    );
                }

                // Firing rate time series (small plot below viewer when RGC view)
                if self.view_layer == ViewLayer::RGC || self.view_layer == ViewLayer::RGCHeatmap {
                    ui.add_space(8.0);
                    let history = self.firing_history.clone();
                    if history.len() >= 2 {
                        let plot = egui_plot::Plot::new("firing_ts")
                            .height(80.0)
                            .label_formatter(|_name, value| format!("{:.0} sp/s", value.y));
                        let points: Vec<[f64; 2]> = history
                            .iter()
                            .enumerate()
                            .map(|(i, &y)| [i as f64 / history.len() as f64, y as f64])
                            .collect();
                        plot.show(ui, |plot_ui| {
                            plot_ui.line(egui_plot::Line::new(egui_plot::PlotPoints::new(points)).color(egui::Color32::from_rgb(100, 200, 255)));
                        });
                    }
                }
            });
    }
}
