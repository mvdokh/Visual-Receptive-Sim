//! All-layers thumbnail grid.

use eframe::egui;

use crate::circuit::{ConeType, Retina};
use crate::stimulus::ColorStimulus;
use super::heatmap::{
    layer_amacrine, layer_bipolar, layer_cones, layer_horizontal, layer_rgc, layer_rgc_heatmap,
    stimulus_rgb,
};

const THUMB_SIZE: usize = 128;

/// Build thumbnail images for all layers.
pub fn build_layer_stack(
    retina: &Retina,
    stimulus: &ColorStimulus,
) -> Vec<(String, egui::ColorImage)> {
    let thumb = THUMB_SIZE;
    vec![
        (
            "Stimulus".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: stimulus_rgb(stimulus, thumb, thumb),
            },
        ),
        (
            "Cones L".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: layer_cones(retina, thumb, thumb, ConeType::L),
            },
        ),
        (
            "Cones M".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: layer_cones(retina, thumb, thumb, ConeType::M),
            },
        ),
        (
            "Cones S".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: layer_cones(retina, thumb, thumb, ConeType::S),
            },
        ),
        (
            "Horizontal".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: layer_horizontal(retina, thumb, thumb),
            },
        ),
        (
            "Bipolar ON".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: layer_bipolar(retina, thumb, thumb, true),
            },
        ),
        (
            "Bipolar OFF".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: layer_bipolar(retina, thumb, thumb, false),
            },
        ),
        (
            "Amacrine".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: layer_amacrine(retina, thumb, thumb),
            },
        ),
        (
            "RGC".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: layer_rgc(retina, thumb, thumb),
            },
        ),
        (
            "RGC heat".into(),
            egui::ColorImage {
                size: [thumb, thumb],
                pixels: layer_rgc_heatmap(retina, thumb, thumb),
            },
        ),
    ]
}
