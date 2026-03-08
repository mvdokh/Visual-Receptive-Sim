//! Visualization: heatmaps, layer stack, 3D volume, firing panel, RF trace.

pub mod cell_trace;
pub mod firing_panel;
pub mod heatmap;
pub mod layer_stack;
pub mod volume;

pub use cell_trace::{CellTraceState, show as show_cell_trace};
pub use firing_panel::show as show_firing_panel;
pub use heatmap::{layer_to_image, Colormap, layer_cones, layer_horizontal, layer_bipolar, layer_amacrine, layer_rgc, layer_rgc_heatmap, stimulus_rgb};
pub use layer_stack::build_layer_stack;
pub use volume::draw_3d_stack;
