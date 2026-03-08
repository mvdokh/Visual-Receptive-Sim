//! RGC Circuit Simulator: receptive fields emerge from layered retinal circuit.
//! Layers: Photoreceptors -> Horizontal + Bipolar -> Amacrine -> RGC.

pub mod circuit;
pub mod compute;
pub mod stimulus;
pub mod viz;
pub mod gui;

pub use circuit::{simulate_frame, Retina};
pub use stimulus::{ColorStimulus, StimulusType, render_stimulus};
pub use gui::CircuitApp;
