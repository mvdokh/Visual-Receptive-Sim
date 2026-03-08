//! Stimulus: spectral light, shapes, and temporal modulation.

pub mod color;

pub use color::{
    ColorStimulus, N_WAVELENGTH_BANDS, StimulusType, WAVELENGTH_MIN_NM, WAVELENGTH_STEP_NM,
    render_stimulus, rgb_to_lms,
};
