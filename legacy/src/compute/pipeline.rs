//! Simulation pipeline: one tick of the retinal circuit.
//! Runs cones → horizontal → H feedback → bipolar → amacrine → RGC.

use crate::circuit::retina::simulate_frame;
use crate::circuit::Retina;
use crate::stimulus::ColorStimulus;

/// Run one simulation frame. Called from GUI or background thread.
pub fn tick(retina: &mut Retina, stimulus: &ColorStimulus, dt_s: f32) {
    simulate_frame(retina, stimulus, dt_s);
}
