//! LN (linear-nonlinear) sigmoid for RGC firing rate.
//! firing_rate = r_max / (1 + exp(-slope * (x - x_half)))
//! Chichilnisky 2001; r_max 50–300 sp/s (Field & Chichilnisky 2007)

/// LN sigmoid: maps generator signal to firing rate (sp/s).
#[inline]
pub fn ln_sigmoid(linear: f32, r_max: f32, x_half: f32, slope: f32) -> f32 {
    r_max / (1.0 + (-slope * (linear - x_half)).exp())
}
