//! Layer 1: Photoreceptors (L, M, S cones).
//! Spectral sensitivity: Stockman & Sharpe (2000).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConeType {
    L,
    M,
    S,
}

#[derive(Clone, Debug)]
pub struct Cone {
    pub x: f32,
    pub y: f32,
    pub cone_type: ConeType,
    pub activation: f32,
    pub effective_activation: f32,
}

impl Cone {
    pub fn new(x: f32, y: f32, cone_type: ConeType) -> Self {
        Cone {
            x,
            y,
            cone_type,
            activation: 0.0,
            effective_activation: 0.0,
        }
    }
}

/// Spectral sensitivity (Stockman & Sharpe 2000). Gaussian approximation.
/// L: peak=559nm, σ=33.5nm; M: peak=530nm, σ=32.5nm; S: peak=421nm, σ=21.0nm.
pub fn spectral_sensitivity(cone_type: ConeType, wavelength_nm: f32) -> f32 {
    let (peak, sigma) = match cone_type {
        ConeType::L => (559.0, 33.5),
        ConeType::M => (530.0, 32.5),
        ConeType::S => (421.0, 21.0),
    };
    let d = wavelength_nm - peak;
    (-d * d / (2.0 * sigma * sigma)).exp()
}
