//! Layer 2a: Horizontal cells. Lateral pooling from cones, sign-inverting feedback.
//! H-cell feedback creates RF surround (Kamermans & Spekreijse 1999).

use super::cones::Cone;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HCellType {
    HI,
    HII,
}

#[derive(Clone, Debug)]
pub struct HorizontalCell {
    pub x: f32,
    pub y: f32,
    pub h_type: HCellType,
    pub activation: f32,
    pub sigma: f32,
}

impl HorizontalCell {
    pub fn new(x: f32, y: f32, h_type: HCellType, sigma: f32) -> Self {
        HorizontalCell {
            x,
            y,
            h_type,
            activation: 0.0,
            sigma,
        }
    }
}

pub fn gaussian_weight(d: f32, sigma: f32) -> f32 {
    if sigma <= 0.0 {
        return if d <= 1e-6 { 1.0 } else { 0.0 };
    }
    (-d * d / (2.0 * sigma * sigma)).exp()
}

pub fn pool_from_cones(h: &HorizontalCell, cones: &[Cone], feedback_cone_activation: bool) -> f32 {
    let a = if feedback_cone_activation {
        |c: &Cone| c.effective_activation
    } else {
        |c: &Cone| c.activation
    };
    let mut sum = 0.0;
    let mut w_sum = 0.0;
    for c in cones {
        let d = ((h.x - c.x).powi(2) + (h.y - c.y).powi(2)).sqrt();
        let w = gaussian_weight(d, h.sigma);
        sum += a(c) * w;
        w_sum += w;
    }
    if w_sum > 1e-9 {
        sum / w_sum
    } else {
        0.0
    }
}
