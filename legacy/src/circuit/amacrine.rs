//! Layer 3: Amacrine cells. Lateral inhibition in inner plexiform layer.
//! Masland (2012): AII (narrow), wide-field GABAergic.

use super::bipolar::BipolarCell;
use super::horizontal::gaussian_weight;

#[derive(Clone, Debug)]
pub struct AmacrineCell {
    pub x: f32,
    pub y: f32,
    pub sigma_um: f32,
    pub activation: f32,
}

impl AmacrineCell {
    pub fn new(x: f32, y: f32, sigma_um: f32) -> Self {
        AmacrineCell {
            x,
            y,
            sigma_um,
            activation: 0.0,
        }
    }
}

pub fn pool_from_bipolars(
    a: &AmacrineCell,
    bipolars: &[BipolarCell],
    scale_um_per_unit: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut w_sum = 0.0;
    for b in bipolars {
        let dx = (a.x - b.x) * scale_um_per_unit;
        let dy = (a.y - b.y) * scale_um_per_unit;
        let d = (dx * dx + dy * dy).sqrt();
        let w = gaussian_weight(d, a.sigma_um);
        sum += b.activation * w;
        w_sum += w;
    }
    if w_sum > 1e-9 {
        sum / w_sum
    } else {
        0.0
    }
}
