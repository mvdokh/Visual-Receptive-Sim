//! Layer 2b: Bipolar cells. Center from cone pool; ON/OFF types.
//! Euler & Wässle (1995): Midget (1 cone), Diffuse (pools ~7 cones).

use super::cones::Cone;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BipolarType {
    MidgetON,
    DiffuseON,
    SConeBipolarON,
    MidgetOFF,
    DiffuseOFF,
    GiantON,
}

#[derive(Clone, Debug)]
pub struct BipolarCell {
    pub x: f32,
    pub y: f32,
    pub bp_type: BipolarType,
    pub activation: f32,
    pub effective_activation: f32,
    pub n_cones_connected: usize,
    pub dendritic_radius_um: f32,
}

impl BipolarCell {
    pub fn new(x: f32, y: f32, bp_type: BipolarType, dendritic_radius_um: f32) -> Self {
        let n = match bp_type {
            BipolarType::MidgetON | BipolarType::MidgetOFF => 1,
            BipolarType::SConeBipolarON => 1,
            BipolarType::DiffuseON | BipolarType::DiffuseOFF => 7,
            BipolarType::GiantON => 15,
        };
        BipolarCell {
            x,
            y,
            bp_type,
            activation: 0.0,
            effective_activation: 0.0,
            n_cones_connected: n,
            dendritic_radius_um,
        }
    }

    pub fn is_on(&self) -> bool {
        matches!(
            self.bp_type,
            BipolarType::MidgetON
                | BipolarType::DiffuseON
                | BipolarType::SConeBipolarON
                | BipolarType::GiantON
        )
    }
}

fn rectify_positive(x: f32) -> f32 {
    x.max(0.0)
}

fn dist_um(bx: f32, by: f32, cx: f32, cy: f32, scale_um_per_unit: f32) -> f32 {
    let dx = (bx - cx) * scale_um_per_unit;
    let dy = (by - cy) * scale_um_per_unit;
    (dx * dx + dy * dy).sqrt()
}

pub fn dendritic_weight(d_um: f32, radius_um: f32) -> f32 {
    if radius_um <= 0.0 {
        return if d_um <= 1e-6 { 1.0 } else { 0.0 };
    }
    (-d_um * d_um / (2.0 * radius_um * radius_um)).exp()
}

pub fn bipolar_response_from_cones(
    bp: &BipolarCell,
    cones: &[Cone],
    scale_um_per_unit: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut w_sum = 0.0;
    for c in cones {
        let d = dist_um(bp.x, bp.y, c.x, c.y, scale_um_per_unit);
        if d > bp.dendritic_radius_um * 2.5 {
            continue;
        }
        let w = dendritic_weight(d, bp.dendritic_radius_um);
        sum += c.effective_activation * w;
        w_sum += w;
    }
    let pooled = if w_sum > 1e-9 {
        sum / w_sum
    } else {
        0.0
    };
    if bp.is_on() {
        rectify_positive(pooled)
    } else {
        rectify_positive(1.0 - pooled)
    }
}
