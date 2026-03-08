//! Layer 4: Retinal ganglion cells. Integrate from bipolars, LN to firing rate.
//! Field & Chichilnisky (2007); Dacey & Packer (2003) for color opponency.

use super::bipolar::{BipolarCell, BipolarType};
use super::horizontal::gaussian_weight;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RGCType {
    MidgetON_L,
    MidgetOFF_L,
    MidgetON_M,
    MidgetOFF_M,
    SmallBistratifiedON,
    ParasolON,
    ParasolOFF,
    ipRGC,
}

#[derive(Clone, Debug)]
pub struct RGC {
    pub x: f32,
    pub y: f32,
    pub rgc_type: RGCType,
    pub dendritic_field_radius_um: f32,
    pub activation: f32,
    pub firing_rate: f32,
}

impl RGC {
    pub fn new(x: f32, y: f32, rgc_type: RGCType, dendritic_field_radius_um: f32) -> Self {
        RGC {
            x,
            y,
            rgc_type,
            dendritic_field_radius_um,
            activation: 0.0,
            firing_rate: 0.0,
        }
    }

    pub fn receives_from(&self, b: &BipolarCell, scale_um_per_unit: f32) -> bool {
        let dx = (self.x - b.x) * scale_um_per_unit;
        let dy = (self.y - b.y) * scale_um_per_unit;
        let d = (dx * dx + dy * dy).sqrt();
        if d > self.dendritic_field_radius_um * 2.5 {
            return false;
        }
        match self.rgc_type {
            RGCType::MidgetON_L => b.bp_type == BipolarType::MidgetON,
            RGCType::MidgetOFF_L => b.bp_type == BipolarType::MidgetOFF,
            RGCType::MidgetON_M => b.bp_type == BipolarType::MidgetON,
            RGCType::MidgetOFF_M => b.bp_type == BipolarType::MidgetOFF,
            RGCType::SmallBistratifiedON => b.bp_type == BipolarType::SConeBipolarON,
            RGCType::ParasolON => b.bp_type == BipolarType::DiffuseON,
            RGCType::ParasolOFF => b.bp_type == BipolarType::DiffuseOFF,
            RGCType::ipRGC => b.bp_type == BipolarType::GiantON,
        }
    }
}

pub fn dendritic_weight_rgc(d_um: f32, radius_um: f32) -> f32 {
    gaussian_weight(d_um, radius_um)
}

pub fn rgc_linear_response(rgc: &RGC, bipolars: &[BipolarCell], scale_um_per_unit: f32) -> f32 {
    bipolars
        .iter()
        .filter(|b| rgc.receives_from(b, scale_um_per_unit))
        .map(|b| {
            let dx = (rgc.x - b.x) * scale_um_per_unit;
            let dy = (rgc.y - b.y) * scale_um_per_unit;
            let d = (dx * dx + dy * dy).sqrt();
            let w = dendritic_weight_rgc(d, rgc.dendritic_field_radius_um);
            b.effective_activation * w
        })
        .sum()
}

/// LN model: firing_rate = r_max / (1 + exp(-slope * (generator - x_half)))
/// Chichilnisky 2001; r_max 50–300 sp/s typical (Field & Chichilnisky 2007)
pub fn rgc_firing_rate(linear: f32, r_max: f32, x_half: f32, slope: f32) -> f32 {
    r_max / (1.0 + (-slope * (linear - x_half)).exp())
}
