//! Retina: layered circuit with optional temporal dynamics for realism.
//! Cone and RGC outputs are low-pass filtered so responses are smooth in time.

use super::amacrine::{pool_from_bipolars, AmacrineCell};
use super::bipolar::{bipolar_response_from_cones, BipolarCell, BipolarType};
use super::cones::{spectral_sensitivity, Cone, ConeType};
use super::horizontal::{gaussian_weight, pool_from_cones, HorizontalCell, HCellType};
use super::rgc::{rgc_firing_rate, rgc_linear_response, RGC, RGCType};
use crate::stimulus::color::{
    ColorStimulus, N_WAVELENGTH_BANDS, WAVELENGTH_MIN_NM, WAVELENGTH_STEP_NM,
};

pub const DEFAULT_UM_PER_UNIT: f32 = 300.0;

/// Time constants (seconds) for exponential smoothing. ~20–50 ms typical for cones/RGC.
pub const DEFAULT_CONE_TAU_S: f32 = 0.03;
pub const DEFAULT_RGC_TAU_S: f32 = 0.04;

#[derive(Clone)]
pub struct Retina {
    pub cones: Vec<Cone>,
    pub horizontals: Vec<HorizontalCell>,
    pub bipolars: Vec<BipolarCell>,
    pub amacrines: Vec<AmacrineCell>,
    pub rgcs: Vec<RGC>,
    pub scale_um_per_unit: f32,
    pub horizontal_feedback_strength: f32,
    pub amacrine_inhibition_strength: f32,
    pub rgc_r_max: f32,
    pub rgc_x_half: f32,
    pub rgc_slope: f32,
    /// Temporal smoothing: cone activation blend factor per dt. 0 = instant.
    pub cone_tau_s: f32,
    pub rgc_tau_s: f32,
    /// Smoothed cone activations (previous frame) for temporal dynamics
    cone_activation_prev: Vec<f32>,
    /// Smoothed RGC firing rates (previous frame)
    rgc_firing_prev: Vec<f32>,
}

impl Retina {
    pub fn new(scale_um_per_unit: f32) -> Self {
        Retina {
            cones: vec![],
            horizontals: vec![],
            bipolars: vec![],
            amacrines: vec![],
            rgcs: vec![],
            scale_um_per_unit,
            horizontal_feedback_strength: 0.5,
            amacrine_inhibition_strength: 0.3,
            rgc_r_max: 100.0,
            rgc_x_half: 0.5,
            rgc_slope: 4.0,
            cone_tau_s: DEFAULT_CONE_TAU_S,
            rgc_tau_s: DEFAULT_RGC_TAU_S,
            cone_activation_prev: vec![],
            rgc_firing_prev: vec![],
        }
    }

    /// Build grid with L:M:S ratio. Uses jittered assignment so all cone types appear across the field.
    pub fn build_demo_grid(n_xy: usize, lms_ratio: (u32, u32, u32)) -> Self {
        use rand::seq::SliceRandom;
        let mut r = Retina::new(DEFAULT_UM_PER_UNIT);
        let total = lms_ratio.0 + lms_ratio.1 + lms_ratio.2;
        let n_cells = n_xy * n_xy;
        let n_l = (n_cells as u32 * lms_ratio.0 / total).max(1) as usize;
        let n_m = ((n_cells as u32 * lms_ratio.1 / total) as usize).min(n_cells.saturating_sub(n_l));
        let n_s = n_cells - n_l - n_m;
        let mut cone_types: Vec<ConeType> = std::iter::repeat(ConeType::L)
            .take(n_l)
            .chain(std::iter::repeat(ConeType::M).take(n_m))
            .chain(std::iter::repeat(ConeType::S).take(n_s))
            .collect();
        let mut rng = rand::thread_rng();
        cone_types.shuffle(&mut rng);
        for iy in 0..n_xy {
            for ix in 0..n_xy {
                let x = (ix as f32 + 0.5) / n_xy as f32;
                let y = (iy as f32 + 0.5) / n_xy as f32;
                let cone_type = cone_types[iy * n_xy + ix];
                r.cones.push(Cone::new(x, y, cone_type));
            }
        }
        r.cone_activation_prev = vec![0.0; r.cones.len()];
        let h_sigma = 0.15;
        for iy in 0..n_xy {
            for ix in 0..n_xy {
                let x = (ix as f32 + 0.5) / n_xy as f32;
                let y = (iy as f32 + 0.5) / n_xy as f32;
                r.horizontals.push(HorizontalCell::new(x, y, HCellType::HI, h_sigma));
            }
        }
        let bp_rad = 0.05;
        for iy in 0..n_xy {
            for ix in 0..n_xy {
                let x = (ix as f32 + 0.5) / n_xy as f32;
                let y = (iy as f32 + 0.5) / n_xy as f32;
                r.bipolars.push(BipolarCell::new(x, y, BipolarType::MidgetON, bp_rad));
                r.bipolars.push(BipolarCell::new(x, y, BipolarType::MidgetOFF, bp_rad));
                r.bipolars.push(BipolarCell::new(x, y, BipolarType::DiffuseON, bp_rad * 1.5));
                r.bipolars.push(BipolarCell::new(x, y, BipolarType::DiffuseOFF, bp_rad * 1.5));
            }
        }
        let a_sigma = 80.0;
        for iy in 0..n_xy {
            for ix in 0..n_xy {
                let x = (ix as f32 + 0.5) / n_xy as f32;
                let y = (iy as f32 + 0.5) / n_xy as f32;
                r.amacrines.push(AmacrineCell::new(x, y, a_sigma));
            }
        }
        let rgc_rad = 0.12;
        let rgc_types = [
            RGCType::MidgetON_L,
            RGCType::MidgetOFF_L,
            RGCType::MidgetON_M,
            RGCType::MidgetOFF_M,
            RGCType::ParasolON,
            RGCType::ParasolOFF,
        ];
        for iy in 0..n_xy {
            for ix in 0..n_xy {
                let x = (ix as f32 + 0.5) / n_xy as f32;
                let y = (iy as f32 + 0.5) / n_xy as f32;
                for &rgc_type in &rgc_types {
                    let rad = match rgc_type {
                        RGCType::ParasolON | RGCType::ParasolOFF => rgc_rad * 1.5,
                        _ => rgc_rad,
                    };
                    r.rgcs.push(RGC::new(x, y, rgc_type, rad));
                }
            }
        }
        r.rgc_firing_prev = vec![0.0; r.rgcs.len()];
        r
    }
}

fn cone_response(cone: &Cone, stimulus: &ColorStimulus) -> f32 {
    let ix = (cone.x * stimulus.width as f32).clamp(0.0, (stimulus.width - 1) as f32) as usize;
    let iy = (cone.y * stimulus.height as f32).clamp(0.0, (stimulus.height - 1) as f32) as usize;
    let mut sum = 0.0;
    for b in 0..N_WAVELENGTH_BANDS {
        let intensity = stimulus.spectra[[iy, ix, b]];
        let sens =
            spectral_sensitivity(cone.cone_type, WAVELENGTH_MIN_NM + b as f32 * WAVELENGTH_STEP_NM);
        sum += intensity * sens;
    }
    sum * WAVELENGTH_STEP_NM / 100.0
}

/// Exponential smoothing: blend factor = 1 - exp(-dt/tau). dt in seconds.
fn blend_factor(dt_s: f32, tau_s: f32) -> f32 {
    if tau_s <= 0.0 {
        return 1.0;
    }
    1.0 - (-(dt_s / tau_s).min(10.0)).exp()
}

pub fn simulate_frame(retina: &mut Retina, stimulus: &ColorStimulus, dt_s: f32) {
    let scale = retina.scale_um_per_unit;
    let cone_alpha = blend_factor(dt_s, retina.cone_tau_s);
    let rgc_alpha = blend_factor(dt_s, retina.rgc_tau_s);

    // 1. Cone responses with temporal smoothing
    for (i, c) in retina.cones.iter_mut().enumerate() {
        let raw = cone_response(c, stimulus);
        let prev = retina.cone_activation_prev.get(i).copied().unwrap_or(0.0);
        c.activation = prev + cone_alpha * (raw - prev);
        c.effective_activation = c.activation;
    }
    for (i, c) in retina.cones.iter().enumerate() {
        if i < retina.cone_activation_prev.len() {
            retina.cone_activation_prev[i] = c.activation;
        }
    }

    // 2. Horizontal
    for h in &mut retina.horizontals {
        h.activation = pool_from_cones(h, &retina.cones, false);
    }

    // 3. H → cone feedback
    for c in &mut retina.cones {
        let mut h_sum = 0.0;
        let mut w_sum = 0.0;
        for h in &retina.horizontals {
            let d = ((c.x - h.x).powi(2) + (c.y - h.y).powi(2)).sqrt() * scale;
            let w = gaussian_weight(d, h.sigma * scale);
            h_sum += h.activation * w;
            w_sum += w;
        }
        let h_val = if w_sum > 1e-9 { h_sum / w_sum } else { 0.0 };
        c.effective_activation =
            (c.activation - retina.horizontal_feedback_strength * h_val).max(0.0);
    }

    // 4. Bipolar
    for bp in &mut retina.bipolars {
        bp.activation = bipolar_response_from_cones(bp, &retina.cones, scale);
        bp.effective_activation = bp.activation;
    }

    // 5. Amacrine
    for a in &mut retina.amacrines {
        a.activation = pool_from_bipolars(a, &retina.bipolars, scale);
    }

    // 6. Amacrine → bipolar
    for bp in &mut retina.bipolars {
        let mut a_sum = 0.0;
        let mut w_sum = 0.0;
        for a in &retina.amacrines {
            let dx = (bp.x - a.x) * scale;
            let dy = (bp.y - a.y) * scale;
            let d = (dx * dx + dy * dy).sqrt();
            let w = gaussian_weight(d, a.sigma_um);
            a_sum += a.activation * w;
            w_sum += w;
        }
        let a_val = if w_sum > 1e-9 { a_sum / w_sum } else { 0.0 };
        bp.effective_activation =
            (bp.activation - retina.amacrine_inhibition_strength * a_val).max(0.0);
    }

    // 7. RGC with temporal smoothing on firing rate
    for (i, rgc) in retina.rgcs.iter_mut().enumerate() {
        let linear = rgc_linear_response(rgc, &retina.bipolars, scale);
        rgc.activation = linear;
        let r_new =
            rgc_firing_rate(linear, retina.rgc_r_max, retina.rgc_x_half, retina.rgc_slope);
        let prev = retina.rgc_firing_prev.get(i).copied().unwrap_or(0.0);
        rgc.firing_rate = prev + rgc_alpha * (r_new - prev);
    }
    for (i, r) in retina.rgcs.iter().enumerate() {
        if i < retina.rgc_firing_prev.len() {
            retina.rgc_firing_prev[i] = r.firing_rate;
        }
    }
}
