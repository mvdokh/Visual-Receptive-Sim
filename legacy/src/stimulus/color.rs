//! Color stimulus: spectral representation, time-varying and drawn stimuli.
//! Cone spectra: Stockman & Sharpe (2000); LMS: Hunt-Pointer-Estevez (1982)

use ndarray::Array3;
use crate::circuit::cones::ConeType;

pub const WAVELENGTH_MIN_NM: f32 = 380.0;
pub const WAVELENGTH_MAX_NM: f32 = 700.0;
pub const WAVELENGTH_STEP_NM: f32 = 5.0;
pub const N_WAVELENGTH_BANDS: usize = ((700.0 - 380.0) / 5.0) as usize + 1;

pub use crate::circuit::cones::spectral_sensitivity;

#[derive(Clone)]
pub struct ColorStimulus {
    pub width: usize,
    pub height: usize,
    pub spectra: Array3<f32>,
}

#[rustfmt::skip]
pub fn rgb_to_lms(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (0.3897 * r + 0.6890 * g - 0.0787 * b,
     -0.2298 * r + 1.1834 * g + 0.0464 * b,
     b)
}

/// Stimulus type. Time-varying ones use time_s in render_stimulus.
#[derive(Clone, Debug)]
pub enum StimulusType {
    MonochromaticSpot { x: f32, y: f32, r: f32, wavelength_nm: f32 },
    WhiteLight { x: f32, y: f32, r: f32 },
    BlinkingSpot { x: f32, y: f32, r: f32, wavelength_nm: f32, period_s: f32, duty: f32 },
    PulsingSpot { x: f32, y: f32, r: f32, wavelength_nm: f32, period_s: f32 },
    MovingSpot { x0: f32, y0: f32, x1: f32, y1: f32, r: f32, wavelength_nm: f32, period_s: f32 },
    DrawnShape { mask: Vec<f32>, width: usize, height: usize, wavelength_nm: f32 },
    ColoredGrating { sf_cpd: f32, orientation_deg: f32, wavelength_nm: f32 },
    RedGreenGrating,
    BlueYellowGrating,
    ConeIsolating { target: ConeType },
    NaturalImage { path: String },
    UniformLMS { l: f32, m: f32, s: f32 },
    FullFieldWavelength { wavelength_nm: f32 },
    DualSpot { x1: f32, y1: f32, r1: f32, wavelength_nm1: f32, x2: f32, y2: f32, r2: f32, wavelength_nm2: f32 },
    MultiSpot { spots: [(f32, f32, f32, f32); 4] },
}

impl StimulusType {
    pub fn is_time_varying(&self) -> bool {
        matches!(self,
            StimulusType::BlinkingSpot { .. }
            | StimulusType::PulsingSpot { .. }
            | StimulusType::MovingSpot { .. }
        )
    }
}

fn set_monochromatic_at(spectra: &mut Array3<f32>, ix: usize, iy: usize, wavelength_nm: f32, intensity: f32) {
    let band = ((wavelength_nm - WAVELENGTH_MIN_NM) / WAVELENGTH_STEP_NM)
        .round()
        .clamp(0.0, (N_WAVELENGTH_BANDS - 1) as f32) as usize;
    for b in 0..N_WAVELENGTH_BANDS {
        spectra[[iy, ix, b]] = if b == band { intensity } else { 0.0 };
    }
}

fn set_white_at(spectra: &mut Array3<f32>, ix: usize, iy: usize, intensity: f32) {
    for b in 0..N_WAVELENGTH_BANDS {
        spectra[[iy, ix, b]] = intensity;
    }
}

fn lms_to_spectrum_at(spectra: &mut Array3<f32>, ix: usize, iy: usize, l: f32, m: f32, s: f32) {
    use crate::circuit::cones::spectral_sensitivity;
    for b in 0..N_WAVELENGTH_BANDS {
        let lam = WAVELENGTH_MIN_NM + b as f32 * WAVELENGTH_STEP_NM;
        let sl = spectral_sensitivity(ConeType::L, lam);
        let sm = spectral_sensitivity(ConeType::M, lam);
        let ss = spectral_sensitivity(ConeType::S, lam);
        spectra[[iy, ix, b]] = (l * sl + m * sm + s * ss) / 3.0_f32.max(sl + sm + ss);
    }
}

pub fn render_stimulus(stim_type: &StimulusType, width: usize, height: usize, time_s: f32) -> ColorStimulus {
    let mut spectra = Array3::zeros((height, width, N_WAVELENGTH_BANDS));
    match stim_type {
        StimulusType::MonochromaticSpot { x, y, r, wavelength_nm } => {
            draw_spot(&mut spectra, width, height, *x, *y, *r, *wavelength_nm, 1.0, true);
        }
        StimulusType::WhiteLight { x, y, r } => {
            draw_white_spot(&mut spectra, width, height, *x, *y, *r, 1.0);
        }
        StimulusType::BlinkingSpot { x, y, r, wavelength_nm, period_s, duty } => {
            let phase = (time_s / period_s.max(0.001)).fract();
            let intensity = if phase < *duty { 1.0 } else { 0.0 };
            draw_spot(&mut spectra, width, height, *x, *y, *r, *wavelength_nm, intensity, true);
        }
        StimulusType::PulsingSpot { x, y, r, wavelength_nm, period_s } => {
            let phase = (time_s / period_s.max(0.001)) * std::f32::consts::TAU;
            let intensity = 0.5 + 0.5 * phase.sin();
            draw_spot(&mut spectra, width, height, *x, *y, *r, *wavelength_nm, intensity, true);
        }
        StimulusType::MovingSpot { x0, y0, x1, y1, r, wavelength_nm, period_s } => {
            let t = (time_s / period_s.max(0.001)).fract();
            let x = x0 + (x1 - x0) * t;
            let y = y0 + (y1 - y0) * t;
            draw_spot(&mut spectra, width, height, x, y, *r, *wavelength_nm, 1.0, true);
        }
        StimulusType::DrawnShape { mask, width: mw, height: mh, wavelength_nm } => {
            for iy in 0..height {
                for ix in 0..width {
                    let sy = (iy * mh / height).min(mh - 1);
                    let sx = (ix * mw / width).min(mw - 1);
                    let intensity = mask.get(sy * mw + sx).copied().unwrap_or(0.0);
                    set_monochromatic_at(&mut spectra, ix, iy, *wavelength_nm, intensity);
                }
            }
        }
        StimulusType::UniformLMS { l, m, s } => {
            for iy in 0..height {
                for ix in 0..width {
                    lms_to_spectrum_at(&mut spectra, ix, iy, *l, *m, *s);
                }
            }
        }
        StimulusType::FullFieldWavelength { wavelength_nm } => {
            for iy in 0..height {
                for ix in 0..width {
                    set_monochromatic_at(&mut spectra, ix, iy, *wavelength_nm, 1.0);
                }
            }
        }
        StimulusType::DualSpot { x1, y1, r1, wavelength_nm1, x2, y2, r2, wavelength_nm2 } => {
            draw_spot(&mut spectra, width, height, *x1, *y1, *r1, *wavelength_nm1, 1.0, true);
            draw_spot(&mut spectra, width, height, *x2, *y2, *r2, *wavelength_nm2, 1.0, true);
        }
        StimulusType::MultiSpot { spots } => {
            for &(x, y, r, wavelength_nm) in spots.iter() {
                if r > 0.0 {
                    draw_spot(&mut spectra, width, height, x, y, r, wavelength_nm, 1.0, true);
                }
            }
        }
        StimulusType::RedGreenGrating | StimulusType::BlueYellowGrating
        | StimulusType::ColoredGrating { .. } | StimulusType::ConeIsolating { .. }
        | StimulusType::NaturalImage { .. } => {
            for iy in 0..height {
                for ix in 0..width {
                    set_white_at(&mut spectra, ix, iy, 0.5);
                }
            }
        }
    }
    ColorStimulus { width, height, spectra }
}

fn draw_spot(
    spectra: &mut Array3<f32>,
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    r: f32,
    wavelength_nm: f32,
    intensity: f32,
    monochrome: bool,
) {
    let cx = x * width as f32;
    let cy = y * height as f32;
    let r2 = r * r * (width as f32).min(height as f32).powi(2) / 4.0;
    for iy in 0..height {
        for ix in 0..width {
            let dx = ix as f32 - cx;
            let dy = iy as f32 - cy;
            if dx * dx + dy * dy <= r2 {
                if monochrome {
                    set_monochromatic_at(spectra, ix, iy, wavelength_nm, intensity);
                } else {
                    set_white_at(spectra, ix, iy, intensity);
                }
            }
        }
    }
}

fn draw_white_spot(
    spectra: &mut Array3<f32>,
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    r: f32,
    intensity: f32,
) {
    let cx = x * width as f32;
    let cy = y * height as f32;
    let r2 = r * r * (width as f32).min(height as f32).powi(2) / 4.0;
    for iy in 0..height {
        for ix in 0..width {
            let dx = ix as f32 - cx;
            let dy = iy as f32 - cy;
            if dx * dx + dy * dy <= r2 {
                set_white_at(spectra, ix, iy, intensity);
            }
        }
    }
}
