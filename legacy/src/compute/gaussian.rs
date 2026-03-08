//! Gaussian spatial kernels. Precomputed at startup; recompute only when sigma/radius changes.

/// Precomputed 1D Gaussian kernel. Use for spatial pooling (H-cells, amacrine).
/// Recompute only when slider changes, not every frame.
#[derive(Clone, Debug)]
pub struct GaussianKernel {
    pub weights: Vec<f32>,
    pub radius: usize,
    pub sigma: f32,
}

impl GaussianKernel {
    /// Build kernel with radius (half-width in samples) and sigma.
    /// Weights sum to ~1 for normalized convolution.
    pub fn new(radius: usize, sigma: f32) -> Self {
        let n = radius * 2 + 1;
        let mut weights = Vec::with_capacity(n * n);
        let mut sum = 0.0;
        for iy in 0..n {
            for ix in 0..n {
                let dx = ix as f32 - radius as f32;
                let dy = iy as f32 - radius as f32;
                let d2 = dx * dx + dy * dy;
                let w = if sigma > 0.0 {
                    (-d2 / (2.0 * sigma * sigma)).exp()
                } else if d2 <= 1e-12 {
                    1.0
                } else {
                    0.0
                };
                sum += w;
                weights.push(w);
            }
        }
        if sum > 1e-9 {
            for w in &mut weights {
                *w /= sum;
            }
        }
        GaussianKernel {
            weights,
            radius,
            sigma,
        }
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }
}

/// Point evaluation: gaussian_weight(d, sigma) = exp(-d²/(2σ²))
#[inline]
pub fn gaussian_weight(d: f32, sigma: f32) -> f32 {
    if sigma <= 0.0 {
        return if d <= 1e-6 { 1.0 } else { 0.0 };
    }
    (-d * d / (2.0 * sigma * sigma)).exp()
}
