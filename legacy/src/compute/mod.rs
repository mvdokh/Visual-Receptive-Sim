//! Compute: pure math, no rendering. Spatial kernels, LMS transforms, sigmoid, simulation tick.

pub mod gaussian;
pub mod lms_matrix;
pub mod pipeline;
pub mod sigmoid;

pub use gaussian::GaussianKernel;
pub use lms_matrix::rgb_to_lms;
pub use pipeline::tick;
pub use sigmoid::ln_sigmoid;
