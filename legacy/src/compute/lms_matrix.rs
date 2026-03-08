//! RGB ↔ LMS color transform. Hunt-Pointer-Estevez, D65 illuminant.
//! Used for stimulus rendering and color opponent signals.

/// Convert RGB (0–1) to LMS using HPE matrix (D65).
/// [L]   [ 0.3897  0.6890 -0.0787] [R]
/// [M] = [-0.2298  1.1834  0.0464] [G]
/// [S]   [ 0.0000  0.0000  1.0000] [B]
#[rustfmt::skip]
pub fn rgb_to_lms(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        0.3897 * r + 0.6890 * g - 0.0787 * b,
        -0.2298 * r + 1.1834 * g + 0.0464 * b,
        b,
    )
}
