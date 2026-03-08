//! GUI state: view layer, resolution, constants.

/// Which layer to show in the main viewer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ViewLayer {
    #[default]
    Stimulus,
    ConesL,
    ConesM,
    ConesS,
    Horizontal,
    BipolarOn,
    BipolarOff,
    Amacrine,
    RGC,
    RGCHeatmap,
    AllLayers,
    /// 3D-style: stacked layers with cell dots
    Stack3D,
}

impl ViewLayer {
    pub const ALL: [ViewLayer; 12] = [
        ViewLayer::Stimulus,
        ViewLayer::ConesL,
        ViewLayer::ConesM,
        ViewLayer::ConesS,
        ViewLayer::Horizontal,
        ViewLayer::BipolarOn,
        ViewLayer::BipolarOff,
        ViewLayer::Amacrine,
        ViewLayer::RGC,
        ViewLayer::RGCHeatmap,
        ViewLayer::AllLayers,
        ViewLayer::Stack3D,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ViewLayer::Stimulus => "Stimulus",
            ViewLayer::ConesL => "Cones L",
            ViewLayer::ConesM => "Cones M",
            ViewLayer::ConesS => "Cones S",
            ViewLayer::Horizontal => "Horizontal",
            ViewLayer::BipolarOn => "Bipolar ON",
            ViewLayer::BipolarOff => "Bipolar OFF",
            ViewLayer::Amacrine => "Amacrine",
            ViewLayer::RGC => "RGC (gray)",
            ViewLayer::RGCHeatmap => "RGC Firing",
            ViewLayer::AllLayers => "All layers",
            ViewLayer::Stack3D => "3D stack",
        }
    }
}

pub const VIEW_SIZE_HIGH: usize = 512;
pub const VIEW_SIZE_MED: usize = 384;
pub const VIEW_SIZE_LOW: usize = 256;
pub const LAYER_THUMB_SIZE: usize = 128;
pub const DRAWN_MASK_SIZE: usize = 64;
pub const REPAINT_INTERVAL_MS: u64 = 33; // ~30 fps when time-varying
