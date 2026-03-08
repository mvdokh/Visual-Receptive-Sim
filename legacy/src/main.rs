//! RGC Circuit Simulator — receptive fields emerge from the circuit.

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "RGC Circuit Simulator",
        eframe::NativeOptions::default(),
        Box::new(|cc| Box::new(rgc_circuit_sim::CircuitApp::new(cc))),
    )
}
