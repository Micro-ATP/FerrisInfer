use std::env;

use ferrisinfer_kernel::{Backend, CpuBackend};
use ferrisinfer_runtime::{ExecutionMode, InferenceEngine, SessionConfig};

fn main() {
    let engine = InferenceEngine::new(CpuBackend::default());
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("plan") => print_plan(&engine, &SessionConfig::default()),
        Some("help") | None => print_help(),
        Some(other) => {
            eprintln!("unknown command: {other}");
            print_help();
        }
    }
}

fn print_help() {
    println!("FerrisInfer CLI");
    println!("Commands:");
    println!("  help    Show this message");
    println!("  plan    Print the current implementation framework");
}

fn print_plan(engine: &InferenceEngine<CpuBackend>, config: &SessionConfig) {
    let prefill = engine.build_plan(ExecutionMode::Prefill);
    let decode = engine.build_plan(ExecutionMode::Decode);

    println!("FerrisInfer implementation framework");
    println!("backend: {}", engine.backend().name());
    println!(
        "default max sequence length: {}",
        config.max_sequence_length
    );
    println!("prefill steps: {}", prefill.steps.len());
    println!("decode steps: {}", decode.steps.len());
    println!("layers: core -> kernel -> model -> io -> runtime -> cli");
}
