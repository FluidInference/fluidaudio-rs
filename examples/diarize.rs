//! Example: Diarize an audio file (identify speakers)
//!
//! Usage: cargo run --example diarize -- path/to/audio.wav [threshold]

use fluidaudio_rs::FluidAudio;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let audio_path = args
        .get(1)
        .ok_or("Usage: diarize <audio_file> [threshold]")?;
    let threshold: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.6);

    println!("FluidAudio Diarization Example");
    println!("==============================\n");

    let audio = FluidAudio::new()?;

    let info = audio.system_info();
    println!("System: {} ({})", info.chip_name, info.platform);
    println!("Apple Silicon: {}\n", audio.is_apple_silicon());

    println!(
        "Initializing diarization (threshold={:.2}, may take a moment on first run)...",
        threshold
    );
    audio.init_diarization(threshold)?;
    println!("Diarization initialized!\n");

    println!("Diarizing: {}", audio_path);
    let segments = audio.diarize_file(audio_path)?;

    println!("\n--- Results ({} segments) ---\n", segments.len());
    for seg in &segments {
        println!(
            "[{:.2}s - {:.2}s] {} (quality: {:.2})",
            seg.start_time, seg.end_time, seg.speaker_id, seg.quality_score
        );
    }

    Ok(())
}
