//! Qwen3-ASR multilingual transcription example
//!
//! This example demonstrates using Qwen3-ASR for multilingual speech-to-text,
//! with excellent support for Japanese, Chinese, Vietnamese, Korean, and 30+ other languages.
//!
//! ## Usage
//!
//! ```bash
//! cargo build --example qwen3_transcribe
//! ./target/debug/examples/qwen3_transcribe <audio_file.wav> [language]
//! ```
//!
//! ## Examples
//!
//! ```bash
//! # Japanese transcription
//! ./target/debug/examples/qwen3_transcribe japanese_audio.wav ja
//!
//! # Chinese (Mandarin) transcription
//! ./target/debug/examples/qwen3_transcribe chinese_audio.wav zh
//!
//! # Automatic language detection
//! ./target/debug/examples/qwen3_transcribe multilingual_audio.wav
//! ```

use fluidaudio_rs::FluidAudio;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio_file> [language]", args[0]);
        eprintln!();
        eprintln!("Language codes (optional):");
        eprintln!("  ja, Japanese    - Japanese");
        eprintln!("  zh, Chinese     - Chinese (Mandarin)");
        eprintln!("  yue, Cantonese  - Cantonese");
        eprintln!("  ko, Korean      - Korean");
        eprintln!("  vi, Vietnamese  - Vietnamese");
        eprintln!("  th, Thai        - Thai");
        eprintln!("  id, Indonesian  - Indonesian");
        eprintln!("  ms, Malay       - Malay");
        eprintln!("  tl, Filipino    - Filipino");
        eprintln!("  en, English     - English");
        eprintln!("  ... and 20+ more European languages");
        eprintln!();
        eprintln!("If no language is specified, automatic detection will be used.");
        std::process::exit(1);
    }

    let audio_file = &args[1];
    let language = args.get(2).map(|s| s.as_str());

    println!("FluidAudio Qwen3-ASR Multilingual Transcription");
    println!("==============================================\n");

    // Create FluidAudio instance
    let audio = FluidAudio::new()?;

    // Get system info
    let info = audio.system_info();
    println!("System: {} ({})", info.chip_name, info.platform);
    println!("Apple Silicon: {}", info.is_apple_silicon);
    println!();

    if !info.is_apple_silicon {
        eprintln!("Warning: Qwen3-ASR works best on Apple Silicon. Intel Macs may have limited support.");
    }

    // Initialize Qwen3-ASR (downloads models on first run)
    println!("Initializing Qwen3-ASR...");
    println!("(First run may take 20-30s to download and compile models)");
    audio.init_qwen3_asr()?;
    println!("✓ Qwen3-ASR initialized\n");

    // Transcribe the audio file
    println!("Transcribing: {}", audio_file);
    if let Some(lang) = language {
        println!("Language hint: {}", lang);
    } else {
        println!("Language: auto-detect");
    }
    println!();

    let result = audio.qwen3_transcribe_file(audio_file, language)?;

    // Display results
    println!("Results:");
    println!("--------");
    println!("Text: {}", result.text);
    println!();
    println!("Duration: {:.2}s", result.duration);
    println!("Processing time: {:.2}s", result.processing_time);
    println!("Real-time factor: {:.1}x", result.rtfx);
    println!(
        "({}x means processing was {} faster than real-time)",
        result.rtfx,
        if result.rtfx > 1.0 {
            format!("{:.1}", result.rtfx)
        } else {
            format!("{:.1}", 1.0 / result.rtfx)
        }
    );

    Ok(())
}
