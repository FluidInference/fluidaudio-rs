//! Qwen3-ASR streaming transcription example
//!
//! This example demonstrates real-time streaming transcription using Qwen3-ASR.
//! Perfect for meeting transcription, live captions, or real-time translation applications.
//!
//! ## Usage
//!
//! ```bash
//! cargo build --example qwen3_streaming
//! ./target/debug/examples/qwen3_streaming <audio_file.wav> [language]
//! ```
//!
//! This simulates real-time streaming by reading an audio file in chunks,
//! similar to how microphone input would work.

use fluidaudio_rs::FluidAudio;
use std::env;
use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio_file> [language]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} japanese_meeting.wav ja", args[0]);
        eprintln!("  {} chinese_audio.wav zh", args[0]);
        eprintln!("  {} multilingual.wav    # auto-detect", args[0]);
        std::process::exit(1);
    }

    let audio_file = &args[1];
    let language = args.get(2).map(|s| s.as_str());

    println!("FluidAudio Qwen3-ASR Streaming Transcription");
    println!("============================================\n");

    // Create FluidAudio instance
    let audio = FluidAudio::new()?;

    // Initialize Qwen3 streaming
    println!("Initializing Qwen3 Streaming...");
    audio.init_qwen3_streaming()?;
    println!("✓ Initialized\n");

    // Start streaming session
    // Parameters: language, min_audio_secs, chunk_secs, max_audio_secs
    println!("Starting streaming session...");
    if let Some(lang) = language {
        println!("Language: {}", lang);
    } else {
        println!("Language: auto-detect");
    }
    audio.qwen3_streaming_start(
        language,
        1.0,  // Start transcribing after 1 second of audio
        2.0,  // Update every 2 seconds
        30.0, // Keep max 30 seconds in buffer
    )?;
    println!("✓ Session started\n");

    // Load audio file (in a real app, this would come from microphone)
    println!("Streaming audio from: {}", audio_file);
    println!("(Simulating real-time chunks)\n");

    // For this example, we'll load the entire file and simulate streaming
    // In a real app, you'd capture from microphone in real-time
    let mut file = File::open(audio_file)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // TODO: Parse WAV format properly - for now assume raw PCM
    // Skip WAV header (44 bytes) if it's a WAV file
    let audio_data = if buffer.starts_with(b"RIFF") && buffer.len() > 44 {
        &buffer[44..]
    } else {
        &buffer[..]
    };

    // Convert bytes to f32 samples (assuming 16-bit PCM)
    let mut samples = Vec::new();
    for chunk in audio_data.chunks(2) {
        if chunk.len() == 2 {
            let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
            let sample_f32 = sample_i16 as f32 / 32768.0;
            samples.push(sample_f32);
        }
    }

    // Feed audio in chunks (simulating real-time streaming)
    let chunk_size = 16000; // 1 second chunks at 16kHz
    let mut partial_count = 0;

    println!("--- Streaming Results ---\n");

    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        // In a real app, this would be: let chunk = capture_from_microphone();
        print!("Feeding chunk {} ({:.1}s)... ", i + 1, chunk.len() as f32 / 16000.0);

        match audio.qwen3_streaming_feed(chunk) {
            Ok(Some(partial_text)) => {
                partial_count += 1;
                println!("✓");
                println!("Partial #{}: {}", partial_count, partial_text);
                println!();
            }
            Ok(None) => {
                println!("(buffering)");
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }

        // Simulate real-time delay (remove in actual real-time app)
        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    println!("\n--- Final Result ---\n");

    // Get final transcription
    let final_text = audio.qwen3_streaming_finish()?;
    println!("Complete transcription:");
    println!("{}", final_text);

    println!("\n✓ Streaming session completed");
    println!(
        "Total partial updates received: {}",
        partial_count
    );

    Ok(())
}
