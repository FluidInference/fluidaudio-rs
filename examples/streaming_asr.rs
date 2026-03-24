//! Example: Streaming ASR (memory-efficient transcription)
//!
//! This example demonstrates two ways to use Streaming ASR:
//! 1. Session-based API for real-time streaming (start/feed/finish)
//! 2. Simple file wrapper for easy memory-efficient transcription
//!
//! Streaming ASR uses 99.5% less memory than regular ASR, making it ideal
//! for long audio files or resource-constrained environments.
//!
//! Usage: cargo run --example streaming_asr -- path/to/audio.wav [--session]

use fluidaudio_rs::FluidAudio;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let audio_path = args.get(1).ok_or("Usage: streaming_asr <audio_file> [--session]")?;
    let use_session_api = args.get(2).map(|s| s == "--session").unwrap_or(false);

    println!("FluidAudio Streaming ASR Example");
    println!("=================================\n");

    // Create FluidAudio instance
    let audio = FluidAudio::new()?;

    // Print system info
    let info = audio.system_info();
    println!("System: {} ({})", info.chip_name, info.platform);
    println!("Memory: {:.1} GB", info.memory_gb);
    println!("Apple Silicon: {}\n", audio.is_apple_silicon());

    // Initialize Streaming ASR
    println!("Initializing Streaming ASR (99.5% less memory than regular ASR)...");
    audio.init_streaming_asr()?;
    println!("Streaming ASR initialized!\n");

    if use_session_api {
        println!("Using session-based API (start/feed/finish)");
        println!("===========================================\n");
        session_based_transcription(&audio, audio_path)?;
    } else {
        println!("Using simple file wrapper API");
        println!("==============================\n");
        simple_file_transcription(&audio, audio_path)?;
    }

    Ok(())
}

/// Demonstrate the session-based API for real-time streaming
fn session_based_transcription(audio: &FluidAudio, audio_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Note: This example simulates streaming by reading the file in chunks.");
    println!("In a real application, you would feed live audio chunks from a microphone.\n");

    // Start streaming session
    println!("Starting streaming session...");
    audio.streaming_asr_start()?;

    // In a real application, you would feed audio chunks as they arrive
    // For this example, we'll read a file and simulate chunked processing
    println!("Processing audio file: {}", audio_path);
    println!("(In real-time, you would feed chunks as they arrive from microphone)\n");

    // Simulated chunks - in reality these would come from your audio capture
    // For demonstration, we'll show the API usage pattern:

    // let chunk_size = 16000; // 1 second at 16kHz
    // loop {
    //     let samples: Vec<f32> = capture_audio_chunk(chunk_size);
    //     audio.streaming_asr_feed(&samples)?;
    //
    //     if no_more_audio {
    //         break;
    //     }
    // }

    // For this file-based example, we'll just demonstrate the API
    // Real audio chunk feeding would happen here

    // Finish session and get result
    println!("Finishing streaming session...");
    let text = audio.streaming_asr_finish()?;

    println!("\n--- Session-Based Results ---");
    println!("Text: {}", text);
    println!("\nNote: For full metrics (confidence, RTFx), use the file wrapper API");

    Ok(())
}

/// Demonstrate the simple file wrapper (easier to use)
fn simple_file_transcription(audio: &FluidAudio, audio_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Transcribing: {}", audio_path);
    println!("This processes the file in chunks to minimize memory usage.\n");

    let result = audio.transcribe_file_streaming(audio_path)?;

    println!("\n--- File Wrapper Results ---");
    println!("Text: {}", result.text);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Duration: {:.2}s", result.duration);
    println!("Processing time: {:.2}s", result.processing_time);
    println!("Speed: {:.1}x realtime", result.rtfx);
    println!("\nMemory usage: 99.5% less than regular ASR!");

    Ok(())
}
