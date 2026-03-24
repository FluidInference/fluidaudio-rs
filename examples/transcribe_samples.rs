//! Example: Transcribe raw audio samples
//!
//! This example demonstrates how to use transcribe_samples() for real-time
//! audio applications where you have raw f32 audio buffers rather than files.
//!
//! Usage: cargo run --example transcribe_samples -- path/to/audio.wav

use fluidaudio_rs::FluidAudio;
use std::env;
use std::fs::File;
use std::io::Read;

/// Simple WAV file parser to extract f32 samples
/// Note: This is a basic implementation for demo purposes
fn read_wav_samples(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Skip WAV header (44 bytes for standard PCM WAV)
    // This assumes 16-bit PCM mono WAV file
    let data = &buffer[44..];

    // Convert 16-bit PCM to f32 samples
    let mut samples = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        let normalized = sample as f32 / 32768.0; // Normalize to -1.0 to 1.0
        samples.push(normalized);
    }

    Ok(samples)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get audio file path from command line
    let args: Vec<String> = env::args().collect();
    let audio_path = args.get(1).ok_or("Usage: transcribe_samples <audio_file.wav>")?;

    println!("FluidAudio Samples Transcription Example");
    println!("========================================\n");

    // Create FluidAudio instance
    let audio = FluidAudio::new()?;

    // Print system info
    let info = audio.system_info();
    println!("System: {} ({})", info.chip_name, info.platform);
    println!("Memory: {:.1} GB", info.memory_gb);
    println!("Apple Silicon: {}\n", audio.is_apple_silicon());

    // Initialize ASR
    println!("Initializing ASR (this may take a moment on first run)...");
    audio.init_asr()?;
    println!("ASR initialized!\n");

    // Read WAV file into samples
    println!("Reading audio samples from: {}", audio_path);
    let samples = read_wav_samples(audio_path)?;
    println!("Loaded {} samples ({:.2}s at 16kHz)\n", samples.len(), samples.len() as f32 / 16000.0);

    // Transcribe samples directly
    println!("Transcribing audio samples...");
    let result = audio.transcribe_samples(&samples)?;

    println!("\n--- Results ---");
    println!("Text: {}", result.text);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Duration: {:.2}s", result.duration);
    println!("Processing time: {:.2}s", result.processing_time);
    println!("Speed: {:.1}x realtime", result.rtfx);

    println!("\n💡 This example demonstrates using transcribe_samples() for real-time");
    println!("   audio applications where you capture audio buffers directly from");
    println!("   a microphone or other streaming source, avoiding file I/O overhead.");

    Ok(())
}
