use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("FluidAudio - Real-time Sample Transcription Example");
    println!("=====================================================\n");

    // Create FluidAudio instance
    let audio = FluidAudio::new()?;
    println!("✓ FluidAudio instance created");

    // Initialize ASR
    println!("Initializing ASR (this may take 20-30 seconds on first run)...");
    audio.init_asr()?;
    println!("✓ ASR initialized\n");

    // Example 1: Transcribe 1 second of silence (16kHz mono)
    println!("Example 1: Transcribing 1 second of silence...");
    let silence_samples: Vec<f32> = vec![0.0; 16000];
    let result = audio.transcribe_samples(&silence_samples)?;
    println!("  Text: '{}'", result.text);
    println!("  Confidence: {:.2}%", result.confidence * 100.0);
    println!("  Duration: {:.2}s", result.duration);
    println!("  Processing time: {:.3}s", result.processing_time);
    println!("  RTFx (realtime factor): {:.1}x\n", result.rtfx);

    // Example 2: Transcribe a longer buffer (3 seconds)
    println!("Example 2: Transcribing 3 seconds of silence...");
    let longer_silence: Vec<f32> = vec![0.0; 16000 * 3];
    let result = audio.transcribe_samples(&longer_silence)?;
    println!("  Text: '{}'", result.text);
    println!("  Duration: {:.2}s", result.duration);
    println!("  Processing time: {:.3}s", result.processing_time);
    println!("  RTFx: {:.1}x\n", result.rtfx);

    // Example 3: Demonstrate typical use case - process audio chunks
    println!("Example 3: Processing audio chunks (simulated streaming)...");
    let chunk_size = 16000; // 1 second chunks at 16kHz
    let num_chunks = 3;

    for i in 0..num_chunks {
        // In a real application, these would be actual audio samples
        // from a microphone or audio stream
        let chunk: Vec<f32> = vec![0.0; chunk_size];

        let result = audio.transcribe_samples(&chunk)?;
        println!("  Chunk {}: '{}' (RTFx: {:.1}x)", i + 1, result.text, result.rtfx);
    }

    println!("\n✓ All examples completed successfully!");
    println!("\nNote: In real usage, you would replace the silence samples with");
    println!("actual audio data from a microphone, audio file, or network stream.");

    Ok(())
}
