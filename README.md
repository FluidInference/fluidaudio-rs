# fluidaudio-rs

Rust bindings for [FluidAudio](https://github.com/FluidInference/FluidAudio) - a Swift library for ASR, VAD, Speaker Diarization, and TTS on Apple platforms.

## Features

- **ASR (Automatic Speech Recognition)** - High-quality speech-to-text using Parakeet TDT models
- **VAD (Voice Activity Detection)** - Detect speech segments in audio
- **Speaker Diarization** - Identify and label different speakers in audio

## Requirements

- macOS 14+ or iOS 17+
- Apple Silicon (M1/M2/M3) recommended
- Rust 1.70+
- Swift 5.10+

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fluidaudio-rs = "0.1"
```

## Usage

### Speech-to-Text (ASR)

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;

    // Check system info
    let info = audio.system_info();
    println!("Running on: {} ({})", info.chip_name, info.platform);
    println!("Apple Silicon: {}", audio.is_apple_silicon());

    // Initialize ASR (downloads models on first run)
    audio.init_asr()?;

    // Transcribe an audio file
    let result = audio.transcribe_file("audio.wav")?;
    println!("Text: {}", result.text);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Processing speed: {:.1}x realtime", result.rtfx);

    Ok(())
}
```

### Voice Activity Detection (VAD)

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;

    // Initialize VAD with threshold (0.0-1.0)
    audio.init_vad(0.85)?;

    println!("VAD available: {}", audio.is_vad_available());

    Ok(())
}
```

### Speaker Diarization

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;

    // Initialize diarization with clustering threshold (0.0-1.0)
    // Lower = more speakers, higher = fewer speakers
    audio.init_diarization(0.6)?;

    // Diarize an audio file
    let segments = audio.diarize_file("meeting.wav")?;
    for seg in &segments {
        println!(
            "[{:.2}s - {:.2}s] {}",
            seg.start_time, seg.end_time, seg.speaker_id
        );
    }

    Ok(())
}
```

## Model Loading

First initialization downloads and compiles ML models (~500MB total). This can take 20-30 seconds as Apple's Neural Engine compiles the models. Subsequent loads use cached compilations (~1 second).

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | Full support |
| macOS (Intel) | Limited (no ASR) |
| iOS | Full support |
| Linux/Windows | Not supported |

## How it Works

This crate uses a C FFI bridge to communicate between Rust and Swift:

1. The Swift layer (`FluidAudioBridge`) wraps the FluidAudio library
2. C-compatible functions are exported using `@_cdecl`
3. Rust calls these functions through `extern "C"` declarations
4. The build.rs script compiles the Swift package and links it

## License

MIT
