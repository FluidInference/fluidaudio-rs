# fluidaudio-rs

Rust bindings for [FluidAudio](https://github.com/FluidInference/FluidAudio) - a Swift library for ASR, VAD, Speaker Diarization, and TTS on Apple platforms.

## Features

- **ASR (Automatic Speech Recognition)**
  - **Parakeet TDT** - High-quality speech-to-text for 25 European languages
    - Regular ASR for maximum accuracy
    - Streaming ASR for 99.5% less memory usage
    - Real-time sample transcription
  - **Qwen3-ASR** - Multilingual transcription with excellent Japanese, Chinese, Vietnamese support
    - 30+ languages including East Asian, Southeast Asian, European, and more
    - Both one-shot and streaming modes available
    - Language hint support or automatic detection
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

#### Real-Time Audio (Samples)

For real-time audio applications, you can transcribe raw audio samples directly without file I/O:

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;
    audio.init_asr()?;

    // Audio samples from microphone or streaming source
    // (16kHz mono, normalized to -1.0 to 1.0)
    let samples: Vec<f32> = capture_audio_from_mic();

    // Transcribe samples directly
    let result = audio.transcribe_samples(&samples)?;
    println!("Text: {}", result.text);

    Ok(())
}
```

This is ideal for meeting transcription apps, voice assistants, and other real-time scenarios where writing to temporary files adds unnecessary overhead.

### Multilingual ASR with Qwen3 (Japanese, Chinese, Vietnamese, etc.)

Qwen3-ASR provides excellent multilingual support, especially for East Asian and Southeast Asian languages. Perfect for applications that need Japanese, Chinese, Vietnamese, or Korean transcription.

#### Supported Languages

**East Asian:** Japanese, Chinese (Mandarin), Cantonese, Korean
**Southeast Asian:** Vietnamese, Indonesian, Malay, Thai, Filipino
**European:** English, French, German, Spanish, Portuguese, Italian, Dutch, Swedish, Danish, Finnish, Polish, Czech, Greek, Hungarian, Romanian
**Other:** Russian, Arabic, Hindi, Turkish, Persian, Macedonian

#### Basic Usage

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;

    // Initialize Qwen3-ASR
    audio.init_qwen3_asr()?;

    // Transcribe Japanese audio with language hint
    let result = audio.qwen3_transcribe_file("japanese_meeting.wav", Some("ja"))?;
    println!("Japanese: {}", result.text);

    // Transcribe with automatic language detection
    let result = audio.qwen3_transcribe_file("multilingual_audio.wav", None)?;
    println!("Auto-detected: {}", result.text);

    Ok(())
}
```

#### Real-Time Samples (Japanese/Chinese/etc.)

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;
    audio.init_qwen3_asr()?;

    // Audio samples from microphone (Japanese speaker)
    let samples: Vec<f32> = capture_japanese_audio();

    // Transcribe with language hint
    let result = audio.qwen3_transcribe_samples(&samples, Some("Japanese"))?;
    println!("日本語: {}", result.text);

    Ok(())
}
```

#### Streaming Mode (Real-Time EN↔JP Translation Use Case)

For real-time meeting translation or live transcription:

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;
    audio.init_qwen3_streaming()?;

    // Start streaming session for Japanese
    // Parameters: language, min_audio_secs, chunk_secs, max_audio_secs
    audio.qwen3_streaming_start(Some("ja"), 1.0, 2.0, 30.0)?;

    // Feed audio chunks as they arrive from microphone
    loop {
        let chunk: Vec<f32> = capture_audio_chunk(); // From mic

        // Get partial transcription if ready
        if let Some(partial_text) = audio.qwen3_streaming_feed(&chunk)? {
            println!("Partial (JP): {}", partial_text);
            // Translate partial_text to English here...
        }

        if meeting_ended {
            break;
        }
    }

    // Get final complete transcription
    let final_text = audio.qwen3_streaming_finish()?;
    println!("Final (JP): {}", final_text);

    Ok(())
}
```

**Language Codes:** Use ISO 639-1 codes (`"ja"`, `"zh"`, `"vi"`, `"ko"`) or English names (`"Japanese"`, `"Chinese"`, `"Vietnamese"`, `"Korean"`).

**System Requirements:** Qwen3-ASR requires macOS 15+ or iOS 18+ (uses Apple's latest CoreML stateful models).

### Streaming ASR (Memory-Efficient)

Streaming ASR uses **99.5% less memory** than regular ASR by processing audio in chunks rather than loading entire files. Perfect for long recordings or resource-constrained environments.

#### Simple File Wrapper

The easiest way to use streaming ASR - just like regular transcription but memory-efficient:

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;
    audio.init_streaming_asr()?;

    // Process large files with minimal memory
    let result = audio.transcribe_file_streaming("long_meeting.wav")?;
    println!("Text: {}", result.text);
    println!("Speed: {:.1}x realtime", result.rtfx);

    Ok(())
}
```

#### Session-Based API

For real-time streaming with full control:

```rust
use fluidaudio_rs::FluidAudio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = FluidAudio::new()?;
    audio.init_streaming_asr()?;

    // Start streaming session
    audio.streaming_asr_start()?;

    // Feed audio chunks as they arrive
    loop {
        let samples: Vec<f32> = capture_audio_chunk(); // From mic, network, etc.
        audio.streaming_asr_feed(&samples)?;

        if done {
            break;
        }
    }

    // Get final transcription
    let text = audio.streaming_asr_finish()?;
    println!("Transcription: {}", text);

    Ok(())
}
```

**When to use Streaming ASR:**
- Long audio files (> 5 minutes)
- Real-time transcription with live audio feed
- Memory-constrained environments
- Continuous streaming scenarios

**When to use Regular ASR:**
- Short audio clips (< 5 minutes)
- When you need maximum accuracy on complete audio
- Batch processing where memory isn't a concern

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
