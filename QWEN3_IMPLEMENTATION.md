# Qwen3-ASR Implementation for fluidaudio-rs

This document summarizes the Qwen3-ASR API implementation for the Rust bindings of FluidAudio, addressing issue #7.

## Overview

Qwen3-ASR provides multilingual transcription with excellent support for Japanese, Chinese, Vietnamese, Korean, and 30+ other languages. This implementation exposes both one-shot and streaming APIs to Rust, matching the existing Parakeet TDT API patterns.

## API Summary

### Initialization

```rust
// Initialize Qwen3-ASR
audio.init_qwen3_asr()?;

// Initialize Qwen3 Streaming
audio.init_qwen3_streaming()?;
```

### One-Shot Transcription

```rust
// Transcribe audio file with language hint
let result = audio.qwen3_transcribe_file("japanese_meeting.wav", Some("ja"))?;
println!("Text: {}", result.text);

// Transcribe samples directly (e.g., from microphone)
let samples: Vec<f32> = capture_audio();
let result = audio.qwen3_transcribe_samples(&samples, Some("Japanese"))?;

// Automatic language detection
let result = audio.qwen3_transcribe_file("audio.wav", None)?;
```

### Streaming Transcription

```rust
// Start streaming session
audio.qwen3_streaming_start(
    Some("ja"),  // Language (or None for auto-detect)
    1.0,         // min_audio_seconds - start transcribing after 1s
    2.0,         // chunk_seconds - update every 2s
    30.0,        // max_audio_seconds - buffer limit
)?;

// Feed audio chunks as they arrive
loop {
    let chunk = capture_audio_chunk();

    // Get partial results if ready
    if let Some(partial) = audio.qwen3_streaming_feed(&chunk)? {
        println!("Partial: {}", partial);
    }

    if done { break; }
}

// Get final complete transcription
let final_text = audio.qwen3_streaming_finish()?;
```

## Supported Languages

**East Asian:**
- Japanese (ja)
- Chinese/Mandarin (zh)
- Cantonese (yue)
- Korean (ko)

**Southeast Asian:**
- Vietnamese (vi)
- Indonesian (id)
- Malay (ms)
- Thai (th)
- Filipino (tl)

**European:**
- English (en)
- French (fr)
- German (de)
- Spanish (es)
- Portuguese (pt)
- Italian (it)
- Dutch (nl)
- Swedish (sv)
- Danish (da)
- Finnish (fi)
- Polish (pl)
- Czech (cs)
- Greek (el)
- Hungarian (hu)
- Romanian (ro)

**Other:**
- Russian (ru)
- Arabic (ar)
- Hindi (hi)
- Turkish (tr)
- Persian (fa)
- Macedonian (mk)

## Language Codes

Accepts both:
- ISO 639-1 codes: `"ja"`, `"zh"`, `"vi"`, `"ko"`, `"en"`, etc.
- English names: `"Japanese"`, `"Chinese"`, `"Vietnamese"`, `"Korean"`, `"English"`, etc.

## System Requirements

- macOS 15+ or iOS 18+ (requires Apple's stateful CoreML models)
- Apple Silicon recommended

## Implementation Details

### Files Modified

1. **swift/FluidAudioBridge.swift**
   - Added `qwen3AsrManager` and `qwen3StreamingManager` properties
   - Implemented initialization methods
   - Implemented transcription methods (file and samples)
   - Implemented streaming API (start, feed, finish)
   - Added C FFI exports for all Qwen3 functions

2. **src/ffi/bridge.rs**
   - Added extern "C" declarations for Qwen3 FFI functions
   - Implemented safe Rust wrappers for all Qwen3 operations
   - Handles memory management (string allocation/deallocation)

3. **src/lib.rs**
   - Added public Qwen3 API methods
   - Comprehensive documentation with examples
   - Language code support (ISO 639-1 and English names)

4. **README.md**
   - Added Qwen3-ASR to features section
   - Added usage examples for one-shot and streaming
   - Documented all supported languages
   - Added EN↔JP translation use case example

5. **examples/qwen3_transcribe.rs**
   - Complete example for one-shot transcription
   - Language code reference
   - System info display

6. **examples/qwen3_streaming.rs**
   - Complete example for streaming transcription
   - Simulates real-time audio feed
   - Shows partial and final results

## Testing

### Build Library
```bash
cargo build --lib
```

### Run Examples
```bash
# One-shot transcription
cargo build --example qwen3_transcribe
./target/debug/examples/qwen3_transcribe audio.wav ja

# Streaming transcription
cargo build --example qwen3_streaming
./target/debug/examples/qwen3_streaming audio.wav zh
```

## Use Case: Real-Time EN↔JP Meeting Translation

This implementation directly addresses the use case mentioned in issue #7:

```rust
let audio = FluidAudio::new()?;
audio.init_qwen3_streaming()?;

// Start Japanese transcription
audio.qwen3_streaming_start(Some("ja"), 1.0, 2.0, 30.0)?;

loop {
    let chunk = capture_microphone_audio();

    if let Some(japanese_text) = audio.qwen3_streaming_feed(&chunk)? {
        // Translate japanese_text to English
        let english_text = translate_to_english(japanese_text);
        display_translation(japanese_text, english_text);
    }

    if meeting_ended { break; }
}

let final_jp = audio.qwen3_streaming_finish()?;
```

## API Parity

The Qwen3 API matches the existing Parakeet streaming pattern:

| Parakeet | Qwen3 |
|----------|-------|
| `init_asr()` | `init_qwen3_asr()` |
| `transcribe_file()` | `qwen3_transcribe_file()` |
| `transcribe_samples()` | `qwen3_transcribe_samples()` |
| `init_streaming_asr()` | `init_qwen3_streaming()` |
| `streaming_asr_start()` | `qwen3_streaming_start()` |
| `streaming_asr_feed()` | `qwen3_streaming_feed()` |
| `streaming_asr_finish()` | `qwen3_streaming_finish()` |

### Key Differences

1. **Language Parameter**: Qwen3 accepts optional language hints
2. **Streaming Config**: Qwen3 streaming allows configuring timing parameters
3. **Partial Results**: Qwen3 streaming returns `Option<String>` for partial results

## Notes

- First initialization downloads models (~500MB) from HuggingFace
- Apple Neural Engine compilation takes 20-30 seconds on first run
- Subsequent loads use cached compilations (~1 second)
- Requires macOS 15+ or iOS 18+ for stateful CoreML models
- Works best on Apple Silicon (limited Intel Mac support)

## Testing Availability

User **flp-stephen** from issue #7 has offered to help with testing, particularly for the Tauri desktop application use case.
