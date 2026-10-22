//! # fluidaudio-rs
//!
//! Rust bindings for [FluidAudio](https://github.com/FluidInference/FluidAudio) -
//! a Swift library for ASR, VAD, Speaker Diarization, and TTS on Apple platforms.
//!
//! ## Features
//!
//! - **ASR (Automatic Speech Recognition)** - High-quality speech-to-text using Parakeet TDT models
//! - **VAD (Voice Activity Detection)** - Detect speech segments in audio
//! - **Speaker Diarization** - Identify and label different speakers in audio
//!
//! ## Requirements
//!
//! - macOS 14+ or iOS 17+
//! - Apple Silicon (M1/M2/M3) recommended
//!
//! ## Example
//!
//! ```rust,no_run
//! use fluidaudio_rs::FluidAudio;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = FluidAudio::new()?;
//!
//!     // Transcribe an audio file
//!     audio.init_asr()?;
//!     let result = audio.transcribe_file("audio.wav")?;
//!     println!("Text: {}", result.text);
//!     println!("Confidence: {:.2}%", result.confidence * 100.0);
//!
//!     Ok(())
//! }
//! ```

mod ffi;

use std::path::Path;
use thiserror::Error;

// Re-export FFI types
pub use ffi::{AsrResult, DiarizationSegment, SystemInfo, VadFrame};

/// Errors that can occur when using FluidAudio
#[derive(Error, Debug)]
pub enum FluidAudioError {
    #[error("FluidAudio not initialized: {0}")]
    NotInitialized(String),

    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Audio file not found: {0}")]
    FileNotFound(String),

    #[error("Swift bridge error: {0}")]
    BridgeError(String),
}

impl From<String> for FluidAudioError {
    fn from(s: String) -> Self {
        FluidAudioError::BridgeError(s)
    }
}

/// CoreML compute-unit preset for the Kokoro TTS pipeline.
///
/// Mirrors FluidAudio's `TtsComputeUnitPreset`; [`Self::as_str`] emits the same
/// kebab-case spellings its `init?(cliValue:)` parser accepts, so the value
/// round-trips across the FFI boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KokoroComputeUnits {
    /// FluidAudio's empirical per-stage mapping (Albert / PostAlbert /
    /// Alignment / Vocoder on the Neural Engine).
    #[default]
    Default,
    /// Every stage on `.cpuAndNeuralEngine`.
    AllAne,
    /// Every stage on `.cpuAndGPU` — skips the ANE entirely.
    CpuAndGpu,
    /// Every stage on `.cpuOnly`.
    CpuOnly,
}

impl KokoroComputeUnits {
    /// Canonical kebab-case name, as accepted by FluidAudio's
    /// `TtsComputeUnitPreset(cliValue:)`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AllAne => "all-ane",
            Self::CpuAndGpu => "cpu-and-gpu",
            Self::CpuOnly => "cpu-only",
        }
    }
}

impl std::str::FromStr for KokoroComputeUnits {
    type Err = FluidAudioError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "default" => Ok(Self::Default),
            "all-ane" | "ane" | "neural-engine" => Ok(Self::AllAne),
            "cpu-and-gpu" | "cpuandgpu" | "gpu" => Ok(Self::CpuAndGpu),
            "cpu-only" | "cpu" | "cpuonly" => Ok(Self::CpuOnly),
            other => Err(FluidAudioError::BridgeError(format!(
                "unknown Kokoro compute-units preset '{other}' \
                 (expected: default, all-ane, cpu-and-gpu, cpu-only)"
            ))),
        }
    }
}

/// Main FluidAudio interface for Rust
///
/// Provides access to ASR and VAD functionality.
pub struct FluidAudio {
    bridge: ffi::FluidAudioBridge,
}

impl FluidAudio {
    /// Create a new FluidAudio instance
    pub fn new() -> Result<Self, FluidAudioError> {
        let bridge = ffi::FluidAudioBridge::new()
            .ok_or_else(|| FluidAudioError::BridgeError("Failed to create bridge".to_string()))?;
        Ok(Self { bridge })
    }

    // ========== ASR Methods ==========

    /// Initialize the ASR (Automatic Speech Recognition) engine
    ///
    /// This downloads and loads the ASR models. First run may take 20-30 seconds
    /// as models are compiled for the Neural Engine.
    pub fn init_asr(&self) -> Result<(), FluidAudioError> {
        self.bridge.initialize_asr().map_err(FluidAudioError::from)
    }

    /// Transcribe an audio file
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata
    pub fn transcribe_file<P: AsRef<Path>>(&self, path: P) -> Result<AsrResult, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .transcribe_file(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Transcribe audio samples directly
    ///
    /// This method accepts raw 16kHz mono audio samples, making it ideal for
    /// real-time audio applications where audio is captured from a microphone
    /// or other streaming source.
    ///
    /// # Arguments
    /// * `samples` - Slice of f32 audio samples (16kHz mono, normalized to -1.0 to 1.0)
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_asr()?;
    ///
    ///     // Simulated audio buffer (16kHz mono)
    ///     let samples: Vec<f32> = vec![0.0; 16000]; // 1 second of silence
    ///
    ///     let result = audio.transcribe_samples(&samples)?;
    ///     println!("Text: {}", result.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<AsrResult, FluidAudioError> {
        self.bridge
            .transcribe_samples(samples)
            .map_err(FluidAudioError::from)
    }

    /// Check if ASR is initialized and ready
    pub fn is_asr_available(&self) -> bool {
        self.bridge.is_asr_available()
    }

    // ========== Streaming ASR Methods ==========

    /// Initialize Streaming ASR (memory-efficient, uses 99.5% less memory than regular ASR)
    ///
    /// Streaming ASR is ideal for long audio files or real-time transcription where
    /// memory usage is a concern. It processes audio in chunks rather than loading
    /// the entire file into memory.
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///
    ///     // Initialize streaming ASR
    ///     audio.init_streaming_asr()?;
    ///
    ///     // Use session-based API for real-time streaming
    ///     audio.streaming_asr_start()?;
    ///
    ///     // Feed audio chunks as they become available
    ///     let chunk1: Vec<f32> = vec![0.0; 16000]; // 1 second
    ///     audio.streaming_asr_feed(&chunk1)?;
    ///
    ///     let chunk2: Vec<f32> = vec![0.0; 16000]; // another second
    ///     audio.streaming_asr_feed(&chunk2)?;
    ///
    ///     // Get final transcription
    ///     let text = audio.streaming_asr_finish()?;
    ///     println!("Transcription: {}", text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn init_streaming_asr(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_streaming_asr()
            .map_err(FluidAudioError::from)
    }

    /// Start a streaming ASR session
    ///
    /// Call this before feeding audio chunks. Use `streaming_asr_feed()` to process
    /// audio chunks, then `streaming_asr_finish()` to get the final result.
    pub fn streaming_asr_start(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .streaming_asr_start()
            .map_err(FluidAudioError::from)
    }

    /// Feed audio samples to the streaming ASR session
    ///
    /// # Arguments
    /// * `samples` - Slice of f32 audio samples (16kHz mono, normalized to -1.0 to 1.0)
    ///
    /// Call this multiple times to process audio in chunks. The transcription engine
    /// will process the audio incrementally.
    pub fn streaming_asr_feed(&self, samples: &[f32]) -> Result<(), FluidAudioError> {
        self.bridge
            .streaming_asr_feed(samples)
            .map_err(FluidAudioError::from)
    }

    /// Finish the streaming ASR session and get the transcription result
    ///
    /// # Returns
    /// * `String` containing the complete transcribed text
    ///
    /// This finalizes processing and returns the full transcription. After calling
    /// this, you must call `streaming_asr_start()` again to start a new session.
    pub fn streaming_asr_finish(&self) -> Result<String, FluidAudioError> {
        self.bridge
            .streaming_asr_finish()
            .map_err(FluidAudioError::from)
    }

    /// Transcribe an audio file using streaming ASR (memory-efficient wrapper)
    ///
    /// This is a convenience method that handles the session lifecycle for you.
    /// For long files or when memory usage is critical, this uses significantly
    /// less memory than `transcribe_file()`.
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_streaming_asr()?;
    ///
    ///     let result = audio.transcribe_file_streaming("long_audio.wav")?;
    ///     println!("Text: {}", result.text);
    ///     println!("RTFx: {:.2}x", result.rtfx);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn transcribe_file_streaming<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<AsrResult, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .transcribe_file_streaming(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Check if streaming ASR is initialized and ready
    pub fn is_streaming_asr_available(&self) -> bool {
        self.bridge.is_streaming_asr_available()
    }

    // ========== VAD Methods ==========

    /// Initialize the VAD (Voice Activity Detection) engine
    ///
    /// # Arguments
    /// * `threshold` - Detection threshold (0.0-1.0, default 0.85)
    pub fn init_vad(&self, threshold: f32) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_vad(threshold)
            .map_err(FluidAudioError::from)
    }

    /// Check if VAD is initialized and ready
    pub fn is_vad_available(&self) -> bool {
        self.bridge.is_vad_available()
    }

    /// Run VAD over an audio file.
    ///
    /// The audio is automatically resampled to 16 kHz mono Float32 and processed
    /// in 4096-sample (256 ms) chunks. One [`VadFrame`] is returned per chunk.
    ///
    /// # Arguments
    /// * `path` - Path to an audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_vad(0.5)?;
    ///     let frames = audio.vad_process_file("speech.wav")?;
    ///     let voiced = frames.iter().filter(|f| f.is_voice_active).count();
    ///     println!("{} / {} chunks classified as voice", voiced, frames.len());
    ///     Ok(())
    /// }
    /// ```
    pub fn vad_process_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Vec<VadFrame>, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .vad_process_file(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Run VAD over raw 16 kHz mono Float32 samples.
    ///
    /// One [`VadFrame`] is returned for every 4096-sample (256 ms) chunk; the
    /// trailing partial chunk (if any) is padded internally.
    pub fn vad_process_samples(&self, samples: &[f32]) -> Result<Vec<VadFrame>, FluidAudioError> {
        self.bridge
            .vad_process_samples(samples)
            .map_err(FluidAudioError::from)
    }

    // ========== Diarization Methods ==========

    /// Initialize the speaker diarization engine
    ///
    /// This downloads and loads the diarization models. First run may take
    /// some time as models are compiled for the Neural Engine.
    ///
    /// # Arguments
    /// * `threshold` - Clustering threshold (0.0-1.0, default 0.6). Lower values
    ///   produce more speakers, higher values merge speakers more aggressively.
    pub fn init_diarization(&self, threshold: f64) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_diarization(threshold)
            .map_err(FluidAudioError::from)
    }

    /// Diarize an audio file using a pre-staged Sortformer model (`.mlpackage`
    /// path), with **no** network download. The caller is responsible for
    /// provisioning and verifying the model. Config is `.balancedV2` (matches
    /// the `SortformerNvidiaLow_v2.mlpackage`). Unlike [`Self::diarize_file`],
    /// this never touches HuggingFace and needs no prior `init_diarization`.
    ///
    /// The compiled model is cached in a writable per-user directory (never next
    /// to `model_path`, so a read-only / air-gapped model location works) and the
    /// loaded handle is retained in-memory, so repeated calls in the same process
    /// reuse it with no reload.
    ///
    /// # Arguments
    /// * `audio` - Path to the audio file (WAV, M4A, MP3, etc.)
    /// * `model_path` - Path to the Sortformer `.mlpackage`
    pub fn diarize_file_with_models<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        audio: P,
        model_path: Q,
    ) -> Result<Vec<DiarizationSegment>, FluidAudioError> {
        let audio_str = audio.as_ref().to_string_lossy();
        if !audio.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(audio_str.to_string()));
        }
        let model_str = model_path.as_ref().to_string_lossy();
        if !model_path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(model_str.to_string()));
        }
        self.bridge
            .diarize_file_with_models(&audio_str, &model_str)
            .map_err(FluidAudioError::from)
    }

    /// Pre-compile the Sortformer `.mlpackage` and warm it, so the first real
    /// diarization is fast. The compiled `.mlmodelc` is written to a writable
    /// per-user cache directory (keyed by a fingerprint of the model — its path,
    /// total size, and newest mtime — never written next
    /// to `model_path`, so a read-only / air-gapped model location works), paying
    /// the one-time ~100s ANE compile up front (e.g. at install time). The loaded
    /// model is also retained in-memory, so within the **same process** subsequent
    /// [`Self::diarize_file_with_models`] calls reuse it with no reload. Across
    /// **separate processes** only the on-disk compiled cache carries over, so the
    /// first diarize in a new process still pays the ~4s warm `MLModel` load (vs
    /// ~100s cold). No audio is processed.
    ///
    /// # Arguments
    /// * `model_path` - Path to the Sortformer `.mlpackage`
    pub fn compile_diarization_model<Q: AsRef<Path>>(
        &self,
        model_path: Q,
    ) -> Result<(), FluidAudioError> {
        let model_str = model_path.as_ref().to_string_lossy();
        if !model_path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(model_str.to_string()));
        }
        self.bridge
            .compile_diarization_model(&model_str)
            .map_err(FluidAudioError::from)
    }

    /// Diarize an audio file to identify speaker segments
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Returns
    /// * `Vec<DiarizationSegment>` containing speaker-labeled time segments
    pub fn diarize_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Vec<DiarizationSegment>, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .diarize_file(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Check if diarization is initialized and ready
    pub fn is_diarization_available(&self) -> bool {
        self.bridge.is_diarization_available()
    }

    // ========== TTS (Kokoro) Methods ==========

    /// Initialize the Kokoro TTS engine with a default voice and language.
    ///
    /// `lang` selects the KokoroAne variant in the Swift bridge (`zh` → Mandarin,
    /// everything else → English). Downloads the variant's model on first run
    /// (FluidAudio-managed cache).
    ///
    /// Uses [`KokoroComputeUnits::Default`]; see
    /// [`init_kokoro_with_compute_units`](Self::init_kokoro_with_compute_units)
    /// to pick another preset.
    pub fn init_kokoro(&self, default_voice: &str, lang: &str) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_kokoro(default_voice, lang)
            .map_err(FluidAudioError::from)
    }

    /// Initialize Kokoro TTS on an explicit CoreML compute-unit preset.
    ///
    /// FluidAudio's default per-stage mapping pins the Albert, PostAlbert,
    /// Alignment and Vocoder stages to the Neural Engine. The other presets let a
    /// caller compare placements or take [`KokoroComputeUnits::CpuAndGpu`] as the
    /// debugging baseline FluidAudio's `KokoroAne.md` recommends for artefact
    /// investigations. They are not a known fix for any host.
    ///
    /// A CoreML prediction failure surfaces past model load: this call only downloads and
    /// loads the mlmodelcs, so it succeeds, and the *first*
    /// [`synthesize_kokoro`](Self::synthesize_kokoro) is what returns the error.
    pub fn init_kokoro_with_compute_units(
        &self,
        default_voice: &str,
        lang: &str,
        compute_units: KokoroComputeUnits,
    ) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_kokoro_with_compute_units(default_voice, lang, compute_units.as_str())
            .map_err(FluidAudioError::from)
    }

    /// Synthesize text via Kokoro TTS in the language the engine was initialized
    /// with (`init_kokoro`'s `lang`): English by default, or Mandarin when
    /// `lang` was `zh` (the `.mandarin` KokoroAne variant).
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    /// * `voice` - Kokoro voice id (e.g. `am_michael`)
    /// * `speed` - Speech rate (0.5-2.0; clamped by FluidAudio)
    ///
    /// # Returns
    /// * `Vec<u8>` - a complete WAV (24 kHz mono, 16-bit PCM)
    pub fn synthesize_kokoro(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
    ) -> Result<Vec<u8>, FluidAudioError> {
        self.bridge
            .kokoro_synthesize(text, voice, speed)
            .map_err(FluidAudioError::from)
    }

    /// Check if Kokoro TTS is initialized and ready.
    pub fn is_kokoro_available(&self) -> bool {
        self.bridge.is_kokoro_available()
    }

    // ========== Qwen3 ASR Methods ==========

    /// Initialize Qwen3-ASR for multilingual transcription (Japanese, Chinese, Vietnamese, etc.)
    ///
    /// Qwen3-ASR supports 30+ languages with high accuracy for non-European languages.
    /// Requires macOS 15+ or iOS 18+.
    ///
    /// # Supported Languages
    /// - East Asian: Japanese, Chinese, Cantonese, Korean
    /// - Southeast Asian: Vietnamese, Indonesian, Malay, Thai, Filipino
    /// - European: English, French, German, Spanish, Portuguese, Italian, Dutch, etc.
    /// - Other: Russian, Arabic, Hindi, Turkish, Persian, and more
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///
    ///     // Initialize Qwen3 ASR
    ///     audio.init_qwen3_asr()?;
    ///
    ///     // Transcribe Japanese audio
    ///     let result = audio.qwen3_transcribe_file("japanese_audio.wav", Some("ja"))?;
    ///     println!("Japanese: {}", result.text);
    ///
    ///     // Or automatic language detection
    ///     let result = audio.qwen3_transcribe_file("audio.wav", None)?;
    ///     println!("Text: {}", result.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn init_qwen3_asr(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_qwen3_asr()
            .map_err(FluidAudioError::from)
    }

    /// Transcribe audio samples using Qwen3-ASR
    ///
    /// # Arguments
    /// * `samples` - Slice of f32 audio samples (16kHz mono, normalized to -1.0 to 1.0)
    /// * `language` - Optional language code (e.g., "ja" for Japanese, "zh" for Chinese).
    ///                Pass None for automatic language detection.
    ///
    /// # Language Codes
    /// Use ISO 639-1 codes or English names:
    /// - Japanese: "ja" or "Japanese"
    /// - Chinese: "zh" or "Chinese"
    /// - Vietnamese: "vi" or "Vietnamese"
    /// - Korean: "ko" or "Korean"
    /// - English: "en" or "English"
    /// - And many more (see init_qwen3_asr documentation)
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_qwen3_asr()?;
    ///
    ///     // Japanese audio samples
    ///     let samples: Vec<f32> = vec![0.0; 16000]; // 1 second
    ///
    ///     let result = audio.qwen3_transcribe_samples(&samples, Some("ja"))?;
    ///     println!("Japanese text: {}", result.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn qwen3_transcribe_samples(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<AsrResult, FluidAudioError> {
        self.bridge
            .qwen3_transcribe_samples(samples, language)
            .map_err(FluidAudioError::from)
    }

    /// Transcribe an audio file using Qwen3-ASR
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    /// * `language` - Optional language code (e.g., "ja", "zh", "vi"). None for auto-detect.
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_qwen3_asr()?;
    ///
    ///     // Transcribe with explicit language hint
    ///     let result = audio.qwen3_transcribe_file("meeting.wav", Some("Japanese"))?;
    ///     println!("Text: {}", result.text);
    ///     println!("RTFx: {:.2}x", result.rtfx);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn qwen3_transcribe_file<P: AsRef<Path>>(
        &self,
        path: P,
        language: Option<&str>,
    ) -> Result<AsrResult, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .qwen3_transcribe_file(&path_str, language)
            .map_err(FluidAudioError::from)
    }

    /// Check if Qwen3-ASR is initialized and ready
    pub fn is_qwen3_asr_available(&self) -> bool {
        self.bridge.is_qwen3_asr_available()
    }

    // ========== Qwen3 Streaming Methods ==========

    /// Initialize Qwen3 Streaming ASR for real-time multilingual transcription
    ///
    /// Streaming mode provides incremental transcription results as audio is fed.
    /// Ideal for real-time applications like meeting transcription or live captions.
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///
    ///     // Initialize Qwen3 streaming
    ///     audio.init_qwen3_streaming()?;
    ///
    ///     // Start session with Japanese language
    ///     audio.qwen3_streaming_start(Some("ja"), 1.0, 2.0, 30.0)?;
    ///
    ///     // Feed audio chunks
    ///     loop {
    ///         let chunk: Vec<f32> = capture_audio_chunk();
    ///
    ///         if let Some(partial) = audio.qwen3_streaming_feed(&chunk)? {
    ///             println!("Partial: {}", partial);
    ///         }
    ///
    ///         if done { break; }
    ///     }
    ///
    ///     // Get final result
    ///     let final_text = audio.qwen3_streaming_finish()?;
    ///     println!("Final: {}", final_text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn init_qwen3_streaming(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_qwen3_streaming()
            .map_err(FluidAudioError::from)
    }

    /// Start a Qwen3 streaming session
    ///
    /// # Arguments
    /// * `language` - Optional language code (e.g., "ja", "zh"). None for auto-detect.
    /// * `min_audio_seconds` - Minimum audio duration before first transcription (default: 1.0)
    /// * `chunk_seconds` - How often to re-transcribe (default: 2.0)
    /// * `max_audio_seconds` - Maximum audio to accumulate (default: 30.0)
    ///
    /// Call this before feeding audio chunks. The streaming engine will provide
    /// partial results as configured by the timing parameters.
    pub fn qwen3_streaming_start(
        &self,
        language: Option<&str>,
        min_audio_seconds: f64,
        chunk_seconds: f64,
        max_audio_seconds: f64,
    ) -> Result<(), FluidAudioError> {
        self.bridge
            .qwen3_streaming_start(language, min_audio_seconds, chunk_seconds, max_audio_seconds)
            .map_err(FluidAudioError::from)
    }

    /// Feed audio samples to Qwen3 streaming session
    ///
    /// # Arguments
    /// * `samples` - Slice of f32 audio samples (16kHz mono, normalized to -1.0 to 1.0)
    ///
    /// # Returns
    /// * `Option<String>` - Partial transcript if enough audio has been accumulated, None otherwise
    ///
    /// Call this repeatedly as audio chunks become available. The engine will return
    /// partial transcripts according to the configuration set in `qwen3_streaming_start`.
    pub fn qwen3_streaming_feed(
        &self,
        samples: &[f32],
    ) -> Result<Option<String>, FluidAudioError> {
        self.bridge
            .qwen3_streaming_feed(samples)
            .map_err(FluidAudioError::from)
    }

    /// Finish Qwen3 streaming session and get final transcription
    ///
    /// # Returns
    /// * `String` - Complete transcription of all audio fed to the session
    ///
    /// This finalizes the session and returns the final transcript. After calling
    /// this, you must call `qwen3_streaming_start()` again to start a new session.
    pub fn qwen3_streaming_finish(&self) -> Result<String, FluidAudioError> {
        self.bridge
            .qwen3_streaming_finish()
            .map_err(FluidAudioError::from)
    }

    /// Check if Qwen3 streaming is initialized and ready
    pub fn is_qwen3_streaming_available(&self) -> bool {
        self.bridge.is_qwen3_streaming_available()
    }

    // ========== System Info ==========

    /// Get system information
    pub fn system_info(&self) -> SystemInfo {
        self.bridge.system_info()
    }

    /// Check if running on Apple Silicon
    pub fn is_apple_silicon(&self) -> bool {
        self.bridge.is_apple_silicon()
    }

    /// Check if running on an Intel Mac (x86_64).
    ///
    /// Local Apple-Silicon-only models (Parakeet, Qwen3, CoreML TTS) will fail
    /// to load on Intel Macs. Apps can use this to gate UI selection of those
    /// models and steer Intel users to cloud transcription instead.
    pub fn is_intel_mac(&self) -> bool {
        self.bridge.is_intel_mac()
    }

    // ========== ITN (Inverse Text Normalization) ==========

    /// Normalize a short spoken-form expression to written form.
    ///
    /// Examples:
    /// - `"two hundred thirty two"` → `"232"`
    /// - `"five dollars and fifty cents"` → `"$5.50"`
    /// - `"period"` → `"."`
    ///
    /// Intended for short fragments. For full sentences with mixed punctuation
    /// commands and ordinary prose, prefer [`itn_normalize_sentence`](Self::itn_normalize_sentence).
    pub fn itn_normalize(&self, text: &str) -> Result<String, FluidAudioError> {
        self.bridge
            .itn_normalize(text)
            .map_err(FluidAudioError::from)
    }

    /// Sentence-mode normalization with sliding-window span matching.
    ///
    /// Uses Apple's NaturalLanguage framework to disambiguate words like
    /// `"period"` between punctuation commands and ordinary nouns/verbs.
    pub fn itn_normalize_sentence(&self, text: &str) -> Result<String, FluidAudioError> {
        self.bridge
            .itn_normalize_sentence(text)
            .map_err(FluidAudioError::from)
    }

    /// Sentence-mode normalization with a caller-controlled maximum span size
    /// (in tokens). Larger spans catch longer multi-word numbers/dates at the
    /// cost of more work per sentence.
    pub fn itn_normalize_sentence_max_span(
        &self,
        text: &str,
        max_span_tokens: u32,
    ) -> Result<String, FluidAudioError> {
        self.bridge
            .itn_normalize_sentence_max_span(text, max_span_tokens)
            .map_err(FluidAudioError::from)
    }

    /// Whether the underlying native NeMo ITN library is loaded.
    ///
    /// When `false`, the Swift-side normalizer falls back to its
    /// Apple-NaturalLanguage-only path (still functional, but with reduced
    /// coverage on long multi-token expressions).
    pub fn itn_is_native_available(&self) -> bool {
        self.bridge.itn_is_native_available()
    }

    // ========== Cleanup ==========

    /// Release all resources
    pub fn cleanup(&self) {
        self.bridge.cleanup()
    }
}

impl Drop for FluidAudio {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_instance() {
        // Note: This test will fail until Swift bridge is properly linked
        // For now, just test the types exist
        let _ = FluidAudioError::NotInitialized("test".to_string());
    }

    #[test]
    fn compute_units_emit_the_spellings_fluidaudio_parses() {
        // The FFI passes the string, not the enum, so `as_str` has to land on a
        // case of FluidAudio's `TtsComputeUnitPreset.init?(cliValue:)`. Nothing
        // in this crate can execute that Swift parser, so the expectations below
        // are transcribed from it and must be re-checked when the pinned
        // FluidAudio version moves (Package.swift, currently 0.14.8 —
        // Sources/FluidAudio/TTS/Shared/TtsComputeUnitPreset.swift). Asserting
        // the literals — not just a `FromStr` round-trip — is what makes a
        // rename of `as_str`'s output fail here instead of at runtime on the
        // Swift side.
        for (units, cli_value) in [
            (KokoroComputeUnits::Default, "default"),
            (KokoroComputeUnits::AllAne, "all-ane"),
            (KokoroComputeUnits::CpuAndGpu, "cpu-and-gpu"),
            (KokoroComputeUnits::CpuOnly, "cpu-only"),
        ] {
            assert_eq!(units.as_str(), cli_value);
            assert_eq!(cli_value.parse::<KokoroComputeUnits>().unwrap(), units);
        }
        assert_eq!(KokoroComputeUnits::default(), KokoroComputeUnits::Default);
    }

    #[test]
    fn compute_units_accepts_aliases_and_rejects_junk() {
        // Every alias FluidAudio 0.14.8's `init?(cliValue:)` accepts, so a
        // caller that learned a spelling from FluidAudio's own `--compute-units`
        // flag is not rejected here before the string ever reaches Swift. Same
        // manual-sync caveat as the test above.
        for (spelling, expected) in [
            ("default", KokoroComputeUnits::Default),
            ("all-ane", KokoroComputeUnits::AllAne),
            ("ane", KokoroComputeUnits::AllAne),
            ("neural-engine", KokoroComputeUnits::AllAne),
            ("cpu-and-gpu", KokoroComputeUnits::CpuAndGpu),
            ("cpuandgpu", KokoroComputeUnits::CpuAndGpu),
            ("gpu", KokoroComputeUnits::CpuAndGpu),
            ("cpu-only", KokoroComputeUnits::CpuOnly),
            ("cpu", KokoroComputeUnits::CpuOnly),
            ("cpuonly", KokoroComputeUnits::CpuOnly),
        ] {
            assert_eq!(
                spelling.parse::<KokoroComputeUnits>().unwrap(),
                expected,
                "alias {spelling:?} should parse"
            );
        }

        // Swift lowercases before matching; so must we, or a mixed-case value
        // would fail on this side and never reach the parser that accepts it.
        assert_eq!(
            "CPU-Only".parse::<KokoroComputeUnits>().unwrap(),
            KokoroComputeUnits::CpuOnly
        );

        // Unknown presets fail in Rust, before the FFI call — the error names
        // the canonical spellings rather than surfacing a bare `-1` from Swift.
        let err = "tpu".parse::<KokoroComputeUnits>().unwrap_err().to_string();
        assert!(
            err.contains("tpu"),
            "error should quote the bad value: {err}"
        );
        assert!(
            err.contains("cpu-and-gpu"),
            "error should list the accepted spellings: {err}"
        );
    }
}
