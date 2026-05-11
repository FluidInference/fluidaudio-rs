//! Integration tests for the Swift→Rust FFI bindings.
//!
//! These tests exercise the FFI surface without requiring large model downloads
//! or network access. They verify:
//!   * Bridge lifecycle (create / drop / cleanup) does not crash
//!   * Pre-init availability checks return `false` instead of panicking
//!   * Strings allocated in Swift survive the round trip into Rust
//!   * Error paths (missing files) are reported instead of segfaulting
//!
//! Tests that require network downloads or compiled CoreML models are guarded
//! with `#[ignore]` and can be run via:
//!     cargo test --test ffi_bindings -- --ignored

use fluidaudio_rs::{FluidAudio, FluidAudioError};

/// The bridge can be created and dropped without panicking. Drop must not
/// crash even if no `init_*` method was ever called.
#[test]
fn bridge_create_and_drop_is_safe() {
    let audio = FluidAudio::new().expect("bridge creation should succeed");
    drop(audio);
}

/// Multiple bridges can coexist. The Swift side maintains a `globalBridge`
/// pointer for compatibility but each Rust handle owns its own retained
/// instance, so creating a second bridge must not invalidate the first.
#[test]
fn multiple_bridges_can_be_created() {
    let a = FluidAudio::new().expect("first bridge");
    let b = FluidAudio::new().expect("second bridge");

    // Both should report consistent system information; that proves both
    // pointers are still valid and the FFI calls succeed on each.
    let info_a = a.system_info();
    let info_b = b.system_info();
    assert_eq!(info_a.platform, info_b.platform);
    assert_eq!(info_a.is_apple_silicon, info_b.is_apple_silicon);
}

/// `cleanup` must be idempotent. The destructor calls it again on drop, so
/// calling it explicitly first must not double-free.
#[test]
fn cleanup_is_idempotent() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.cleanup();
    audio.cleanup();
    // Drop runs cleanup once more; should still be safe.
}

/// Before any `init_*` call, every `is_*_available` accessor must return
/// `false` rather than reading uninitialized state.
#[test]
fn availability_is_false_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(!audio.is_asr_available(), "ASR should be unavailable pre-init");
    assert!(
        !audio.is_streaming_asr_available(),
        "streaming ASR should be unavailable pre-init"
    );
    assert!(!audio.is_vad_available(), "VAD should be unavailable pre-init");
    assert!(
        !audio.is_diarization_available(),
        "diarization should be unavailable pre-init"
    );
    assert!(
        !audio.is_qwen3_asr_available(),
        "Qwen3 ASR should be unavailable pre-init"
    );
    assert!(
        !audio.is_qwen3_streaming_available(),
        "Qwen3 streaming should be unavailable pre-init"
    );
}

/// `system_info()` exercises Swift→Rust string ownership: the Swift side
/// allocates with `strdup`, hands a pointer to Rust, and Rust frees it via
/// `fluidaudio_free_string`. The platform string must come back populated.
#[test]
fn system_info_round_trips_swift_strings() {
    let audio = FluidAudio::new().expect("bridge creation");
    let info = audio.system_info();

    assert!(
        info.platform == "macOS" || info.platform == "iOS",
        "unexpected platform: {}",
        info.platform
    );
    assert!(!info.chip_name.is_empty(), "chip name should be populated");
    assert!(info.memory_gb > 0.0, "memory should be reported as positive");
}

/// `is_apple_silicon()` reads `SystemInfo.isAppleSilicon` from Swift and
/// must return a value consistent with `system_info()`.
#[test]
fn is_apple_silicon_matches_system_info() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert_eq!(audio.is_apple_silicon(), audio.system_info().is_apple_silicon);
}

/// `is_intel_mac()` and `is_apple_silicon()` must be mutually exclusive: a
/// process is either arm64 or x86_64, never both, never neither (for the
/// architectures FluidAudio supports).
#[test]
fn is_intel_mac_and_is_apple_silicon_are_exclusive() {
    let audio = FluidAudio::new().expect("bridge creation");
    let arm = audio.is_apple_silicon();
    let intel = audio.is_intel_mac();
    assert!(
        arm != intel,
        "expected exactly one of (apple_silicon, intel_mac) to be true, got both={arm} and intel={intel}"
    );
}

/// File-based methods must validate the path before crossing the FFI boundary,
/// so a missing file returns `FileNotFound` rather than panicking inside Swift.
#[test]
fn missing_file_returns_file_not_found() {
    let audio = FluidAudio::new().expect("bridge creation");
    let err = audio
        .transcribe_file("/this/path/definitely/does/not/exist.wav")
        .expect_err("transcribe_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));

    let err = audio
        .transcribe_file_streaming("/this/path/definitely/does/not/exist.wav")
        .expect_err("transcribe_file_streaming with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));

    let err = audio
        .diarize_file("/this/path/definitely/does/not/exist.wav")
        .expect_err("diarize_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));

    let err = audio
        .qwen3_transcribe_file("/this/path/definitely/does/not/exist.wav", None)
        .expect_err("qwen3_transcribe_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));
}

/// Calling streaming ASR session methods before initializing must surface a
/// bridge error rather than panicking. We don't assert on the message because
/// it depends on the Swift error path; we just want to confirm we get back a
/// `Result::Err` cleanly across the FFI.
#[test]
fn streaming_asr_session_methods_error_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(audio.streaming_asr_start().is_err());
    assert!(audio.streaming_asr_feed(&[0.0_f32; 1600]).is_err());
    assert!(audio.streaming_asr_finish().is_err());
}

/// Same as above for the Qwen3 streaming session methods. On macOS 14 these
/// will fail because Qwen3 requires macOS 15; on macOS 15+ they fail because
/// the manager isn't initialized. Either way, no crash, just an error.
#[test]
fn qwen3_streaming_session_methods_error_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(audio.qwen3_streaming_start(None, 1.0, 2.0, 30.0).is_err());
    assert!(audio.qwen3_streaming_feed(&[0.0_f32; 1600]).is_err());
    assert!(audio.qwen3_streaming_finish().is_err());
}

/// VAD `process` methods must error before `init_vad` rather than crash.
#[test]
fn vad_process_errors_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(audio.vad_process_samples(&[0.0_f32; 4096]).is_err());
    assert!(audio
        .vad_process_file("/this/path/definitely/does/not/exist.wav")
        .is_err());
}

/// VAD `process_file` validates the path on the Rust side and returns
/// `FileNotFound` (not a Swift-side error) when the path doesn't exist.
#[test]
fn vad_process_file_returns_file_not_found() {
    let audio = FluidAudio::new().expect("bridge creation");
    let err = audio
        .vad_process_file("/this/path/definitely/does/not/exist.wav")
        .expect_err("vad_process_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));
}

/// ITN does not require model loading — `TextNormalizer.shared` is always
/// available. Calling `itn_normalize` on a fresh bridge should round-trip a
/// string from Swift back to Rust without crashing.
#[test]
fn itn_normalize_round_trips() {
    let audio = FluidAudio::new().expect("bridge creation");
    // Use a string that won't be modified by ITN regardless of which backend
    // is active. We only care that we got a String back, not that it matches
    // a specific normalized form.
    let result = audio.itn_normalize("hello").expect("itn_normalize");
    assert!(!result.is_empty(), "ITN result should not be empty");
}

/// Sentence-mode ITN must also work without init and accept arbitrary text.
#[test]
fn itn_normalize_sentence_round_trips() {
    let audio = FluidAudio::new().expect("bridge creation");
    let result = audio
        .itn_normalize_sentence("the quick brown fox")
        .expect("itn_normalize_sentence");
    assert!(!result.is_empty());

    let result = audio
        .itn_normalize_sentence_max_span("five plus five", 8)
        .expect("itn_normalize_sentence_max_span");
    assert!(!result.is_empty());
}

/// `itn_is_native_available()` must return a definite bool without crashing,
/// regardless of whether the native NeMo library is loaded in this process.
#[test]
fn itn_is_native_available_returns_bool() {
    let audio = FluidAudio::new().expect("bridge creation");
    let _ = audio.itn_is_native_available();
}

// ---------------------------------------------------------------------------
// Heavier tests below: gated behind `--ignored` because they download models
// from HuggingFace and warm up the Apple Neural Engine (cold start ~20s).
// ---------------------------------------------------------------------------

/// VAD initialization downloads a small CoreML model on first run.
#[test]
#[ignore = "downloads VAD model from HuggingFace on first run"]
fn vad_initializes() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_vad(0.85).expect("VAD init");
    assert!(audio.is_vad_available());
}

/// End-to-end VAD on a silence buffer. We don't assert classification
/// (which depends on the threshold) — only that we get one frame per 4096
/// samples and the per-frame fields look sane.
#[test]
#[ignore = "downloads VAD model from HuggingFace on first run"]
fn vad_processes_silence_buffer() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_vad(0.85).expect("VAD init");

    // 2 seconds of silence at 16kHz mono = 32_000 samples = ~7.8 chunks of 4096.
    // The Swift side pads the trailing partial chunk, so expect ceil(32000/4096) = 8.
    let samples = vec![0.0_f32; 16_000 * 2];
    let frames = audio.vad_process_samples(&samples).expect("vad process");

    assert_eq!(frames.len(), 8, "expected 8 chunks for 2s at 16kHz");
    for frame in &frames {
        assert!(frame.probability >= 0.0 && frame.probability <= 1.0);
        assert!(frame.processing_time >= 0.0);
    }
}

/// End-to-end ASR sanity check on a buffer of silence. We don't assert on the
/// transcript content, only that the call returns successfully and the
/// reported duration matches the input length.
#[test]
#[ignore = "downloads Parakeet TDT models (~600MB) and triggers ANE compilation"]
fn asr_transcribes_silence_buffer() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_asr().expect("ASR init");
    assert!(audio.is_asr_available());

    // 2 seconds of silence at 16kHz mono.
    let samples = vec![0.0_f32; 16_000 * 2];
    let result = audio.transcribe_samples(&samples).expect("transcribe");
    assert!(result.duration >= 1.9 && result.duration <= 2.1);
}
