//! Swift bridge definitions for FluidAudio bindings
//!
//! Using manual FFI instead of swift-bridge to avoid complexity with Vec types.

// Raw FFI functions - called directly from Rust, implemented in Swift
#[link(name = "FluidAudioBridge")]
extern "C" {
    // Constructor / Destructor
    fn fluidaudio_bridge_create() -> *mut std::ffi::c_void;
    fn fluidaudio_bridge_destroy(bridge: *mut std::ffi::c_void);

    // ASR
    fn fluidaudio_initialize_asr(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_transcribe_file(
        bridge: *mut std::ffi::c_void,
        path: *const i8,
        out_text: *mut *mut i8,
        out_confidence: *mut f32,
        out_duration: *mut f64,
        out_processing_time: *mut f64,
        out_rtfx: *mut f32,
    ) -> i32;
    fn fluidaudio_is_asr_available(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_transcribe_samples(
        bridge: *mut std::ffi::c_void,
        samples: *const f32,
        sample_count: u32,
        out_text: *mut *mut i8,
        out_confidence: *mut f32,
        out_duration: *mut f64,
        out_processing_time: *mut f64,
        out_rtfx: *mut f32,
    ) -> i32;

    // Streaming ASR
    fn fluidaudio_initialize_streaming_asr(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_streaming_asr_start(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_streaming_asr_feed(
        bridge: *mut std::ffi::c_void,
        samples: *const f32,
        count: u32,
    ) -> i32;
    fn fluidaudio_streaming_asr_finish(
        bridge: *mut std::ffi::c_void,
        out_text: *mut *mut i8,
    ) -> i32;
    fn fluidaudio_transcribe_file_streaming(
        bridge: *mut std::ffi::c_void,
        path: *const i8,
        out_text: *mut *mut i8,
        out_confidence: *mut f32,
        out_duration: *mut f64,
        out_processing_time: *mut f64,
        out_rtfx: *mut f32,
    ) -> i32;
    fn fluidaudio_is_streaming_asr_available(bridge: *mut std::ffi::c_void) -> i32;

    // VAD
    fn fluidaudio_initialize_vad(bridge: *mut std::ffi::c_void, threshold: f32) -> i32;
    fn fluidaudio_is_vad_available(bridge: *mut std::ffi::c_void) -> i32;

    // Diarization
    fn fluidaudio_initialize_diarization(bridge: *mut std::ffi::c_void, threshold: f64) -> i32;
    fn fluidaudio_diarize_file(
        bridge: *mut std::ffi::c_void,
        path: *const i8,
        out_speaker_ids: *mut *mut *mut i8,
        out_start_times: *mut *mut f32,
        out_end_times: *mut *mut f32,
        out_quality_scores: *mut *mut f32,
        out_count: *mut u32,
    ) -> i32;
    fn fluidaudio_is_diarization_available(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_free_diarization_result(
        speaker_ids: *mut *mut i8,
        start_times: *mut f32,
        end_times: *mut f32,
        quality_scores: *mut f32,
        count: u32,
    );

    // System Info
    fn fluidaudio_get_platform(out: *mut *mut i8);
    fn fluidaudio_get_chip_name(out: *mut *mut i8);
    fn fluidaudio_get_memory_gb() -> f64;
    fn fluidaudio_is_apple_silicon() -> i32;

    // Qwen3 ASR
    fn fluidaudio_initialize_qwen3_asr(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_qwen3_transcribe_samples(
        bridge: *mut std::ffi::c_void,
        samples: *const f32,
        sample_count: u32,
        language: *const i8,
        out_text: *mut *mut i8,
        out_confidence: *mut f32,
        out_duration: *mut f64,
        out_processing_time: *mut f64,
        out_rtfx: *mut f32,
    ) -> i32;
    fn fluidaudio_qwen3_transcribe_file(
        bridge: *mut std::ffi::c_void,
        path: *const i8,
        language: *const i8,
        out_text: *mut *mut i8,
        out_confidence: *mut f32,
        out_duration: *mut f64,
        out_processing_time: *mut f64,
        out_rtfx: *mut f32,
    ) -> i32;
    fn fluidaudio_is_qwen3_asr_available(bridge: *mut std::ffi::c_void) -> i32;

    // Qwen3 Streaming
    fn fluidaudio_initialize_qwen3_streaming(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_qwen3_streaming_start(
        bridge: *mut std::ffi::c_void,
        language: *const i8,
        min_audio_seconds: f64,
        chunk_seconds: f64,
        max_audio_seconds: f64,
    ) -> i32;
    fn fluidaudio_qwen3_streaming_feed(
        bridge: *mut std::ffi::c_void,
        samples: *const f32,
        count: u32,
        out_partial_text: *mut *mut i8,
    ) -> i32;
    fn fluidaudio_qwen3_streaming_finish(
        bridge: *mut std::ffi::c_void,
        out_text: *mut *mut i8,
    ) -> i32;
    fn fluidaudio_is_qwen3_streaming_available(bridge: *mut std::ffi::c_void) -> i32;

    // Cleanup
    fn fluidaudio_cleanup(bridge: *mut std::ffi::c_void);

    // String free
    fn fluidaudio_free_string(s: *mut i8);
}

use std::ffi::{CStr, CString};

/// Safe wrapper for the FluidAudio bridge
pub struct FluidAudioBridge {
    ptr: *mut std::ffi::c_void,
}

// The Swift bridge is thread-safe as it uses internal synchronization
unsafe impl Send for FluidAudioBridge {}
unsafe impl Sync for FluidAudioBridge {}

impl FluidAudioBridge {
    pub fn new() -> Option<Self> {
        let ptr = unsafe { fluidaudio_bridge_create() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn initialize_asr(&self) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_asr(self.ptr) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize ASR".to_string())
        }
    }

    pub fn transcribe_file(&self, path: &str) -> Result<AsrResult, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path")?;

        let mut text_ptr: *mut i8 = std::ptr::null_mut();
        let mut confidence: f32 = 0.0;
        let mut duration: f64 = 0.0;
        let mut processing_time: f64 = 0.0;
        let mut rtfx: f32 = 0.0;

        let result = unsafe {
            fluidaudio_transcribe_file(
                self.ptr,
                c_path.as_ptr(),
                &mut text_ptr,
                &mut confidence,
                &mut duration,
                &mut processing_time,
                &mut rtfx,
            )
        };

        if result != 0 {
            return Err("Transcription failed".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(AsrResult {
            text,
            confidence,
            duration,
            processing_time,
            rtfx,
        })
    }

    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<AsrResult, String> {
        let mut text_ptr: *mut i8 = std::ptr::null_mut();
        let mut confidence: f32 = 0.0;
        let mut duration: f64 = 0.0;
        let mut processing_time: f64 = 0.0;
        let mut rtfx: f32 = 0.0;

        let result = unsafe {
            fluidaudio_transcribe_samples(
                self.ptr,
                samples.as_ptr(),
                samples.len() as u32,
                &mut text_ptr,
                &mut confidence,
                &mut duration,
                &mut processing_time,
                &mut rtfx,
            )
        };

        if result != 0 {
            return Err("Transcription failed".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(AsrResult {
            text,
            confidence,
            duration,
            processing_time,
            rtfx,
        })
    }

    pub fn is_asr_available(&self) -> bool {
        unsafe { fluidaudio_is_asr_available(self.ptr) != 0 }
    }

    pub fn initialize_streaming_asr(&self) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_streaming_asr(self.ptr) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize streaming ASR".to_string())
        }
    }

    pub fn streaming_asr_start(&self) -> Result<(), String> {
        let result = unsafe { fluidaudio_streaming_asr_start(self.ptr) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to start streaming ASR session".to_string())
        }
    }

    pub fn streaming_asr_feed(&self, samples: &[f32]) -> Result<(), String> {
        let result = unsafe {
            fluidaudio_streaming_asr_feed(self.ptr, samples.as_ptr(), samples.len() as u32)
        };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to feed samples to streaming ASR".to_string())
        }
    }

    pub fn streaming_asr_finish(&self) -> Result<String, String> {
        let mut text_ptr: *mut i8 = std::ptr::null_mut();

        let result = unsafe { fluidaudio_streaming_asr_finish(self.ptr, &mut text_ptr) };

        if result != 0 {
            return Err("Failed to finish streaming ASR session".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(text)
    }

    pub fn transcribe_file_streaming(&self, path: &str) -> Result<AsrResult, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path")?;

        let mut text_ptr: *mut i8 = std::ptr::null_mut();
        let mut confidence: f32 = 0.0;
        let mut duration: f64 = 0.0;
        let mut processing_time: f64 = 0.0;
        let mut rtfx: f32 = 0.0;

        let result = unsafe {
            fluidaudio_transcribe_file_streaming(
                self.ptr,
                c_path.as_ptr(),
                &mut text_ptr,
                &mut confidence,
                &mut duration,
                &mut processing_time,
                &mut rtfx,
            )
        };

        if result != 0 {
            return Err("Streaming file transcription failed".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(AsrResult {
            text,
            confidence,
            duration,
            processing_time,
            rtfx,
        })
    }

    pub fn is_streaming_asr_available(&self) -> bool {
        unsafe { fluidaudio_is_streaming_asr_available(self.ptr) != 0 }
    }

    pub fn initialize_vad(&self, threshold: f32) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_vad(self.ptr, threshold) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize VAD".to_string())
        }
    }

    pub fn is_vad_available(&self) -> bool {
        unsafe { fluidaudio_is_vad_available(self.ptr) != 0 }
    }

    pub fn initialize_diarization(&self, threshold: f64) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_diarization(self.ptr, threshold) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize diarization".to_string())
        }
    }

    pub fn diarize_file(&self, path: &str) -> Result<Vec<DiarizationSegment>, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path")?;

        let mut speaker_ids_ptr: *mut *mut i8 = std::ptr::null_mut();
        let mut start_times_ptr: *mut f32 = std::ptr::null_mut();
        let mut end_times_ptr: *mut f32 = std::ptr::null_mut();
        let mut quality_scores_ptr: *mut f32 = std::ptr::null_mut();
        let mut count: u32 = 0;

        let result = unsafe {
            fluidaudio_diarize_file(
                self.ptr,
                c_path.as_ptr(),
                &mut speaker_ids_ptr,
                &mut start_times_ptr,
                &mut end_times_ptr,
                &mut quality_scores_ptr,
                &mut count,
            )
        };

        if result != 0 {
            return Err("Diarization failed".to_string());
        }

        let mut segments = Vec::with_capacity(count as usize);

        if count > 0
            && !speaker_ids_ptr.is_null()
            && !start_times_ptr.is_null()
            && !end_times_ptr.is_null()
            && !quality_scores_ptr.is_null()
        {
            for i in 0..count as usize {
                let id_ptr = unsafe { *speaker_ids_ptr.add(i) };
                let speaker_id = if id_ptr.is_null() {
                    String::new()
                } else {
                    unsafe { CStr::from_ptr(id_ptr) }
                        .to_string_lossy()
                        .into_owned()
                };
                segments.push(DiarizationSegment {
                    speaker_id,
                    start_time: unsafe { *start_times_ptr.add(i) },
                    end_time: unsafe { *end_times_ptr.add(i) },
                    quality_score: unsafe { *quality_scores_ptr.add(i) },
                });
            }

            unsafe {
                fluidaudio_free_diarization_result(
                    speaker_ids_ptr,
                    start_times_ptr,
                    end_times_ptr,
                    quality_scores_ptr,
                    count,
                )
            };
        }

        Ok(segments)
    }

    pub fn is_diarization_available(&self) -> bool {
        unsafe { fluidaudio_is_diarization_available(self.ptr) != 0 }
    }

    pub fn initialize_qwen3_asr(&self) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_qwen3_asr(self.ptr) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize Qwen3 ASR".to_string())
        }
    }

    pub fn qwen3_transcribe_samples(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<AsrResult, String> {
        let c_language = language.and_then(|l| CString::new(l).ok());

        let mut text_ptr: *mut i8 = std::ptr::null_mut();
        let mut confidence: f32 = 0.0;
        let mut duration: f64 = 0.0;
        let mut processing_time: f64 = 0.0;
        let mut rtfx: f32 = 0.0;

        let result = unsafe {
            fluidaudio_qwen3_transcribe_samples(
                self.ptr,
                samples.as_ptr(),
                samples.len() as u32,
                c_language
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                &mut text_ptr,
                &mut confidence,
                &mut duration,
                &mut processing_time,
                &mut rtfx,
            )
        };

        if result != 0 {
            return Err("Qwen3 transcription failed".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(AsrResult {
            text,
            confidence,
            duration,
            processing_time,
            rtfx,
        })
    }

    pub fn qwen3_transcribe_file(
        &self,
        path: &str,
        language: Option<&str>,
    ) -> Result<AsrResult, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path")?;
        let c_language = language.and_then(|l| CString::new(l).ok());

        let mut text_ptr: *mut i8 = std::ptr::null_mut();
        let mut confidence: f32 = 0.0;
        let mut duration: f64 = 0.0;
        let mut processing_time: f64 = 0.0;
        let mut rtfx: f32 = 0.0;

        let result = unsafe {
            fluidaudio_qwen3_transcribe_file(
                self.ptr,
                c_path.as_ptr(),
                c_language
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                &mut text_ptr,
                &mut confidence,
                &mut duration,
                &mut processing_time,
                &mut rtfx,
            )
        };

        if result != 0 {
            return Err("Qwen3 file transcription failed".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(AsrResult {
            text,
            confidence,
            duration,
            processing_time,
            rtfx,
        })
    }

    pub fn is_qwen3_asr_available(&self) -> bool {
        unsafe { fluidaudio_is_qwen3_asr_available(self.ptr) != 0 }
    }

    pub fn initialize_qwen3_streaming(&self) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_qwen3_streaming(self.ptr) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize Qwen3 Streaming".to_string())
        }
    }

    pub fn qwen3_streaming_start(
        &self,
        language: Option<&str>,
        min_audio_seconds: f64,
        chunk_seconds: f64,
        max_audio_seconds: f64,
    ) -> Result<(), String> {
        let c_language = language.and_then(|l| CString::new(l).ok());

        let result = unsafe {
            fluidaudio_qwen3_streaming_start(
                self.ptr,
                c_language
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                min_audio_seconds,
                chunk_seconds,
                max_audio_seconds,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err("Failed to start Qwen3 streaming session".to_string())
        }
    }

    pub fn qwen3_streaming_feed(&self, samples: &[f32]) -> Result<Option<String>, String> {
        let mut partial_text_ptr: *mut i8 = std::ptr::null_mut();

        let result = unsafe {
            fluidaudio_qwen3_streaming_feed(
                self.ptr,
                samples.as_ptr(),
                samples.len() as u32,
                &mut partial_text_ptr,
            )
        };

        if result != 0 {
            return Err("Failed to feed samples to Qwen3 streaming".to_string());
        }

        if partial_text_ptr.is_null() {
            Ok(None)
        } else {
            let text = unsafe { CStr::from_ptr(partial_text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(partial_text_ptr) };
            Ok(Some(text))
        }
    }

    pub fn qwen3_streaming_finish(&self) -> Result<String, String> {
        let mut text_ptr: *mut i8 = std::ptr::null_mut();

        let result = unsafe { fluidaudio_qwen3_streaming_finish(self.ptr, &mut text_ptr) };

        if result != 0 {
            return Err("Failed to finish Qwen3 streaming session".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(text)
    }

    pub fn is_qwen3_streaming_available(&self) -> bool {
        unsafe { fluidaudio_is_qwen3_streaming_available(self.ptr) != 0 }
    }

    pub fn system_info(&self) -> SystemInfo {
        let mut platform_ptr: *mut i8 = std::ptr::null_mut();
        let mut chip_ptr: *mut i8 = std::ptr::null_mut();

        unsafe {
            fluidaudio_get_platform(&mut platform_ptr);
            fluidaudio_get_chip_name(&mut chip_ptr);
        }

        let platform = unsafe {
            if platform_ptr.is_null() {
                "unknown".to_string()
            } else {
                let s = CStr::from_ptr(platform_ptr).to_string_lossy().into_owned();
                fluidaudio_free_string(platform_ptr);
                s
            }
        };

        let chip_name = unsafe {
            if chip_ptr.is_null() {
                "unknown".to_string()
            } else {
                let s = CStr::from_ptr(chip_ptr).to_string_lossy().into_owned();
                fluidaudio_free_string(chip_ptr);
                s
            }
        };

        let memory_gb = unsafe { fluidaudio_get_memory_gb() };
        let is_apple_silicon = unsafe { fluidaudio_is_apple_silicon() != 0 };

        SystemInfo {
            platform,
            chip_name,
            memory_gb,
            is_apple_silicon,
        }
    }

    pub fn is_apple_silicon(&self) -> bool {
        unsafe { fluidaudio_is_apple_silicon() != 0 }
    }

    pub fn cleanup(&self) {
        unsafe { fluidaudio_cleanup(self.ptr) };
    }
}

impl Drop for FluidAudioBridge {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { fluidaudio_bridge_destroy(self.ptr) };
        }
    }
}

// Result types
#[derive(Debug, Clone)]
pub struct AsrResult {
    pub text: String,
    pub confidence: f32,
    pub duration: f64,
    pub processing_time: f64,
    pub rtfx: f32,
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub platform: String,
    pub chip_name: String,
    pub memory_gb: f64,
    pub is_apple_silicon: bool,
}

/// A speaker segment from diarization
#[derive(Debug, Clone)]
pub struct DiarizationSegment {
    /// Speaker identifier (e.g. "SPEAKER_00", "SPEAKER_01")
    pub speaker_id: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
}

impl DiarizationSegment {
    /// Duration of this segment in seconds
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }
}
