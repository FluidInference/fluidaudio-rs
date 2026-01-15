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

    // VAD
    fn fluidaudio_initialize_vad(bridge: *mut std::ffi::c_void, threshold: f32) -> i32;
    fn fluidaudio_is_vad_available(bridge: *mut std::ffi::c_void) -> i32;

    // System Info
    fn fluidaudio_get_platform(out: *mut *mut i8);
    fn fluidaudio_get_chip_name(out: *mut *mut i8);
    fn fluidaudio_get_memory_gb() -> f64;
    fn fluidaudio_is_apple_silicon() -> i32;

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

    pub fn is_asr_available(&self) -> bool {
        unsafe { fluidaudio_is_asr_available(self.ptr) != 0 }
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
