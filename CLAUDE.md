# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fluidaudio-rs provides Rust bindings for [FluidAudio](https://github.com/FluidInference/FluidAudio) - a Swift library for ASR, VAD, Speaker Diarization, and TTS on Apple platforms.

## Git Commit Guidelines

- **NEVER add Claude as co-author** in commit messages
- Do NOT include `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>` or similar lines
- Keep commit messages clean and professional without AI attribution

## Build System

This project uses a custom build system that bridges Rust and Swift via FFI:

1. `build.rs` compiles the Swift package using `swift build`
2. C-compatible functions are exported from Swift using `@_cdecl`
3. Rust calls these functions through `extern "C"` declarations
4. The Swift library is statically linked into the Rust library

### Build Commands

```bash
# Check library builds
cargo build --lib

# Build release
cargo build --release --lib

# Run examples (note: may have linking issues, library is primary target)
cargo build --example <example_name>
```

## Architecture

### FFI Bridge Structure

- **`swift/FluidAudioBridge.swift`** - Swift wrapper around FluidAudio with C FFI exports
- **`src/ffi/bridge.rs`** - Rust FFI declarations and safe wrappers
- **`src/lib.rs`** - Public Rust API

### Adding New Features

When adding new FluidAudio features to the Rust bindings:

1. Add Swift wrapper method in `FluidAudioBridge.swift`
2. Export C-compatible function using `@_cdecl`
3. Add FFI declaration in `src/ffi/bridge.rs` extern "C" block
4. Add safe wrapper method in `FluidAudioBridge` impl block
5. Add public API method in `src/lib.rs` FluidAudio impl block
6. Update README.md with usage example
7. Create example in `examples/` if appropriate

## Platform Requirements

- macOS 14+ or iOS 17+
- Apple Silicon recommended (Intel has limited support)
- Rust 1.70+
- Swift 5.10+

## Known Issues

- Examples may fail to link due to Swift compatibility library symbols
- This is a known issue with Swift static linking
- The library itself (`cargo build --lib`) builds successfully
- Focus on library functionality rather than example executables
