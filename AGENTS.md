# AGENTS.md

Guidance for coding agents (Claude Code, Cursor, Codex, etc.) working in this repository. This is the canonical agent doc; `CLAUDE.md` points here.

## Project Overview

fluidaudio-rs provides Rust bindings for [FluidAudio](https://github.com/FluidInference/FluidAudio) - a Swift library for ASR, VAD, Speaker Diarization, and TTS on Apple platforms.

## Git Commit Guidelines

- **NEVER add an AI co-author** in commit messages
- Do NOT include `Co-Authored-By: Claude <noreply@anthropic.com>`, `Co-Authored-By: Cursor`, or similar lines
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

## Release Process

This repo publishes to crates.io automatically. Versioning tracks the underlying FluidAudio Swift package: when `Package.swift` pins FluidAudio to `X.Y.Z`, the crate version should be `X.Y.Z`.

### Versioning rule

`Cargo.toml`, `Package.swift` (FluidAudio dep), and the published crates.io version must all match. The `sync-version.yml` workflow runs daily and opens a sync PR when FluidAudio publishes a new release.

### When to open a release PR

Open one whenever any of these are out of sync:

- `Cargo.toml` version differs from the FluidAudio version in `Package.swift`
- A PR (e.g. an API-compat update) bumped the FluidAudio dep without bumping `Cargo.toml`
- crates.io is behind GitHub HEAD

### Steps for an agent to cut a release PR

1. Read the FluidAudio version from `Package.swift` (look for `exact: "X.Y.Z"`).
2. Read the crate version from `Cargo.toml`.
3. If they differ, on a fresh branch `release/vX.Y.Z`:
   - Update `version = "X.Y.Z"` in `Cargo.toml`.
   - Run `cargo update -p fluidaudio-rs` to refresh `Cargo.lock`.
   - Run `cargo check` to verify the build.
4. Commit with a `Bump version to X.Y.Z` message. **Do not add an AI co-author line** (see Git Commit Guidelines above).
5. Push the branch and open a PR titled `Release vX.Y.Z`. Reference the version-tracking issue (#11) and note that tagging `vX.Y.Z` after merge triggers `release.yml`.
6. Do **not** create the tag automatically. A human merges the PR, then pushes the tag.

### What happens after the tag is pushed

`.github/workflows/release.yml` (triggered on `v*` tags):

1. Verifies the tag matches `Cargo.toml`.
2. Runs `cargo build --release`.
3. Runs `cargo publish` using the `CARGO_REGISTRY_TOKEN` repo secret.
4. Creates a GitHub Release with auto-generated notes.

### Prerequisites to verify before tagging

- `CARGO_REGISTRY_TOKEN` secret exists on the repo: `gh secret list -R FluidInference/fluidaudio-rs`
- The token belongs to a crates.io owner of `fluidaudio-rs`: `cargo owner --list fluidaudio-rs`
- The version in `Cargo.toml` is higher than the latest on crates.io

If the token is missing or expired, generate a new one at https://crates.io/me and set it via `gh secret set CARGO_REGISTRY_TOKEN -R FluidInference/fluidaudio-rs`.
