use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell Cargo to rerun if Swift files change
    println!("cargo:rerun-if-changed=swift/");
    println!("cargo:rerun-if-changed=Package.swift");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // Build the Swift package first to get FluidAudio dependency
    println!("cargo:warning=Building Swift package...");

    let swift_build_dir = out_dir.join("swift-build");
    std::fs::create_dir_all(&swift_build_dir).expect("Failed to create swift-build directory");

    // Build Swift package in release mode
    let status = Command::new("swift")
        .args(&[
            "build",
            "-c",
            "release",
            "--build-path",
            swift_build_dir.to_str().unwrap(),
        ])
        .current_dir(&manifest_dir)
        .status()
        .expect("Failed to run swift build");

    if !status.success() {
        panic!("Swift package build failed");
    }

    // Find the built library
    let lib_path = swift_build_dir.join("release");

    // Link the Swift library
    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=static=FluidAudioBridge");

    // Link Apple frameworks
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // Link Swift runtime
    println!("cargo:rustc-link-lib=dylib=swiftCore");

    // Link C++ standard library (needed for FastClusterWrapper.cpp in FluidAudio)
    println!("cargo:rustc-link-lib=c++");
}
