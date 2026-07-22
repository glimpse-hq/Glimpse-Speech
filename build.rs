use std::path::PathBuf;
use std::process::Command;

fn main() {
    let macos_arm = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64");

    if macos_arm
        && (std::env::var_os("CARGO_FEATURE_CLEANUP_APPLE").is_some()
            || std::env::var_os("CARGO_FEATURE_APPLE_SPEECH").is_some())
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        // Weak-link so binaries still launch on macOS < 26, where the
        // framework is absent. cleanup.rs guards every call by OS version.
        println!("cargo:rustc-link-arg=-Wl,-weak_framework,FoundationModels");
    }

    if macos_arm && std::env::var_os("CARGO_FEATURE_APPLE_SPEECH").is_some() {
        compile_apple_speech_shim();
    }
}

fn compile_apple_speech_shim() {
    println!("cargo:rerun-if-changed=swift/apple_speech.swift");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let lib_path = out_dir.join("libglimpse_apple_speech.a");

    let status = Command::new("swiftc")
        .args([
            "-O",
            "-parse-as-library",
            "-module-name",
            "glimpse_apple_speech",
            "-emit-library",
            "-static",
            "-target",
            "arm64-apple-macosx14.0",
            "swift/apple_speech.swift",
            "-o",
        ])
        .arg(&lib_path)
        .status()
        .expect("failed to run swiftc; the apple-speech feature needs Xcode command line tools");
    assert!(
        status.success(),
        "swiftc failed for swift/apple_speech.swift"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=glimpse_apple_speech");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=CoreMedia");
    // The macOS 26 analyzer classes live in Speech.framework, which predates
    // them; the framework link is safe on macOS 14, and the new symbols are
    // weak because the shim's deployment target is 14 with #available guards.
    println!("cargo:rustc-link-lib=framework=Speech");
    println!("cargo:rustc-link-search=native=/usr/lib/swift");
}
