use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let macos = target_os == "macos";
    let macos_arm = macos && target_arch == "aarch64";
    let intel_mac = macos && target_arch == "x86_64";

    let whisper = feature("WHISPER");
    let nvidia = feature("NVIDIA") && !intel_mac;
    let apple_speech = macos_arm && feature("APPLE_SPEECH");
    let apple_cleanup = macos_arm && feature("CLEANUP_APPLE");
    let transcribe = feature("TRANSCRIBE");

    // Named cfgs so source files can gate on capabilities instead of
    // repeating feature and target predicates.
    emit_cfg("onnx_runtime", !intel_mac);
    emit_cfg("nvidia_engines", nvidia);
    emit_cfg("apple_speech_engine", apple_speech);
    emit_cfg("apple_cleanup", apple_cleanup);
    emit_cfg("transcribe_engine", transcribe);
    emit_cfg(
        "local_engines",
        whisper || nvidia || apple_speech || transcribe,
    );
    emit_cfg("streaming_engines", nvidia || apple_speech);

    if apple_cleanup || apple_speech {
        println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        // Weak-link so binaries still launch on macOS < 26, where the
        // framework is absent. cleanup.rs guards every call by OS version.
        println!("cargo::rustc-link-arg=-Wl,-weak_framework,FoundationModels");
    }

    if apple_speech {
        compile_apple_speech_shim();
    }
}

fn feature(name: &str) -> bool {
    std::env::var_os(format!("CARGO_FEATURE_{name}")).is_some()
}

fn emit_cfg(name: &str, enabled: bool) {
    println!("cargo::rustc-check-cfg=cfg({name})");
    if enabled {
        println!("cargo::rustc-cfg={name}");
    }
}

fn compile_apple_speech_shim() {
    println!("cargo::rerun-if-changed=swift/apple_speech.swift");
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

    println!("cargo::rustc-link-search=native={}", out_dir.display());
    println!("cargo::rustc-link-lib=static=glimpse_apple_speech");
    println!("cargo::rustc-link-lib=framework=Foundation");
    println!("cargo::rustc-link-lib=framework=AVFoundation");
    println!("cargo::rustc-link-lib=framework=CoreMedia");
    // The macOS 26 analyzer classes live in Speech.framework, which predates
    // them; the framework link is safe on macOS 14, and the new symbols are
    // weak because the shim's deployment target is 14 with #available guards.
    println!("cargo::rustc-link-lib=framework=Speech");
    println!("cargo::rustc-link-search=native=/usr/lib/swift");
}
