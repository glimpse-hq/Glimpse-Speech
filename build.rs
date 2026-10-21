use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let macos = target_os == "macos";
    let macos_arm = macos && target_arch == "aarch64";

    let apple_speech = macos_arm && feature("APPLE_SPEECH");
    let transcribe = feature("TRANSCRIBE");

    // Named cfgs so source files can gate on capabilities instead of
    // repeating feature and target predicates.
    emit_cfg("apple_speech_engine", apple_speech);
    emit_cfg("transcribe_engine", transcribe);
    // `whisper` implies `transcribe`, so it needs no check of its own.
    emit_cfg("local_engines", apple_speech || transcribe);
    emit_cfg("streaming_engines", apple_speech || transcribe);

    if apple_speech {
        println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        compile_swift_shim("apple_speech");
        println!("cargo::rustc-link-lib=framework=AVFoundation");
        println!("cargo::rustc-link-lib=framework=CoreMedia");
        // The macOS 26 analyzer classes live in Speech.framework, which predates
        // them; the framework link is safe on macOS 14, and the new symbols are
        // weak because the shim's deployment target is 14 with #available guards.
        println!("cargo::rustc-link-lib=framework=Speech");
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

/// Builds swift/<name>.swift into a static lib with a macOS 14 deployment
/// target, so newer APIs stay weak behind #available.
fn compile_swift_shim(name: &str) {
    let source = format!("swift/{name}.swift");
    println!("cargo::rerun-if-changed={source}");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let lib_name = format!("glimpse_{name}");
    let lib_path = out_dir.join(format!("lib{lib_name}.a"));

    let status = Command::new("swiftc")
        .args([
            "-O",
            "-parse-as-library",
            "-module-name",
            &lib_name,
            "-emit-library",
            "-static",
            "-target",
            "arm64-apple-macosx14.0",
            &source,
            "-o",
        ])
        .arg(&lib_path)
        .status()
        .expect("failed to run swiftc; the Apple features need Xcode command line tools");
    assert!(status.success(), "swiftc failed for {source}");

    println!("cargo::rustc-link-search=native={}", out_dir.display());
    println!("cargo::rustc-link-lib=static={lib_name}");
    println!("cargo::rustc-link-lib=framework=Foundation");
    println!("cargo::rustc-link-search=native=/usr/lib/swift");
}
