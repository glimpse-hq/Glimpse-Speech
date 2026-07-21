fn main() {
    // fm-rs links a Swift shim; binaries need the Swift runtime rpath.
    // Consumers of the lib (the Glimpse app) must add the same link arg.
    if std::env::var_os("CARGO_FEATURE_CLEANUP_APPLE").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64")
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        // Weak-link so binaries still launch on macOS < 26, where the
        // framework is absent. cleanup.rs guards every call by OS version.
        println!("cargo:rustc-link-arg=-Wl,-weak_framework,FoundationModels");
    }
}
