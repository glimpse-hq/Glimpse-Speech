fn main() {
    // fm-rs links a Swift shim; binaries need the Swift runtime rpath.
    // Consumers of the lib (the Glimpse app) must add the same link arg.
    if std::env::var_os("CARGO_FEATURE_CLEANUP_APPLE").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
