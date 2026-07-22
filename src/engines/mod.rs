#[cfg(all(feature = "apple-speech", target_os = "macos", target_arch = "aarch64"))]
pub mod apple;
#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
pub mod nemotron;
#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
pub mod parakeet;
#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(any(
    feature = "whisper",
    all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    )
))]
pub(crate) fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
pub(crate) fn validate_model_dir(
    model_path: &std::path::Path,
    engine: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !model_path.exists() {
        return Err(io_error(format!(
            "{engine} model directory not found: {}",
            model_path.display()
        )));
    }
    if !model_path.is_dir() {
        return Err(io_error(format!(
            "{engine} model path must be a directory: {}",
            model_path.display()
        )));
    }
    Ok(())
}

/// Inference thread count: physical parallelism, capped where extra threads
/// stop paying for themselves on hybrid-core CPUs.
#[cfg(any(
    feature = "whisper",
    all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    )
))]
pub(crate) fn inference_threads() -> usize {
    #[cfg(target_os = "macos")]
    if let Some(performance_cores) = macos_performance_cores() {
        return performance_cores.min(8);
    }

    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8)
}

/// Performance-core count on hybrid Apple Silicon. Evenly-partitioned
/// parallel ops stall on efficiency cores, so threads beyond the P-core
/// count hurt more than they help. Absent on Intel Macs (falls back).
#[cfg(all(
    target_os = "macos",
    any(
        feature = "whisper",
        all(
            feature = "nvidia",
            not(all(target_os = "macos", target_arch = "x86_64"))
        )
    )
))]
fn macos_performance_cores() -> Option<usize> {
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            &mut value as *mut libc::c_int as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && value > 0).then_some(value as usize)
}

#[cfg(all(
    test,
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
mod tests {
    use super::validate_model_dir;

    #[test]
    fn validate_model_dir_rejects_missing_and_file_paths() {
        let dir = std::env::temp_dir().join(format!(
            "glimpse-speech-validate-dir-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let err = validate_model_dir(&dir, "Parakeet").unwrap_err();
        assert!(err.to_string().contains("Parakeet"));
        assert!(err.to_string().contains("not found"));

        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.bin");
        std::fs::write(&file, b"x").unwrap();
        let err = validate_model_dir(&file, "Nemotron").unwrap_err();
        assert!(err.to_string().contains("Nemotron"));
        assert!(err.to_string().contains("must be a directory"));

        assert!(validate_model_dir(&dir, "Parakeet").is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
