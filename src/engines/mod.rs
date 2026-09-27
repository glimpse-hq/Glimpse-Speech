#[cfg(apple_speech_engine)]
pub mod apple;
#[cfg(transcribe_engine)]
pub mod transcribe;

#[cfg(transcribe_engine)]
pub(crate) fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

/// Inference thread count: physical parallelism, capped where extra threads
/// stop paying for themselves on hybrid-core CPUs.
#[cfg(transcribe_engine)]
pub(crate) fn inference_threads() -> usize {
    const MAX_THREADS: usize = 8;

    #[cfg(target_os = "macos")]
    if let Some(performance_cores) = macos_performance_cores() {
        return performance_cores.min(MAX_THREADS);
    }

    std::thread::available_parallelism()
        .map_or(4, std::num::NonZeroUsize::get)
        .min(MAX_THREADS)
}

/// Performance-core count on hybrid Apple Silicon. Evenly-partitioned
/// parallel ops stall on efficiency cores, so threads beyond the P-core
/// count hurt more than they help. Absent on Intel Macs (falls back).
#[cfg(all(target_os = "macos", transcribe_engine))]
fn macos_performance_cores() -> Option<usize> {
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&raw mut value).cast::<libc::c_void>(),
            &raw mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && value > 0).then_some(value as usize)
}
