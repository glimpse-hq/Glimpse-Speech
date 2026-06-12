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
