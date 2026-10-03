#[cfg(apple_speech_engine)]
pub mod apple;
#[cfg(transcribe_engine)]
pub mod transcribe;

#[cfg(transcribe_engine)]
pub(crate) fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

/// Inference thread count: one per performance core. CPU inference still
/// scaled at 24 threads on a 24-core Xeon, the most measured.
#[cfg(transcribe_engine)]
pub(crate) fn inference_threads() -> usize {
    const MAX_THREADS: usize = 24;

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    if let Some(cores) = performance_cores() {
        return cores.min(MAX_THREADS);
    }

    std::thread::available_parallelism()
        .map_or(4, std::num::NonZeroUsize::get)
        .min(8)
}

/// Performance-core count on Apple Silicon, physical cores on Intel Macs.
/// Evenly-partitioned parallel ops stall on efficiency cores, so threads
/// beyond the P-core count hurt more than they help.
#[cfg(all(target_os = "macos", transcribe_engine))]
fn performance_cores() -> Option<usize> {
    let count = |name: &std::ffi::CStr| {
        let mut value: libc::c_int = 0;
        let mut size = std::mem::size_of::<libc::c_int>();
        let result = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&raw mut value).cast::<libc::c_void>(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (result == 0 && value > 0).then_some(value as usize)
    };
    count(c"hw.perflevel0.physicalcpu").or_else(|| count(c"hw.physicalcpu"))
}

/// Physical cores of the fastest efficiency class. Hybrid Intel CPUs keep
/// at least the 8 threads earlier versions used, which can include
/// efficiency cores.
#[cfg(all(target_os = "windows", transcribe_engine))]
fn performance_cores() -> Option<usize> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: i32,
            buffer: *mut u8,
            length: *mut u32,
        ) -> i32;
    }
    const RELATION_PROCESSOR_CORE: i32 = 0;
    let mut length = 0u32;
    unsafe {
        GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE,
            std::ptr::null_mut(),
            &raw mut length,
        )
    };
    let mut buffer = vec![0u8; length as usize];
    if unsafe {
        GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE,
            buffer.as_mut_ptr(),
            &raw mut length,
        )
    } == 0
    {
        return None;
    }
    // One record per core: Relationship and Size (u32 each), then a
    // PROCESSOR_RELATIONSHIP whose second byte is the EfficiencyClass.
    let mut classes = Vec::new();
    let mut rest = buffer.get(..length as usize)?;
    while let Some(size) = rest
        .get(4..8)
        .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as usize)
        && size > 9
        && size <= rest.len()
    {
        classes.push(rest[9]);
        rest = &rest[size..];
    }
    let fastest = *classes.iter().max()?;
    let performance = classes.iter().filter(|&&class| class == fastest).count();
    Some(performance.max(classes.len().min(8)))
}
