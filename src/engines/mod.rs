#[cfg(all(
    feature = "parakeet",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
pub mod parakeet;
#[cfg(feature = "whisper")]
pub mod whisper;
