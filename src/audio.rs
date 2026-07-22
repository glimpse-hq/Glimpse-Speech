use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const PCM16_SCALE: f32 = 32_768.0;

/// Requirements: 16 kHz, mono, PCM int16 WAV file.
pub fn read_wav_samples(wav_path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(format!("Expected 1 channel, found {}", spec.channels).into());
    }

    if spec.sample_rate != 16_000 {
        return Err(format!(
            "Expected 16000 Hz sample rate, found {} Hz",
            spec.sample_rate
        )
        .into());
    }

    if spec.bits_per_sample != 16 {
        return Err(format!(
            "Expected 16 bits per sample, found {}",
            spec.bits_per_sample
        )
        .into());
    }

    if spec.sample_format != hound::SampleFormat::Int {
        return Err(format!("Expected Int sample format, found {:?}", spec.sample_format).into());
    }

    let mut samples = Vec::with_capacity(reader.len() as usize);
    for sample in reader.samples::<i16>() {
        samples.push(sample? as f32 / PCM16_SCALE);
    }

    Ok(samples)
}

pub fn read_audio_samples(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match read_wav_samples(path) {
        Ok(samples) => Ok(samples),
        Err(wav_error) => {
            let ffmpeg = find_ffmpeg().ok_or_else(|| {
                io_error(format!(
                    "Audio must be a 16 kHz mono PCM WAV, or ffmpeg must be installed to decode {}: {wav_error}",
                    path.display()
                ))
            })?;
            let converted = temp_wav_path()?;
            let status = Command::new(&ffmpeg)
                .arg("-n")
                .arg("-nostdin")
                .arg("-loglevel")
                .arg("error")
                .arg("-i")
                .arg(path)
                .arg("-ar")
                .arg("16000")
                .arg("-ac")
                .arg("1")
                .arg("-sample_fmt")
                .arg("s16")
                .arg(&converted.path)
                .status()
                .map_err(|err| io_error(format!("Failed to run ffmpeg: {err}")))?;

            if !status.success() {
                return Err(io_error(format!(
                    "ffmpeg failed to decode {}",
                    path.display()
                )));
            }

            read_wav_samples(&converted.path)
        }
    }
}

fn find_ffmpeg() -> Option<PathBuf> {
    let binary = if cfg!(target_os = "windows") {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };

    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|path| path.join(binary))
            .find(|candidate| candidate.is_file())
    })
}

static TEMP_DECODE_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempWav {
    dir: PathBuf,
    path: PathBuf,
}

impl Drop for TempWav {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        let _ = fs::remove_dir(&self.dir);
    }
}

fn temp_wav_path() -> Result<TempWav, Box<dyn std::error::Error>> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    for _ in 0..16 {
        let sequence = TEMP_DECODE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "glimpse-speech-decode-{}-{timestamp}-{sequence}",
            std::process::id(),
        ));
        match fs::create_dir(&dir) {
            Ok(()) => {
                return Ok(TempWav {
                    path: dir.join("audio.wav"),
                    dir,
                });
            }
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(io_error(format!(
                    "Failed to create temp decode directory {}: {err}",
                    dir.display()
                )));
            }
        }
    }

    Err(io_error("Failed to create a unique temp decode directory"))
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    io::Error::other(message.into()).into()
}

#[cfg(any(
    feature = "whisper",
    all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    ),
    all(feature = "apple-speech", target_os = "macos", target_arch = "aarch64")
))]
pub(crate) fn resample_i16_to_f32(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<f32> {
    const SCALE: f32 = 1.0 / PCM16_SCALE;

    if samples.is_empty() {
        return Vec::new();
    }
    if from_rate == 0 || to_rate == 0 || from_rate == to_rate {
        return samples.iter().map(|&s| s as f32 * SCALE).collect();
    }

    let step = from_rate as f64 / to_rate as f64;
    let target_len = ((samples.len() as f64 / step).ceil().max(1.0)) as usize;
    let last_index = samples.len() - 1;
    let mut output = Vec::with_capacity(target_len);

    for idx in 0..target_len {
        let src_pos = idx as f64 * step;
        let base = src_pos as usize;
        if base >= last_index {
            output.push(samples[last_index] as f32 * SCALE);
        } else {
            let frac = (src_pos - base as f64) as f32;
            let current = samples[base] as f32 * SCALE;
            let next = samples[base + 1] as f32 * SCALE;
            output.push(current + (next - current) * frac);
        }
    }

    output
}

#[cfg(all(
    test,
    any(
        feature = "whisper",
        all(
            feature = "nvidia",
            not(all(target_os = "macos", target_arch = "x86_64"))
        )
    )
))]
mod resample_tests {
    use super::resample_i16_to_f32;

    const SCALE: f32 = 1.0 / super::PCM16_SCALE;

    #[test]
    fn passthrough_when_rate_unchanged() {
        let input = [0i16, 16_384, -16_384, 32_767];
        let out = resample_i16_to_f32(&input, 16_000, 16_000);
        let expected: Vec<f32> = input.iter().map(|&s| s as f32 * SCALE).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn zero_rate_does_not_panic_or_overflow() {
        let input = [1i16, 2, 3, 4];
        assert_eq!(resample_i16_to_f32(&input, 0, 16_000).len(), input.len());
        assert_eq!(resample_i16_to_f32(&input, 48_000, 0).len(), input.len());
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(resample_i16_to_f32(&[], 48_000, 16_000).is_empty());
    }

    #[test]
    fn upsampling_interpolates_between_samples() {
        let input = [0i16, 1000];
        let out = resample_i16_to_f32(&input, 8_000, 16_000);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 500.0 * SCALE).abs() < 1e-6);
        assert!((out[2] - 1000.0 * SCALE).abs() < 1e-6);
        assert!((out[3] - 1000.0 * SCALE).abs() < 1e-6);
    }
}
