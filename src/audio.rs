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
        samples.push(f32::from(sample?) / PCM16_SCALE);
    }
    Ok(samples)
}

pub fn read_audio_samples(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let wav_error = match read_pcm16_wav(path) {
        Ok(samples) => return Ok(samples),
        Err(error) => error,
    };

    let ffmpeg = find_ffmpeg().ok_or_else(|| {
        io_error(format!(
            "Audio must be a PCM int16 WAV, or ffmpeg must be installed to decode {}: {wav_error}",
            path.display()
        ))
    })?;
    let converted = temp_wav_path()?;
    let status = Command::new(&ffmpeg)
        .args(["-n", "-nostdin", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-ar", "16000", "-ac", "1", "-sample_fmt", "s16"])
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

fn read_pcm16_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.bits_per_sample != 16 || spec.sample_format != hound::SampleFormat::Int {
        return Err(format!(
            "Expected PCM int16 samples, found {} bit {:?}",
            spec.bits_per_sample, spec.sample_format
        )
        .into());
    }
    if spec.channels == 0 {
        return Err("WAV has no channels".into());
    }

    let samples = reader
        .into_samples::<i16>()
        .collect::<Result<Vec<_>, _>>()?;
    let mono = downmix(samples, usize::from(spec.channels));
    Ok(resample_i16_to_f32(&mono, spec.sample_rate, 16_000))
}

fn downmix(samples: Vec<i16>, channels: usize) -> Vec<i16> {
    if channels == 1 {
        return samples;
    }
    samples
        .chunks_exact(channels)
        .map(|frame| (frame.iter().map(|&s| i32::from(s)).sum::<i32>() / channels as i32) as i16)
        .collect()
}

fn find_ffmpeg() -> Option<PathBuf> {
    let binary = if cfg!(target_os = "windows") {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };

    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|path| path.join(binary))
        .find(|candidate| candidate.is_file())
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

/// Converts PCM16 to normalized f32 and linearly resamples to `to_rate` in a
/// single pass. Equal or zero rates only scale.
pub(crate) fn resample_i16_to_f32(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<f32> {
    const SCALE: f32 = 1.0 / PCM16_SCALE;

    let scaled = |sample: i16| f32::from(sample) * SCALE;

    if samples.is_empty() {
        return Vec::new();
    }
    if from_rate == 0 || to_rate == 0 || from_rate == to_rate {
        return samples.iter().copied().map(scaled).collect();
    }

    let step = f64::from(from_rate) / f64::from(to_rate);
    let target_len = (samples.len() as f64 / step).ceil().max(1.0) as usize;
    let last_index = samples.len() - 1;

    (0..target_len)
        .map(|idx| {
            let src_pos = idx as f64 * step;
            let base = src_pos as usize;
            if base >= last_index {
                return scaled(samples[last_index]);
            }
            let frac = (src_pos - base as f64) as f32;
            let current = scaled(samples[base]);
            let next = scaled(samples[base + 1]);
            current + (next - current) * frac
        })
        .collect()
}

#[cfg(test)]
mod resample_tests {
    use super::{downmix, resample_i16_to_f32};

    const SCALE: f32 = 1.0 / super::PCM16_SCALE;

    #[test]
    fn passthrough_when_rate_unchanged() {
        let input = [0i16, 16_384, -16_384, 32_767];
        let out = resample_i16_to_f32(&input, 16_000, 16_000);
        let expected: Vec<f32> = input.iter().map(|&s| f32::from(s) * SCALE).collect();
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

    #[test]
    fn downmix_averages_interleaved_frames() {
        assert_eq!(
            downmix(vec![100, 300, -32_768, -32_768, 7, 8], 2),
            vec![200, -32_768, 7]
        );
        assert_eq!(downmix(vec![1, 2, 3], 1), vec![1, 2, 3]);
    }
}
