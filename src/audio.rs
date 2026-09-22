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
    if spec.sample_rate == 0 {
        return Err("WAV has a zero sample rate".into());
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

/// Converts PCM16 to normalized f32 and resamples to `to_rate`, low-pass
/// filtering when downsampling so content above the new Nyquist frequency
/// cannot alias. Equal or zero rates only scale.
pub(crate) fn resample_i16_to_f32(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<f32> {
    const SCALE: f32 = 1.0 / PCM16_SCALE;

    let scaled: Vec<f32> = samples
        .iter()
        .map(|&sample| f32::from(sample) * SCALE)
        .collect();
    if scaled.is_empty() || from_rate == 0 || to_rate == 0 || from_rate == to_rate {
        return scaled;
    }

    let step = f64::from(from_rate) / f64::from(to_rate);
    let target_len = (scaled.len() as f64 / step).ceil().max(1.0) as usize;
    if from_rate > to_rate {
        let filter = PolyphaseFilter::new(f64::from(to_rate) / f64::from(from_rate));
        return (0..target_len)
            .map(|idx| filter.sample(&scaled, idx as f64 * step))
            .collect();
    }

    let last_index = scaled.len() - 1;
    (0..target_len)
        .map(|idx| {
            let src_pos = idx as f64 * step;
            let base = src_pos as usize;
            if base >= last_index {
                return scaled[last_index];
            }
            let frac = (src_pos - base as f64) as f32;
            scaled[base] + (scaled[base + 1] - scaled[base]) * frac
        })
        .collect()
}

struct PolyphaseFilter {
    half: usize,
    width: usize,
    phases: Vec<f32>,
}

impl PolyphaseFilter {
    const PHASES: usize = 128;
    const ZERO_CROSSINGS: f64 = 12.0;
    const ROLLOFF: f64 = 0.9;

    fn new(bandwidth: f64) -> Self {
        let cutoff = bandwidth * Self::ROLLOFF;
        let half = (Self::ZERO_CROSSINGS / cutoff).ceil() as usize;
        let width = (2 * half + 1).next_multiple_of(8);
        let mut phases = vec![0.0f32; Self::PHASES * width];
        for (phase, row) in phases.chunks_exact_mut(width).enumerate() {
            let frac = phase as f64 / Self::PHASES as f64;
            let taps: Vec<f64> = (0..=2 * half)
                .map(|k| {
                    let x = k as f64 - half as f64 - frac;
                    if x.abs() >= half as f64 {
                        return 0.0;
                    }
                    let arg = std::f64::consts::PI * cutoff * x;
                    let sinc = if arg == 0.0 { 1.0 } else { arg.sin() / arg };
                    let angle = std::f64::consts::PI * x / half as f64;
                    let window = 0.42 + 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos();
                    sinc * window
                })
                .collect();
            let sum: f64 = taps.iter().sum();
            for (slot, tap) in row.iter_mut().zip(taps) {
                *slot = (tap / sum) as f32;
            }
        }
        Self {
            half,
            width,
            phases,
        }
    }

    fn sample(&self, samples: &[f32], position: f64) -> f32 {
        let base = position as usize;
        let phase =
            (((position - base as f64) * Self::PHASES as f64) as usize).min(Self::PHASES - 1);
        let taps = &self.phases[phase * self.width..(phase + 1) * self.width];
        if base >= self.half && base - self.half + self.width <= samples.len() {
            return dot(
                &samples[base - self.half..base - self.half + self.width],
                taps,
            );
        }
        let last = samples.len() as isize - 1;
        taps.iter()
            .enumerate()
            .map(|(offset, tap)| {
                let index = (base as isize + offset as isize - self.half as isize).clamp(0, last);
                samples[index as usize] * tap
            })
            .sum()
    }
}

fn dot(samples: &[f32], taps: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    for (chunk, tap_chunk) in samples.chunks_exact(8).zip(taps.chunks_exact(8)) {
        for lane in 0..8 {
            lanes[lane] += chunk[lane] * tap_chunk[lane];
        }
    }
    lanes.iter().sum()
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

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    fn tone(frequency: f32, rate: u32, seconds: f32) -> Vec<i16> {
        (0..(rate as f32 * seconds) as usize)
            .map(|n| {
                let t = n as f32 / rate as f32;
                (16_000.0 * (2.0 * std::f32::consts::PI * frequency * t).sin()) as i16
            })
            .collect()
    }

    #[test]
    fn downsampling_rejects_content_above_the_new_nyquist() {
        let out = resample_i16_to_f32(&tone(12_000.0, 48_000, 1.0), 48_000, 16_000);
        let steady = &out[1_000..out.len() - 1_000];
        assert!(rms(steady) < 0.002, "aliased energy {}", rms(steady));
    }

    #[test]
    fn downsampling_keeps_speech_band_content() {
        let out = resample_i16_to_f32(&tone(1_000.0, 44_100, 1.0), 44_100, 16_000);
        let steady = &out[1_000..out.len() - 1_000];
        let expected = 16_000.0 * SCALE / std::f32::consts::SQRT_2;
        assert!(
            (rms(steady) / expected - 1.0).abs() < 0.02,
            "rms {}",
            rms(steady)
        );
    }
}
