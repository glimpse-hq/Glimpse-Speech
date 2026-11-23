use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    errors::Error as SymphoniaError,
    formats::{FormatOptions, TrackType, probe::Hint},
    io::MediaSourceStream,
    meta::MetadataOptions,
};

use crate::io_error;

const PCM16_SCALE: f32 = 32_768.0;
pub(crate) const MAX_SAMPLE_RATE: u32 = 384_000;
const FFMPEG_FORMATS: &str = "mov,matroska,ogg,mp3,aac,flac,wav,w64,wv,ape,au,aiff,caf,asf,amr,ac3,eac3,dts,mpeg,mpegts,avi,flv";

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

/// WAV in-process with hound, other formats with Symphonia, and what Symphonia
/// can't decode (such as Opus) with ffmpeg when it is installed.
pub fn read_audio_samples(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if let Ok(samples) = read_wav(path) {
        return Ok(samples);
    }
    let decode_error = match decode(path) {
        Ok(samples) => return Ok(samples),
        Err(error) => error,
    };

    let ffmpeg = find_ffmpeg().ok_or_else(|| {
        io_error(format!(
            "Could not decode {} without ffmpeg: {decode_error}",
            path.display()
        ))
    })?;
    // Only local files and single-file demuxers, so a playlist or concat file
    // can't make ffmpeg read other files or URLs.
    let output = Command::new(&ffmpeg)
        .args([
            "-nostdin",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file",
        ])
        .args(["-format_whitelist", FFMPEG_FORMATS, "-i"])
        .arg(path)
        .args(["-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"])
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|err| io_error(format!("Failed to run ffmpeg: {err}")))?;
    if !output.status.success() || output.stdout.is_empty() {
        return Err(io_error(format!(
            "ffmpeg failed to decode {}: {decode_error}",
            path.display()
        ))
        .into());
    }
    Ok(output
        .stdout
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&bytes| f32::from(i16::from_le_bytes(bytes)) / PCM16_SCALE)
        .collect())
}

fn decode(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let source = MediaSourceStream::new(Box::new(fs::File::open(path)?), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe().probe(
        &hint,
        source,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or("no audio track")?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or("no audio track")?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())?;
    let mut sample_rate = None;
    let mut mono = Vec::new();
    let mut interleaved = Vec::new();
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => return Err(error.into()),
        };
        // HE-AAC decodes without SBR, at half the rate its container states.
        let rate = decoded.spec().rate();
        if *sample_rate.get_or_insert(rate) != rate {
            return Err("sample rate changes mid-stream".into());
        }
        let channels = decoded.spec().channels().count().max(1);
        decoded.copy_to_vec_interleaved(&mut interleaved);
        mono.extend(downmix(std::mem::take(&mut interleaved), channels));
    }
    if mono.is_empty() {
        return Err("no audio decoded".into());
    }
    let sample_rate = sample_rate
        .filter(|rate| (1..=MAX_SAMPLE_RATE).contains(rate))
        .ok_or("unsupported sample rate")?;
    Ok(resample(mono, sample_rate, 16_000))
}

fn read_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err("WAV has no channels".into());
    }
    if spec.sample_rate == 0 || spec.sample_rate > MAX_SAMPLE_RATE {
        return Err(format!("Unsupported WAV sample rate {} Hz", spec.sample_rate).into());
    }

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            reader.into_samples::<f32>().collect::<Result<_, _>>()?
        }
        (hound::SampleFormat::Int, bits @ 1..=32) => {
            let scale = 1.0 / (1u64 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 * scale))
                .collect::<Result<_, _>>()?
        }
        (format, bits) => {
            return Err(format!("Unsupported WAV encoding: {bits} bit {format:?}").into());
        }
    };
    let mono = downmix(samples, usize::from(spec.channels));
    Ok(resample(mono, spec.sample_rate, 16_000))
}

fn downmix(samples: Vec<f32>, channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples;
    }
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
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

/// Converts PCM16 to normalized f32 and resamples to `to_rate`.
#[cfg_attr(not(local_engines), allow(dead_code))]
pub(crate) fn resample_i16_to_f32(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<f32> {
    const SCALE: f32 = 1.0 / PCM16_SCALE;

    let scaled = samples
        .iter()
        .map(|&sample| f32::from(sample) * SCALE)
        .collect();
    resample(scaled, from_rate, to_rate)
}

/// Resamples to `to_rate`, low-pass filtering when downsampling so content
/// above the new Nyquist frequency cannot alias. Equal or zero rates pass
/// through.
fn resample(samples: Vec<f32>, from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() || from_rate == 0 || to_rate == 0 || from_rate == to_rate {
        return samples;
    }

    let step = f64::from(from_rate) / f64::from(to_rate);
    let target_len = (samples.len() as f64 / step).ceil().max(1.0) as usize;
    if from_rate > to_rate {
        let filter = PolyphaseFilter::new(f64::from(to_rate) / f64::from(from_rate));
        return (0..target_len)
            .map(|idx| filter.sample(&samples, idx as f64 * step))
            .collect();
    }

    let last_index = samples.len() - 1;
    (0..target_len)
        .map(|idx| {
            let src_pos = idx as f64 * step;
            let base = src_pos as usize;
            if base >= last_index {
                return samples[last_index];
            }
            let frac = (src_pos - base as f64) as f32;
            samples[base] + (samples[base + 1] - samples[base]) * frac
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
    const MAX_HALF_WIDTH: usize = 320;

    fn new(bandwidth: f64) -> Self {
        let cutoff = bandwidth * Self::ROLLOFF;
        let half = ((Self::ZERO_CROSSINGS / cutoff).ceil() as usize).min(Self::MAX_HALF_WIDTH);
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
    use super::{downmix, read_wav, resample_i16_to_f32};

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
            downmix(vec![0.25, 0.75, -1.0, -1.0, 7.0, 8.0], 2),
            vec![0.5, -1.0, 7.5]
        );
        assert_eq!(downmix(vec![1.0, 2.0, 3.0], 1), vec![1.0, 2.0, 3.0]);
    }

    fn write_wav(
        name: &str,
        spec: hound::WavSpec,
        write: impl FnOnce(&mut hound::WavWriter<std::io::BufWriter<std::fs::File>>),
    ) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("glimpse-speech-{name}-{}.wav", std::process::id()));
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        write(&mut writer);
        writer.finalize().unwrap();
        path
    }

    #[test]
    fn decodes_float_wav_in_process() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let path = write_wav("float", spec, |writer| {
            for _ in 0..100 {
                writer.write_sample(0.5f32).unwrap();
                writer.write_sample(-0.25f32).unwrap();
            }
        });
        let samples = read_wav(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(samples, vec![0.125; 100]);
    }

    #[test]
    fn decodes_24_bit_wav_in_process() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let path = write_wav("int24", spec, |writer| {
            for _ in 0..4_800 {
                writer.write_sample(1i32 << 22).unwrap();
            }
        });
        let samples = read_wav(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(samples.len(), 1_600);
        assert!((samples[800] - 0.5).abs() < 1e-3, "sample {}", samples[800]);
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

    #[test]
    fn extreme_rates_keep_the_filter_bounded() {
        let out = resample_i16_to_f32(&[1_000i16; 4_096], u32::MAX, 16_000);
        assert_eq!(out.len(), 1);
    }
}
