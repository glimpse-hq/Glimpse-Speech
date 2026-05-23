use std::{
    path::{Path, PathBuf},
    process::Command,
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

    let samples: Result<Vec<f32>, _> = reader
        .samples::<i16>()
        .map(|sample| sample.map(|s| s as f32 / PCM16_SCALE))
        .collect();

    Ok(samples?)
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
            let converted = temp_wav_path();
            let status = Command::new(&ffmpeg)
                .arg("-y")
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
                .arg(&converted)
                .status()
                .map_err(|err| io_error(format!("Failed to run ffmpeg: {err}")))?;

            if !status.success() {
                let _ = std::fs::remove_file(&converted);
                return Err(io_error(format!(
                    "ffmpeg failed to decode {}",
                    path.display()
                )));
            }

            let result = read_wav_samples(&converted);
            let _ = std::fs::remove_file(converted);
            result
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

fn temp_wav_path() -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(
        "glimpse-speech-decode-{}-{timestamp}.wav",
        std::process::id()
    ))
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}
