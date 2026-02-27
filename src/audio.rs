use std::path::Path;

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
        .map(|sample| sample.map(|s| s as f32 / i16::MAX as f32))
        .collect();

    Ok(samples?)
}
