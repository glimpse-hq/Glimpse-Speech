use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use glimpse_speech::audio::read_wav_samples;

static NEXT_TEMP_WAV_ID: AtomicU64 = AtomicU64::new(0);

#[test]
fn reads_pcm16_mono_16khz_wav() {
    let path = write_temp_wav(16_000, &[0, 1000, -1000, 250]);
    let samples = read_wav_samples(&path).expect("wav should load");
    let _ = std::fs::remove_file(path);

    assert_eq!(samples.len(), 4);
    assert!(samples[1] > 0.0);
    assert!(samples[2] < 0.0);
}

#[test]
fn rejects_non_16khz_wav() {
    let path = write_temp_wav(8_000, &[0, 100, -100, 50]);
    let error = read_wav_samples(&path).expect_err("8kHz input must fail");
    let _ = std::fs::remove_file(path);

    assert!(error.to_string().contains("16000"));
}

fn write_temp_wav(sample_rate: u32, samples: &[i16]) -> PathBuf {
    let nonce = NEXT_TEMP_WAV_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "glimpse-speech-test-{}-{nonce}.wav",
        std::process::id()
    ));

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&path, spec).expect("wav file should be created");
    for sample in samples {
        writer
            .write_sample(*sample)
            .expect("sample should be written");
    }
    writer.finalize().expect("wav should be finalized");

    path
}
