//! Real-model check for the transcribe.cpp engine. Run with --ignored when
//! `GLIMPSE_SPEECH_TRANSCRIBE_MODEL` names a GGUF and
//! `GLIMPSE_SPEECH_TRANSCRIBE_WAV` a 16 kHz mono WAV of English speech.
#![cfg(feature = "transcribe")]

use std::path::PathBuf;

use glimpse_speech::{
    TranscriptionEngine,
    engines::transcribe::{TranscribeEngine, TranscribeInferenceParams, TranscribeModelParams},
};

fn fixture() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var_os("GLIMPSE_SPEECH_TRANSCRIBE_MODEL")?;
    let wav = std::env::var_os("GLIMPSE_SPEECH_TRANSCRIBE_WAV")?;
    Some((PathBuf::from(model), PathBuf::from(wav)))
}

#[test]
#[ignore = "requires GLIMPSE_SPEECH_TRANSCRIBE_MODEL and GLIMPSE_SPEECH_TRANSCRIBE_WAV"]
fn transcribes_english_with_and_without_companion() {
    let (model, wav) = fixture().expect("set both model and WAV fixture paths");
    let companion = TranscribeEngine::companion_for(&model);
    let mut encoders = vec![None];
    if let Some(companion) = companion {
        encoders.push(Some(companion));
    }
    for coreml_encoder in encoders {
        let mut engine = TranscribeEngine::new();
        engine
            .load_model_with_params(
                &model,
                TranscribeModelParams {
                    coreml_encoder,
                    ..Default::default()
                },
            )
            .expect("model loads");
        let result = engine
            .transcribe_file(
                &wav,
                Some(TranscribeInferenceParams {
                    language: Some("en".into()),
                    ..Default::default()
                }),
            )
            .expect("transcribes");
        assert!(!result.text.is_empty());
        assert!(result.segments.is_none(), "no timestamps were requested");
        assert_eq!(result.language.as_deref(), Some("en"));
    }
    // Core ML uses FP16: both paths must work, but exact text parity is not
    // guaranteed. Accuracy comparisons belong in the recorded-audio benchmark.
}

#[test]
fn rejects_missing_model_file() {
    let mut engine = TranscribeEngine::new();
    let err = engine
        .load_model(&PathBuf::from("/definitely/missing.gguf"))
        .expect_err("missing file must fail");
    assert!(err.to_string().contains("not found"));
}

#[test]
#[ignore = "requires Qwen GGUF and WAV fixture paths"]
fn qwen_dictionary_is_applied_per_request_across_chunks() {
    let (model, wav) = fixture().expect("set both model and WAV fixture paths");
    let samples = glimpse_speech::audio::read_audio_samples(&wav).unwrap();
    let mut long = samples.clone();
    while long.len() <= 16_000 * 30 {
        long.extend_from_slice(&samples);
    }
    let mut companions = vec![None];
    if let Some(companion) = TranscribeEngine::companion_for(&model) {
        companions.push(Some(companion));
    }
    for coreml_encoder in companions {
        let mut engine = TranscribeEngine::new();
        engine
            .load_model_with_params(
                &model,
                TranscribeModelParams {
                    coreml_encoder,
                    ..Default::default()
                },
            )
            .unwrap();
        let plain = TranscribeInferenceParams {
            language: Some("en".into()),
            ..Default::default()
        };
        let baseline = engine
            .transcribe_samples(samples.clone(), Some(plain.clone()))
            .unwrap()
            .text;
        let hints = TranscribeInferenceParams {
            dictionary: vec![
                "Kennedy".into(),
                "Glimpse".into(),
                "José".into(),
                "東京".into(),
            ],
            ..plain.clone()
        };
        let result = engine
            .transcribe_samples(long.clone(), Some(hints))
            .unwrap();
        assert!(!result.text.is_empty());
        assert_eq!(
            engine
                .transcribe_samples(samples.clone(), Some(plain))
                .unwrap()
                .text,
            baseline
        );
    }
}
