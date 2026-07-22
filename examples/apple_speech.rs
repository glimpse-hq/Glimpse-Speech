use glimpse_speech::engines::apple;
use glimpse_speech::models::ModelEngine;
use glimpse_speech::service::{AudioInput, SpeechService, TranscribeRequest};

fn main() {
    println!("available: {}", apple::available());
    println!("locale status (default): {}", apple::locale_status(None));

    let Some(wav_path) = std::env::args().nth(1) else {
        println!("usage: apple_speech <wav-path> [language]");
        return;
    };
    let language = std::env::args().skip_while(|arg| arg != "--lang").nth(1);
    let dictation = std::env::args().any(|arg| arg == "--dictation");
    let dictionary: Vec<String> = std::env::args()
        .skip_while(|arg| arg != "--vocab")
        .nth(1)
        .map(|terms| terms.split(',').map(str::to_string).collect())
        .unwrap_or_default();

    let service = SpeechService::new_loose_with_engine(std::env::temp_dir(), ModelEngine::Apple);

    if std::env::args().any(|arg| arg == "--stream") {
        let mut reader = hound::WavReader::open(&wav_path).expect("open wav");
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / 32_768.0)
            .collect();
        for (index, chunk) in samples.chunks(8_960).enumerate() {
            let snapshot = service
                .streaming_transcribe_chunk("apple", chunk)
                .expect("stream chunk failed");
            if index % 4 == 0 {
                println!("  [chunk {index:3}] {snapshot}");
            }
            std::thread::sleep(std::time::Duration::from_millis(560));
        }
        println!("final: {}", service.streaming_finalize());
        service.streaming_reset();
        return;
    }

    let started = std::time::Instant::now();
    let transcription = service
        .transcribe(TranscribeRequest {
            audio: AudioInput::WavPath(wav_path.into()),
            model_id: "apple".to_string(),
            language,
            prompt: None,
            dictionary,
            timestamps: !dictation,
            timestamp_granularity: None,
        })
        .expect("transcription failed");
    println!(
        "transcribed {} ms of audio in {} ms",
        transcription.duration_ms,
        started.elapsed().as_millis()
    );
    println!("text: {}", transcription.text);
    for segment in transcription.segments.unwrap_or_default() {
        println!(
            "  [{:7.2} - {:7.2}] {}",
            segment.start, segment.end, segment.text
        );
    }
}
