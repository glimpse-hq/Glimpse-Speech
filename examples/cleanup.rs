use glimpse_speech::cleanup::CleanupProvider;
use glimpse_speech::Transcription;

#[tokio::main]
async fn main() {
    println!(
        "apple availability: {:?}",
        CleanupProvider::apple_availability()
    );

    let raw = "um so i was thinking uh maybe we should we should move the the launch to thursday \
               because uh the build isnt ready and and jen said the uh the release notes arent done either";
    let transcription = Transcription {
        text: raw.to_string(),
        segments: None,
        words: None,
        model_id: "example".to_string(),
        language: Some("en".to_string()),
        duration_ms: 0,
    };

    let cleaned = CleanupProvider::apple().apply(transcription).await;
    println!("before: {raw}");
    println!("after:  {}", cleaned.text);
}
