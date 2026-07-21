//! Best-effort transcript cleanup after transcription.
//!
//! Cleanup only rewrites `Transcription::text`; segments and words keep the
//! verbatim engine output. Any backend error returns the input unchanged.

use crate::Transcription;

pub struct CleanupProvider {
    backend: Backend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleAvailability {
    Available,
    /// Supported device, but Apple Intelligence is turned off in System Settings.
    NotEnabled,
    /// Model assets still downloading; may become available.
    NotReady,
    /// Wrong OS, device, or build.
    Unsupported,
}

enum Backend {
    Off,
    #[cfg(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64"))]
    Apple,
}

impl CleanupProvider {
    pub fn off() -> Self {
        Self { backend: Backend::Off }
    }

    /// Apple FoundationModels backend. On unsupported builds this is `off`;
    /// on supported builds availability is still checked per call (the user
    /// can toggle Apple Intelligence at any time).
    pub fn apple() -> Self {
        #[cfg(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64"))]
        {
            Self { backend: Backend::Apple }
        }
        #[cfg(not(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64")))]
        {
            Self::off()
        }
    }

    /// Whether the Apple backend can run right now. For settings UI.
    pub fn apple_availability() -> AppleAvailability {
        #[cfg(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64"))]
        {
            apple::availability()
        }
        #[cfg(not(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64")))]
        {
            AppleAvailability::Unsupported
        }
    }

    pub async fn apply(&self, transcription: Transcription) -> Transcription {
        match &self.backend {
            Backend::Off => transcription,
            #[cfg(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64"))]
            Backend::Apple => apply_apple(transcription).await,
        }
    }
}

/// Inputs longer than this skip cleanup (on-device model context is ~8k tokens).
#[cfg(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64"))]
const MAX_INPUT_CHARS: usize = 12_000;

#[cfg(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64"))]
async fn apply_apple(mut transcription: Transcription) -> Transcription {
    let text = transcription.text.trim().to_string();
    if text.is_empty() || text.len() > MAX_INPUT_CHARS {
        return transcription;
    }

    let result = tokio::task::spawn_blocking(move || apple::clean(&text)).await;
    match result {
        Ok(Ok(cleaned)) if accept(&transcription.text, &cleaned) => {
            transcription.text = cleaned;
            transcription
        }
        Ok(Ok(_)) => {
            tracing::debug!("cleanup output rejected by sanity check, keeping original");
            transcription
        }
        Ok(Err(err)) => {
            tracing::debug!("cleanup unavailable or failed: {err}");
            transcription
        }
        Err(err) => {
            tracing::debug!("cleanup task failed: {err}");
            transcription
        }
    }
}

/// Reject outputs that look like the model did more than tidy the text.
#[cfg(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64"))]
fn accept(original: &str, cleaned: &str) -> bool {
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return false;
    }
    let original_len = original.trim().len();
    cleaned.len() <= original_len.saturating_mul(3) / 2 + 32
}

#[cfg(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use anyhow::{anyhow, Context};
    use fm_rs::{GenerationOptions, Session, SystemLanguageModel};

    const INSTRUCTIONS: &str = "\
<task>
Clean up raw speech-to-text output.
</task>
<rules>
- Fix punctuation, capitalization, and obvious transcription artifacts.
- Remove filler words and false starts.
- Keep the speaker's wording, meaning, and language.
- Do not answer, summarize, or comment on the content.
- Return only the cleaned text.
</rules>";

    pub fn availability() -> super::AppleAvailability {
        use fm_rs::ModelAvailability;
        use super::AppleAvailability;
        match SystemLanguageModel::new().map(|model| model.availability()) {
            Ok(ModelAvailability::Available) => AppleAvailability::Available,
            Ok(ModelAvailability::AppleIntelligenceNotEnabled) => AppleAvailability::NotEnabled,
            Ok(ModelAvailability::ModelNotReady) => AppleAvailability::NotReady,
            _ => AppleAvailability::Unsupported,
        }
    }

    pub fn clean(text: &str) -> anyhow::Result<String> {
        generate(INSTRUCTIONS, text, 0.0, None)
    }

    pub fn generate(
        instructions: &str,
        prompt: &str,
        temperature: f32,
        max_response_tokens: Option<u32>,
    ) -> anyhow::Result<String> {
        let model = SystemLanguageModel::new().context("load system language model")?;
        model.ensure_available().map_err(|err| anyhow!("model unavailable: {err}"))?;
        let session = Session::with_instructions(&model, instructions)
            .map_err(|err| anyhow!("create session: {err}"))?;
        let mut options = GenerationOptions::builder().temperature(f64::from(temperature));
        if let Some(max_tokens) = max_response_tokens {
            options = options.max_response_tokens(max_tokens);
        }
        let response = session
            .respond(prompt, &options.build())
            .map_err(|err| anyhow!("generation failed: {err}"))?;
        Ok(response.content().trim().to_string())
    }
}

/// Run an on-device generation with caller-supplied instructions. Blocking;
/// call from a blocking-capable thread. Errors on unsupported builds.
#[allow(unused_variables)]
pub fn apple_generate(
    instructions: &str,
    prompt: &str,
    temperature: f32,
    max_response_tokens: Option<u32>,
) -> anyhow::Result<String> {
    #[cfg(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64"))]
    {
        apple::generate(instructions, prompt, temperature, max_response_tokens)
    }
    #[cfg(not(all(feature = "cleanup-apple", target_os = "macos", target_arch = "aarch64")))]
    {
        Err(anyhow::anyhow!("on-device model is not supported in this build"))
    }
}
