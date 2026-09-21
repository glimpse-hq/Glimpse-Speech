use reqwest::multipart::{Form, Part};

use crate::TimestampGranularity;
use crate::remote::ResponseFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointProfile {
    pub timestamp_mode: TimestampMode,
    pub dictionary_mode: DictionaryMode,
    pub sends_response_format: bool,
    pub sends_temperature: bool,
    pub duration_source: DurationSource,
    pub supports_word_timestamps: bool,
    pub keep_timestamps_on_format_fallback: bool,
    pub uploads_flac: bool,
    pub audio_request: AudioRequest,
    pub models_query: &'static str,
    pub diarization: Diarization,
    pub transcriptions_path: &'static str,
    /// Sends `format=true` for written-form numbers. Needs a language.
    pub sends_itn_format: bool,
    pub auth: AuthScheme,
    pub model_field: &'static str,
    pub language_field: &'static str,
    /// Options sent on every transcription request.
    pub fixed_params: &'static [(&'static str, &'static str)],
}

impl EndpointProfile {
    pub fn supports_diarization(&self, model: &str) -> bool {
        match self.diarization {
            Diarization::None => false,
            Diarization::SegmentSpeakers | Diarization::WordSpeakers => true,
            Diarization::DiarizedJson => model.to_ascii_lowercase().contains("diarize"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diarization {
    None,
    /// `diarize=true`, speakers come back on segments.
    SegmentSpeakers,
    /// `diarize=true`, speakers come back on words only.
    WordSpeakers,
    /// OpenAI: only `*-diarize` models, via `response_format=diarized_json`.
    DiarizedJson,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioRequest {
    Multipart,
    Base64Json,
    /// Audio bytes as the body, options as query parameters.
    RawBody,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthScheme {
    Bearer,
    Token,
    XiApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampMode {
    OpenAiVerboseJson,
    NativeGranularities,
    /// A single on/off option; times come back on words.
    WordTimings,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictionaryMode {
    Prompt,
    ContextBias,
    KeyTerms(KeyTermRules),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyTermRules {
    pub field: &'static str,
    /// Only models starting with this accept key terms.
    pub model_prefix: &'static str,
    pub max_chars: usize,
    pub max_words: usize,
    pub max_total_chars: usize,
    pub rejected_chars: &'static [char],
}

impl KeyTermRules {
    fn terms(&self, model: &str, dictionary: &[String]) -> Vec<String> {
        if !model.to_ascii_lowercase().starts_with(self.model_prefix) {
            return Vec::new();
        }
        let mut total = 0;
        crate::dictionary::sanitize_dictionary_entries(dictionary)
            .into_iter()
            .filter(|term| {
                term.chars().count() <= self.max_chars
                    && term.split_whitespace().count() <= self.max_words
                    && !term.contains(self.rejected_chars)
            })
            .take_while(|term| {
                total += term.chars().count();
                total <= self.max_total_chars
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationSource {
    TopLevel,
    UsagePromptAudioSeconds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compatibility {
    DirectOpenAi,
    SelfHosted,
}

impl Compatibility {
    const fn base_profile(self) -> EndpointProfile {
        match self {
            Self::DirectOpenAi => EndpointProfile {
                timestamp_mode: TimestampMode::OpenAiVerboseJson,
                dictionary_mode: DictionaryMode::Prompt,
                sends_response_format: true,
                sends_temperature: true,
                duration_source: DurationSource::TopLevel,
                supports_word_timestamps: true,
                keep_timestamps_on_format_fallback: false,
                uploads_flac: true,
                audio_request: AudioRequest::Multipart,
                models_query: "",
                diarization: Diarization::None,
                transcriptions_path: OPENAI_TRANSCRIPTIONS_PATH,
                sends_itn_format: false,
                auth: AuthScheme::Bearer,
                model_field: "model",
                language_field: "language",
                fixed_params: &[],
            },
            Self::SelfHosted => EndpointProfile {
                timestamp_mode: TimestampMode::OpenAiVerboseJson,
                dictionary_mode: DictionaryMode::Prompt,
                sends_response_format: true,
                sends_temperature: true,
                duration_source: DurationSource::TopLevel,
                supports_word_timestamps: false,
                keep_timestamps_on_format_fallback: true,
                uploads_flac: false,
                audio_request: AudioRequest::Multipart,
                models_query: "",
                diarization: Diarization::None,
                transcriptions_path: OPENAI_TRANSCRIPTIONS_PATH,
                sends_itn_format: false,
                auth: AuthScheme::Bearer,
                model_field: "model",
                language_field: "language",
                fixed_params: &[],
            },
        }
    }
}

const OPENAI_TRANSCRIPTIONS_PATH: &str = "/audio/transcriptions";

struct HostProfile {
    host_suffixes: &'static [&'static str],
    profile: EndpointProfile,
}

const MISTRAL: EndpointProfile = EndpointProfile {
    timestamp_mode: TimestampMode::NativeGranularities,
    dictionary_mode: DictionaryMode::ContextBias,
    sends_response_format: false,
    sends_temperature: false,
    duration_source: DurationSource::UsagePromptAudioSeconds,
    supports_word_timestamps: false,
    keep_timestamps_on_format_fallback: true,
    uploads_flac: true,
    audio_request: AudioRequest::Multipart,
    models_query: "",
    diarization: Diarization::SegmentSpeakers,
    transcriptions_path: OPENAI_TRANSCRIPTIONS_PATH,
    sends_itn_format: false,
    auth: AuthScheme::Bearer,
    model_field: "model",
    language_field: "language",
    fixed_params: &[],
};

const OPENROUTER: EndpointProfile = EndpointProfile {
    timestamp_mode: TimestampMode::None,
    dictionary_mode: DictionaryMode::Prompt,
    sends_response_format: false,
    sends_temperature: false,
    duration_source: DurationSource::TopLevel,
    supports_word_timestamps: false,
    keep_timestamps_on_format_fallback: false,
    uploads_flac: false,
    audio_request: AudioRequest::Base64Json,
    models_query: "?output_modalities=transcription",
    diarization: Diarization::None,
    transcriptions_path: OPENAI_TRANSCRIPTIONS_PATH,
    sends_itn_format: false,
    auth: AuthScheme::Bearer,
    model_field: "model",
    language_field: "language",
    fixed_params: &[],
};

// Words always come back, so there is nothing to request.
const XAI: EndpointProfile = EndpointProfile {
    timestamp_mode: TimestampMode::None,
    dictionary_mode: DictionaryMode::KeyTerms(KeyTermRules {
        field: "keyterm",
        model_prefix: "",
        max_chars: 50,
        max_words: usize::MAX,
        max_total_chars: usize::MAX,
        rejected_chars: &[],
    }),
    sends_response_format: false,
    sends_temperature: false,
    duration_source: DurationSource::TopLevel,
    supports_word_timestamps: true,
    keep_timestamps_on_format_fallback: false,
    uploads_flac: true,
    audio_request: AudioRequest::Multipart,
    models_query: "",
    diarization: Diarization::WordSpeakers,
    transcriptions_path: "/stt",
    sends_itn_format: true,
    auth: AuthScheme::Bearer,
    model_field: "model",
    language_field: "language",
    fixed_params: &[],
};

const OPENAI: EndpointProfile = EndpointProfile {
    diarization: Diarization::DiarizedJson,
    ..Compatibility::DirectOpenAi.base_profile()
};

// Diarization needs verbose_json with word timestamps, and speakers land on words.
const FIREWORKS: EndpointProfile = EndpointProfile {
    diarization: Diarization::WordSpeakers,
    ..Compatibility::DirectOpenAi.base_profile()
};

// Spacing and audio events arrive as entries next to the words.
const ELEVENLABS: EndpointProfile = EndpointProfile {
    timestamp_mode: TimestampMode::WordTimings,
    dictionary_mode: DictionaryMode::KeyTerms(KeyTermRules {
        field: "keyterms",
        model_prefix: "scribe_v2",
        max_chars: 49,
        max_words: 5,
        max_total_chars: usize::MAX,
        rejected_chars: &['<', '>', '{', '}', '[', ']', '\\'],
    }),
    sends_response_format: false,
    sends_temperature: false,
    duration_source: DurationSource::TopLevel,
    supports_word_timestamps: true,
    keep_timestamps_on_format_fallback: false,
    uploads_flac: true,
    audio_request: AudioRequest::Multipart,
    models_query: "",
    diarization: Diarization::WordSpeakers,
    transcriptions_path: "/speech-to-text",
    sends_itn_format: false,
    auth: AuthScheme::XiApiKey,
    model_field: "model_id",
    language_field: "language_code",
    // Otherwise tags like "(laughter)" land in the text.
    fixed_params: &[("tag_audio_events", "false")],
};

// Words always come back; utterances add timed segments with speakers.
const DEEPGRAM: EndpointProfile = EndpointProfile {
    timestamp_mode: TimestampMode::WordTimings,
    // The limit is 500 tokens across all terms; over it the request fails.
    dictionary_mode: DictionaryMode::KeyTerms(KeyTermRules {
        field: "keyterm",
        model_prefix: "nova-3",
        max_chars: 50,
        max_words: usize::MAX,
        max_total_chars: 500,
        rejected_chars: &[],
    }),
    sends_response_format: false,
    sends_temperature: false,
    duration_source: DurationSource::TopLevel,
    supports_word_timestamps: true,
    keep_timestamps_on_format_fallback: false,
    uploads_flac: true,
    audio_request: AudioRequest::RawBody,
    models_query: "",
    diarization: Diarization::SegmentSpeakers,
    transcriptions_path: "/listen",
    sends_itn_format: false,
    auth: AuthScheme::Token,
    model_field: "model",
    language_field: "language",
    fixed_params: &[("smart_format", "true"), ("punctuate", "true")],
};

const HOST_PROFILES: &[HostProfile] = &[
    HostProfile {
        host_suffixes: &["mistral.ai"],
        profile: MISTRAL,
    },
    HostProfile {
        host_suffixes: &["openrouter.ai"],
        profile: OPENROUTER,
    },
    HostProfile {
        host_suffixes: &["x.ai"],
        profile: XAI,
    },
    HostProfile {
        host_suffixes: &["openai.com"],
        profile: OPENAI,
    },
    HostProfile {
        host_suffixes: &["fireworks.ai"],
        profile: FIREWORKS,
    },
    HostProfile {
        host_suffixes: &["elevenlabs.io"],
        profile: ELEVENLABS,
    },
    HostProfile {
        host_suffixes: &["deepgram.com"],
        profile: DEEPGRAM,
    },
];

pub fn resolve_profile(endpoint: &str) -> EndpointProfile {
    let endpoint = endpoint.trim().to_ascii_lowercase();
    let url = reqwest::Url::parse(&endpoint)
        .or_else(|_| reqwest::Url::parse(&format!("https://{endpoint}")));
    let host = url
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase));

    if let Some(entry) = host.as_deref().and_then(|host| {
        HOST_PROFILES.iter().find(|entry| {
            entry
                .host_suffixes
                .iter()
                .any(|suffix| host_matches(host, suffix))
        })
    }) {
        return entry.profile;
    }

    if host.as_deref().is_some_and(is_self_hosted_host) {
        Compatibility::SelfHosted.base_profile()
    } else {
        Compatibility::DirectOpenAi.base_profile()
    }
}

pub(crate) fn is_self_hosted_host(host: &str) -> bool {
    if host == "localhost" || host.ends_with(".local") {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) => ip.is_private() || ip.is_loopback() || ip.is_link_local(),
        Ok(std::net::IpAddr::V6(ip)) => {
            ip.is_loopback()
                || (ip.segments()[0] & 0xfe00) == 0xfc00
                || (ip.segments()[0] & 0xffc0) == 0xfe80
        }
        Err(_) => false,
    }
}

fn host_matches(host: &str, suffix: &str) -> bool {
    host == suffix
        || host
            .strip_suffix(suffix)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelCapabilities {
    pub supports_timestamps: bool,
    pub supports_word_timestamps: bool,
}

fn effective_capabilities(profile: &EndpointProfile) -> ModelCapabilities {
    ModelCapabilities {
        supports_timestamps: profile.timestamp_mode != TimestampMode::None,
        supports_word_timestamps: profile.supports_word_timestamps,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestPlan {
    pub response_format: ResponseFormat,
    pub timestamp_granularities: Vec<&'static str>,
}

pub fn plan_request(
    profile: &EndpointProfile,
    wants_timestamps: bool,
    granularity: Option<TimestampGranularity>,
    diarize: bool,
) -> RequestPlan {
    let (wants_timestamps, granularity) = match (diarize, profile.diarization) {
        (false, _) | (true, Diarization::None) => (wants_timestamps, granularity),
        (true, Diarization::SegmentSpeakers) => {
            (true, granularity.or(Some(TimestampGranularity::Segment)))
        }
        (true, Diarization::WordSpeakers) => (true, Some(TimestampGranularity::Word)),
        // diarized_json rejects timestamp_granularities; its segments carry times.
        (true, Diarization::DiarizedJson) => (false, None),
    };
    let caps = effective_capabilities(profile);
    if !wants_timestamps || !caps.supports_timestamps {
        return RequestPlan {
            response_format: ResponseFormat::Json,
            timestamp_granularities: Vec::new(),
        };
    }

    let timestamp_granularities = match granularity {
        Some(TimestampGranularity::Word) if caps.supports_word_timestamps => {
            vec!["segment", "word"]
        }
        Some(TimestampGranularity::Word | TimestampGranularity::Segment) | None => {
            vec!["segment"]
        }
    };

    let response_format = if profile.sends_response_format
        && profile.timestamp_mode == TimestampMode::OpenAiVerboseJson
    {
        ResponseFormat::VerboseJson
    } else {
        ResponseFormat::Json
    };

    RequestPlan {
        response_format,
        timestamp_granularities,
    }
}

pub struct TranscriptionFormParams<'a> {
    pub model: &'a str,
    pub response_format: ResponseFormat,
    pub timestamp_granularities: &'a [&'a str],
    pub language: Option<&'a str>,
    pub dictionary: &'a [String],
    pub prompt: Option<&'a str>,
    pub(crate) diarize: bool,
}

pub fn build_transcription_form(
    profile: &EndpointProfile,
    file_part: Part,
    params: TranscriptionFormParams<'_>,
) -> Form {
    // `file` goes last: xAI ignores fields sent after it.
    let mut form = Form::new().text(profile.model_field, params.model.to_string());

    let diarized_json = params.diarize && profile.diarization == Diarization::DiarizedJson;
    let wants_timestamps =
        !params.timestamp_granularities.is_empty() && profile.timestamp_mode != TimestampMode::None;
    let use_verbose_json = profile.sends_response_format
        && params.response_format == ResponseFormat::VerboseJson
        && profile.timestamp_mode == TimestampMode::OpenAiVerboseJson;

    if diarized_json {
        // Required for inputs over 30 seconds.
        form = form
            .text("response_format", "diarized_json")
            .text("chunking_strategy", "auto");
    } else if profile.sends_response_format {
        form = form.text(
            "response_format",
            if use_verbose_json {
                ResponseFormat::VerboseJson.as_str()
            } else {
                ResponseFormat::Json.as_str()
            },
        );
    }

    if profile.sends_temperature {
        form = form.text("temperature", "0");
    }

    match profile.timestamp_mode {
        TimestampMode::OpenAiVerboseJson
            if wants_timestamps
                && (use_verbose_json || profile.keep_timestamps_on_format_fallback) =>
        {
            form = append_timestamps(form, profile.timestamp_mode, params.timestamp_granularities);
        }
        TimestampMode::NativeGranularities if wants_timestamps => {
            form = append_timestamps(form, profile.timestamp_mode, params.timestamp_granularities);
        }
        TimestampMode::WordTimings => {
            form = form.text(
                "timestamps_granularity",
                if wants_timestamps { "word" } else { "none" },
            );
        }
        _ => {}
    }

    if let Some(language) = params.language {
        form = form.text(profile.language_field, language.to_string());
        if profile.sends_itn_format {
            form = form.text("format", "true");
        }
    }

    if params.diarize
        && matches!(
            profile.diarization,
            Diarization::SegmentSpeakers | Diarization::WordSpeakers
        )
    {
        form = form.text("diarize", "true");
    }

    for (name, value) in profile.fixed_params {
        form = form.text(*name, *value);
    }

    // Diarize models reject prompts, which is where the dictionary goes.
    if !diarized_json {
        form = apply_dictionary_and_prompt(
            form,
            profile.dictionary_mode,
            params.model,
            params.dictionary,
            params.prompt,
        );
    }
    form.part("file", file_part)
}

/// Adds transcription options to the URL of a raw-body request.
pub fn append_transcription_query(
    profile: &EndpointProfile,
    url: &mut reqwest::Url,
    params: &TranscriptionFormParams<'_>,
) {
    let mut query = url.query_pairs_mut();
    query.append_pair(profile.model_field, params.model);
    for (name, value) in profile.fixed_params {
        query.append_pair(name, value);
    }
    match params.language {
        Some(language) => query.append_pair(profile.language_field, language),
        None => query.append_pair("detect_language", "true"),
    };
    if params.diarize && profile.diarization != Diarization::None {
        query.append_pair("diarize_model", "latest");
    }
    if !params.timestamp_granularities.is_empty() {
        query.append_pair("utterances", "true");
    }
    if let DictionaryMode::KeyTerms(rules) = profile.dictionary_mode {
        for term in rules.terms(params.model, params.dictionary) {
            query.append_pair(rules.field, &term);
        }
    }
}

fn append_timestamps(mut form: Form, mode: TimestampMode, granularities: &[&str]) -> Form {
    for granularity in granularities {
        let value = granularity.trim();
        if value.is_empty() {
            continue;
        }
        form = match mode {
            TimestampMode::OpenAiVerboseJson => {
                form.text("timestamp_granularities[]", value.to_string())
            }
            TimestampMode::NativeGranularities => {
                form.text("timestamp_granularities", value.to_string())
            }
            TimestampMode::WordTimings | TimestampMode::None => form,
        };
    }
    form
}

fn apply_dictionary_and_prompt(
    mut form: Form,
    mode: DictionaryMode,
    model: &str,
    dictionary: &[String],
    prompt: Option<&str>,
) -> Form {
    let trimmed_prompt = prompt.map(str::trim).filter(|value| !value.is_empty());

    match mode {
        DictionaryMode::Prompt => {
            let dictionary_terms = crate::dictionary::build_dictionary_prompt(dictionary);
            if let Some(prompt) = compose_openai_prompt(trimmed_prompt, dictionary_terms.as_deref())
            {
                form = form.text("prompt", prompt);
            }
        }
        DictionaryMode::KeyTerms(rules) => {
            for term in rules.terms(model, dictionary) {
                form = form.text(rules.field, term);
            }
        }
        DictionaryMode::ContextBias => {
            let mut seen = std::collections::HashSet::new();
            let tokens = crate::dictionary::sanitize_dictionary_entries(dictionary)
                .into_iter()
                .flat_map(|term| {
                    term.split_whitespace()
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .filter(|token| seen.insert(token.to_lowercase()))
                .take(crate::dictionary::MAX_DICTIONARY_ENTRIES);
            for token in tokens {
                form = form.text("context_bias", token);
            }
            if let Some(prompt) = trimmed_prompt {
                form = form.text("prompt", prompt.to_string());
            }
        }
    }

    form
}

fn compose_openai_prompt(extra: Option<&str>, dictionary_terms: Option<&str>) -> Option<String> {
    let dictionary_prompt = dictionary_terms.map(|terms| {
        format!("Prefer these names, product terms, and spellings when they are spoken: {terms}")
    });
    match (extra, dictionary_prompt) {
        (Some(extra), Some(dict)) => Some(format!("{extra}\n\n{dict}")),
        (Some(extra), None) => Some(extra.to_string()),
        (None, Some(dict)) => Some(dict),
        (None, None) => None,
    }
}

pub fn apply_auth(
    builder: reqwest::RequestBuilder,
    scheme: AuthScheme,
    api_key: &str,
) -> reqwest::RequestBuilder {
    if api_key.is_empty() {
        return builder;
    }
    match scheme {
        AuthScheme::Bearer => builder.header("Authorization", format!("Bearer {api_key}")),
        AuthScheme::Token => builder.header("Authorization", format!("Token {api_key}")),
        AuthScheme::XiApiKey => builder.header("xi-api-key", api_key),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_is_openai_compatible() {
        let profile = resolve_profile("https://api.openai.com/v1");
        assert!(profile.sends_response_format);
        assert_eq!(profile.timestamp_mode, TimestampMode::OpenAiVerboseJson);
    }

    #[test]
    fn mistral_resolves_from_host() {
        let profile = resolve_profile("https://api.mistral.ai/v1");
        assert_eq!(profile.timestamp_mode, TimestampMode::NativeGranularities);
        assert_eq!(profile.dictionary_mode, DictionaryMode::ContextBias);
        assert!(!profile.sends_response_format);

        assert_eq!(resolve_profile("api.mistral.ai/v1"), profile);
    }

    #[test]
    fn mistral_profile_does_not_match_path_or_host_substring() {
        let proxy = resolve_profile("https://proxy.example/mistral.ai/v1");
        assert_eq!(proxy.dictionary_mode, DictionaryMode::Prompt);
        assert!(proxy.sends_response_format);

        let lookalike = resolve_profile("https://mistral.ai.evil.example/v1");
        assert_eq!(lookalike.dictionary_mode, DictionaryMode::Prompt);
        assert!(lookalike.sends_response_format);
    }

    #[test]
    fn self_hosted_endpoints_disable_word_timestamps() {
        let profile = resolve_profile("http://127.0.0.1:8080/v1");
        assert!(!profile.supports_word_timestamps);
        assert!(profile.keep_timestamps_on_format_fallback);
    }

    #[test]
    fn does_not_request_timestamps_when_caller_opts_out() {
        let profile = resolve_profile("https://api.openai.com/v1");
        let plan = plan_request(&profile, false, None, false);
        assert_eq!(plan.response_format, ResponseFormat::Json);
        assert!(plan.timestamp_granularities.is_empty());
    }

    #[test]
    fn requests_verbose_json_with_word_timestamps() {
        let profile = resolve_profile("https://api.openai.com/v1");
        let plan = plan_request(&profile, true, Some(TimestampGranularity::Word), false);
        assert_eq!(plan.response_format, ResponseFormat::VerboseJson);
        assert_eq!(plan.timestamp_granularities, vec!["segment", "word"]);
    }
}
