use reqwest::multipart::{Form, Part};

use crate::remote::ResponseFormat;
use crate::TimestampGranularity;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointProfile {
    pub timestamp_mode: TimestampMode,
    pub dictionary_mode: DictionaryMode,
    pub sends_response_format: bool,
    pub sends_temperature: bool,
    pub duration_source: DurationSource,
    pub supports_word_timestamps: bool,
    pub keep_timestamps_on_format_fallback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampMode {
    OpenAiVerboseJson,
    NativeGranularities,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictionaryMode {
    Prompt,
    ContextBias,
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
            },
            Self::SelfHosted => EndpointProfile {
                timestamp_mode: TimestampMode::OpenAiVerboseJson,
                dictionary_mode: DictionaryMode::Prompt,
                sends_response_format: true,
                sends_temperature: true,
                duration_source: DurationSource::TopLevel,
                supports_word_timestamps: false,
                keep_timestamps_on_format_fallback: true,
            },
        }
    }
}

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
};

const HOST_PROFILES: &[HostProfile] = &[HostProfile {
    host_suffixes: &["mistral.ai"],
    profile: MISTRAL,
}];

pub fn resolve_profile(endpoint: &str) -> EndpointProfile {
    let endpoint = endpoint.trim().to_ascii_lowercase();
    let host = reqwest::Url::parse(&endpoint)
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

    if host
        .as_deref()
        .is_some_and(|host| host == "localhost" || host.starts_with("127."))
    {
        Compatibility::SelfHosted.base_profile()
    } else {
        Compatibility::DirectOpenAi.base_profile()
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
) -> RequestPlan {
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
        Some(TimestampGranularity::Word) | Some(TimestampGranularity::Segment) | None => {
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
}

pub fn build_transcription_form(
    profile: &EndpointProfile,
    file_part: Part,
    params: TranscriptionFormParams<'_>,
) -> Form {
    let mut form = Form::new()
        .part("file", file_part)
        .text("model", params.model.to_string());

    let wants_timestamps =
        !params.timestamp_granularities.is_empty() && profile.timestamp_mode != TimestampMode::None;
    let use_verbose_json = profile.sends_response_format
        && params.response_format == ResponseFormat::VerboseJson
        && profile.timestamp_mode == TimestampMode::OpenAiVerboseJson;

    if profile.sends_response_format {
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

    let send_granularities = wants_timestamps
        && match profile.timestamp_mode {
            TimestampMode::OpenAiVerboseJson => {
                use_verbose_json || profile.keep_timestamps_on_format_fallback
            }
            TimestampMode::NativeGranularities => true,
            TimestampMode::None => false,
        };
    if send_granularities {
        form = append_timestamps(form, profile.timestamp_mode, params.timestamp_granularities);
    }

    if let Some(language) = params.language {
        form = form.text("language", language.to_string());
    }

    apply_dictionary_and_prompt(
        form,
        profile.dictionary_mode,
        params.dictionary,
        params.prompt,
    )
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
            TimestampMode::None => form,
        };
    }
    form
}

fn apply_dictionary_and_prompt(
    mut form: Form,
    mode: DictionaryMode,
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
        DictionaryMode::ContextBias => {
            let mut seen = std::collections::HashSet::new();
            let mut count = 0;
            'outer: for term in crate::dictionary::sanitize_dictionary_entries(dictionary) {
                for token in term.split_whitespace() {
                    if !seen.insert(token.to_lowercase()) {
                        continue;
                    }
                    form = form.text("context_bias", token.to_string());
                    count += 1;
                    if count >= crate::dictionary::MAX_DICTIONARY_ENTRIES {
                        break 'outer;
                    }
                }
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

pub fn apply_auth(builder: reqwest::RequestBuilder, api_key: &str) -> reqwest::RequestBuilder {
    if api_key.is_empty() {
        builder
    } else {
        builder.header("Authorization", format!("Bearer {api_key}"))
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
        let plan = plan_request(&profile, false, None);
        assert_eq!(plan.response_format, ResponseFormat::Json);
        assert!(plan.timestamp_granularities.is_empty());
    }

    #[test]
    fn requests_verbose_json_with_word_timestamps() {
        let profile = resolve_profile("https://api.openai.com/v1");
        let plan = plan_request(&profile, true, Some(TimestampGranularity::Word));
        assert_eq!(plan.response_format, ResponseFormat::VerboseJson);
        assert_eq!(plan.timestamp_granularities, vec!["segment", "word"]);
    }
}
