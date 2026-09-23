use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use base64::Engine as _;
use reqwest::{
    Client,
    header::{CONTENT_LENGTH, CONTENT_TYPE, RETRY_AFTER},
    multipart,
};
use serde::{Deserialize, Serialize};
use tokio_util::io::ReaderStream;

use super::provider::{
    AudioRequest, Diarization, DurationSource, EndpointProfile, TranscriptionFormParams,
    append_transcription_query, apply_auth, build_transcription_form, is_self_hosted_host,
    plan_request, resolve_profile,
};
use super::{
    RemoteError, RemoteErrorKind, ResponseFormat, config_error, parse_retry_after,
    parse_upstream_error, transport_error,
};
use crate::{TimestampGranularity, Transcription};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const MODELS_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct RemoteConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RemoteRequestParams<'a> {
    pub model: &'a str,
    pub language: Option<&'a str>,
    pub dictionary: &'a [String],
    pub prompt: Option<&'a str>,
    pub timestamps: bool,
    pub timestamp_granularity: Option<TimestampGranularity>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiarizedSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiarizedTranscription {
    pub transcription: Transcription,
    pub segments: Option<Vec<DiarizedSegment>>,
}

pub struct RemoteEngine {
    client: Client,
    config: RemoteConfig,
}

impl RemoteEngine {
    pub fn new(client: Client, config: RemoteConfig) -> Self {
        Self { client, config }
    }

    pub fn config(&self) -> &RemoteConfig {
        &self.config
    }

    pub async fn transcribe_file(
        &self,
        audio_path: &Path,
        params: RemoteRequestParams<'_>,
    ) -> Result<Transcription, RemoteError> {
        self.transcribe_file_inner(audio_path, params, false)
            .await
            .map(|response| response.transcription)
    }

    pub async fn transcribe_file_diarized(
        &self,
        audio_path: &Path,
        params: RemoteRequestParams<'_>,
    ) -> Result<DiarizedTranscription, RemoteError> {
        self.transcribe_file_inner(audio_path, params, true).await
    }

    async fn transcribe_file_inner(
        &self,
        audio_path: &Path,
        params: RemoteRequestParams<'_>,
        diarize: bool,
    ) -> Result<DiarizedTranscription, RemoteError> {
        let endpoint = self.config.endpoint.trim();
        if endpoint.is_empty() {
            return Err(config_error("Remote speech endpoint is not configured"));
        }
        let model = params.model.trim();
        if model.is_empty() {
            return Err(config_error("Remote speech model is not configured"));
        }

        let profile = resolve_profile(endpoint);
        // Unsupported endpoints and models transcribe without speakers.
        let mut diarize = diarize && profile.supports_diarization(model);
        let url = format!("{}{}", api_base(endpoint), profile.transcriptions_path);
        let api_key = self.config.api_key.trim();
        let language = params
            .language
            .map(str::trim)
            .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("auto"))
            .map(str::to_string);

        if profile.audio_request == AudioRequest::Base64Json {
            return transcribe_base64(
                &self.client,
                &url,
                model,
                audio_path,
                language.as_deref(),
                api_key,
                &profile,
            )
            .await;
        }

        let plan = plan_request(
            &profile,
            params.timestamps,
            params.timestamp_granularity,
            diarize,
        );

        let extension = audio_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase);
        let mime_type = audio_mime_for_extension(extension.as_deref());
        let file_name = audio_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("recording.wav")
            .to_string();

        let flac = if profile.uploads_flac && extension.as_deref() == Some("wav") {
            let path = audio_path.to_path_buf();
            tokio::task::spawn_blocking(move || encode_wav_to_flac_file(&path))
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        let flac_file_name = format!(
            "{}.flac",
            audio_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("recording")
        );
        let mut use_flac = flac.is_some();

        let mut effective_format = plan.response_format;
        let mut granularities = plan.timestamp_granularities.clone();
        let body = loop {
            let (upload_path, upload_name, upload_mime) =
                if let (true, Some(flac)) = (use_flac, flac.as_ref()) {
                    (flac.0.as_path(), &flac_file_name, "audio/flac")
                } else {
                    (audio_path, &file_name, mime_type)
                };
            let file = tokio::fs::File::open(upload_path).await.map_err(|err| {
                transport_error(format!(
                    "Failed to read recording at {}: {err}",
                    upload_path.display()
                ))
            })?;
            let audio = reqwest::Body::wrap_stream(ReaderStream::new(file));
            let form_params = TranscriptionFormParams {
                model,
                response_format: effective_format,
                timestamp_granularities: &granularities,
                language: language.as_deref(),
                dictionary: params.dictionary,
                prompt: params.prompt,
                diarize,
            };
            let request = if profile.audio_request == AudioRequest::RawBody {
                let mut request_url = reqwest::Url::parse(&url)
                    .map_err(|_| config_error("Remote speech endpoint is not a valid URL"))?;
                append_transcription_query(&profile, &mut request_url, &form_params);
                // A known length avoids a chunked upload.
                let length = tokio::fs::metadata(upload_path)
                    .await
                    .map_err(|err| {
                        transport_error(format!(
                            "Failed to read recording at {}: {err}",
                            upload_path.display()
                        ))
                    })?
                    .len();
                self.client
                    .post(request_url)
                    .header(CONTENT_TYPE, upload_mime)
                    .header(CONTENT_LENGTH, length)
                    .body(audio)
            } else {
                let file_part = multipart::Part::stream(audio)
                    .file_name(upload_name.clone())
                    .mime_str(upload_mime)
                    .map_err(|err| {
                        transport_error(format!("Failed to prepare audio upload: {err}"))
                    })?;
                self.client.post(&url).multipart(build_transcription_form(
                    &profile,
                    file_part,
                    form_params,
                ))
            };
            let builder = apply_auth(request.timeout(DEFAULT_TIMEOUT), profile.auth, api_key);

            let response = builder.send().await.map_err(|err| {
                transport_error(format!("Failed to reach remote speech endpoint: {err}"))
            })?;
            let status = response.status();
            let retry_after = parse_retry_after(response.headers().get(RETRY_AFTER));
            let body_text = response.text().await.map_err(|err| {
                transport_error(format!("Failed to read remote speech response: {err}"))
            })?;
            if status.is_success() {
                break body_text;
            }
            let err = parse_upstream_error(status, retry_after, &body_text);
            if profile.sends_response_format
                && effective_format == ResponseFormat::VerboseJson
                && is_verbose_unsupported(&err)
            {
                effective_format = ResponseFormat::Json;
                if profile.keep_timestamps_on_format_fallback {
                    granularities = vec!["segment"];
                } else {
                    granularities.clear();
                }
                continue;
            }
            if use_flac && is_flac_unsupported(&err) {
                use_flac = false;
                continue;
            }
            if diarize && is_diarize_unsupported(&err) {
                diarize = false;
                let plan = plan_request(
                    &profile,
                    params.timestamps,
                    params.timestamp_granularity,
                    false,
                );
                effective_format = plan.response_format;
                granularities = plan.timestamp_granularities;
                continue;
            }
            return Err(err);
        };

        parse_transcription_body(&body, model, &profile, diarize)
    }

    pub async fn list_models(&self) -> Result<Vec<String>, RemoteError> {
        let endpoint = self.config.endpoint.trim();
        if endpoint.is_empty() {
            return Ok(Vec::new());
        }
        let profile = resolve_profile(endpoint);
        let url = format!("{}/models{}", api_base(endpoint), profile.models_query);
        let builder = apply_auth(
            self.client.get(url).timeout(MODELS_TIMEOUT),
            profile.auth,
            self.config.api_key.trim(),
        );

        let response = builder.send().await.map_err(|err| {
            transport_error(format!(
                "Failed to reach remote speech models endpoint: {err}"
            ))
        })?;
        let status = response.status();
        let retry_after = parse_retry_after(response.headers().get(RETRY_AFTER));
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(parse_upstream_error(status, retry_after, &body));
        }

        let parsed: ModelsResponse = response.json().await.map_err(|err| RemoteError {
            kind: RemoteErrorKind::Other,
            status: status.as_u16(),
            message: format!("Failed to parse remote speech models response: {err}"),
            error_type: None,
            code: None,
            param: None,
            retry_after: None,
        })?;
        Ok(parsed.into_ids())
    }
}

async fn transcribe_base64(
    client: &Client,
    url: &str,
    model: &str,
    audio_path: &Path,
    language: Option<&str>,
    api_key: &str,
    profile: &EndpointProfile,
) -> Result<DiarizedTranscription, RemoteError> {
    let bytes = tokio::fs::read(audio_path).await.map_err(|err| {
        transport_error(format!(
            "Failed to read recording at {}: {err}",
            audio_path.display()
        ))
    })?;
    let format = audio_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map_or_else(|| "wav".to_string(), str::to_ascii_lowercase);
    let request = Base64AudioRequest {
        model,
        input_audio: Base64Audio {
            data: base64::engine::general_purpose::STANDARD.encode(&bytes),
            format: &format,
        },
        language,
    };

    let builder = apply_auth(
        client.post(url).json(&request).timeout(DEFAULT_TIMEOUT),
        profile.auth,
        api_key,
    );
    let response = builder
        .send()
        .await
        .map_err(|err| transport_error(format!("Failed to reach remote speech endpoint: {err}")))?;
    let status = response.status();
    let retry_after = parse_retry_after(response.headers().get(RETRY_AFTER));
    let body_text = response
        .text()
        .await
        .map_err(|err| transport_error(format!("Failed to read remote speech response: {err}")))?;
    if !status.is_success() {
        return Err(parse_upstream_error(status, retry_after, &body_text));
    }

    parse_transcription_body(&body_text, model, profile, false)
}

#[derive(Serialize)]
struct Base64AudioRequest<'a> {
    model: &'a str,
    input_audio: Base64Audio<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
}

#[derive(Serialize)]
struct Base64Audio<'a> {
    data: String,
    format: &'a str,
}

#[derive(Debug, Deserialize)]
struct TranscriptionBody {
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Option<Vec<UpstreamSegment>>,
    #[serde(default)]
    words: Option<Vec<UpstreamSegment>>,
    #[serde(default, alias = "language_code")]
    language: Option<String>,
    #[serde(default, alias = "audio_duration_secs")]
    duration: Option<f32>,
    #[serde(default)]
    usage: Option<UsageBody>,
    #[serde(default)]
    metadata: Option<DeepgramMetadata>,
    #[serde(default)]
    results: Option<DeepgramResults>,
}

#[derive(Debug, Deserialize)]
struct UsageBody {
    #[serde(default)]
    prompt_audio_seconds: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct DeepgramMetadata {
    #[serde(default)]
    duration: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct DeepgramResults {
    #[serde(default)]
    channels: Vec<DeepgramChannel>,
    #[serde(default)]
    utterances: Option<Vec<UpstreamSegment>>,
}

#[derive(Debug, Deserialize)]
struct DeepgramChannel {
    #[serde(default)]
    alternatives: Vec<DeepgramAlternative>,
    #[serde(default)]
    detected_language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeepgramAlternative {
    #[serde(default)]
    transcript: String,
    #[serde(default)]
    words: Vec<UpstreamSegment>,
}

impl TranscriptionBody {
    // Deepgram nests the transcript under results.channels[0].alternatives[0].
    fn flatten_results(&mut self) {
        let Some(results) = self.results.take() else {
            return;
        };
        if let Some(channel) = results.channels.into_iter().next() {
            if let Some(alternative) = channel.alternatives.into_iter().next() {
                self.text = alternative.transcript;
                self.words = Some(alternative.words);
            }
            self.language = self.language.take().or(channel.detected_language);
        }
        self.segments = self.segments.take().or(results.utterances);
        if let Some(metadata) = self.metadata.take() {
            self.duration = self.duration.or(metadata.duration);
        }
    }
}

#[derive(Debug, Deserialize)]
struct UpstreamSegment {
    #[serde(default)]
    start: f32,
    #[serde(default)]
    end: f32,
    #[serde(default, alias = "punctuated_word", alias = "transcript")]
    text: String,
    #[serde(default)]
    word: String,
    #[serde(default, alias = "speaker_id")]
    speaker: Option<SpeakerLabel>,
    /// ElevenLabs: "word", "spacing", or "audio_event".
    #[serde(default, rename = "type")]
    kind: Option<String>,
    /// Follows the previous word with no space, as in unspaced scripts.
    #[serde(skip)]
    attached: bool,
}

// xAI and Deepgram number speakers, the others name them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SpeakerLabel {
    Name(String),
    Index(i64),
}

impl SpeakerLabel {
    fn to_id(&self) -> String {
        match self {
            Self::Name(name) => name.clone(),
            Self::Index(index) => format!("speaker_{index}"),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ModelsResponse {
    Wrapped { data: Vec<ModelEntry> },
    Deepgram { stt: Vec<DeepgramModel> },
    List(Vec<ModelEntry>),
}

impl ModelsResponse {
    fn into_ids(self) -> Vec<String> {
        match self {
            ModelsResponse::Wrapped { data } | ModelsResponse::List(data) => {
                data.into_iter().map(|entry| entry.id).collect()
            }
            // One entry per model and language.
            ModelsResponse::Deepgram { stt } => {
                let mut ids: Vec<String> = Vec::new();
                for model in stt.into_iter().filter(|model| model.batch) {
                    if !ids.contains(&model.canonical_name) {
                        ids.push(model.canonical_name);
                    }
                }
                ids
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Debug, Deserialize)]
struct DeepgramModel {
    canonical_name: String,
    #[serde(default = "default_true")]
    batch: bool,
}

fn default_true() -> bool {
    true
}

fn parse_transcription_body(
    body: &str,
    model: &str,
    profile: &EndpointProfile,
    include_speakers: bool,
) -> Result<DiarizedTranscription, RemoteError> {
    let mut parsed =
        serde_json::from_str::<TranscriptionBody>(body).map_err(|err| RemoteError {
            kind: RemoteErrorKind::Other,
            status: 200,
            message: format!("Failed to parse remote speech response: {err}"),
            error_type: None,
            code: None,
            param: None,
            retry_after: None,
        })?;
    parsed.flatten_results();
    let duration_seconds = match profile.duration_source {
        DurationSource::TopLevel => parsed.duration,
        DurationSource::UsagePromptAudioSeconds => parsed
            .usage
            .as_ref()
            .and_then(|usage| usage.prompt_audio_seconds),
    };
    let words = parsed.words.map(spoken_words);
    let diarized_segments = match profile.diarization {
        _ if !include_speakers => None,
        Diarization::WordSpeakers => words.as_deref().and_then(group_speaker_words),
        _ => parsed.segments.as_deref().map(map_diarized_text),
    };
    let segments = match parsed.segments.filter(|segments| !segments.is_empty()) {
        Some(segments) => Some(map_timed_text(&segments)),
        None => words
            .as_deref()
            .filter(|words| !words.is_empty())
            .map(sentence_segments),
    };
    let words = if profile.supports_word_timestamps {
        words.as_deref().map(map_timed_text)
    } else {
        None
    };
    Ok(DiarizedTranscription {
        transcription: Transcription {
            text: parsed.text,
            segments,
            words,
            model_id: model.to_string(),
            language: parsed.language,
            duration_ms: duration_seconds.map_or(0, |seconds| (seconds.max(0.0) * 1000.0) as u128),
        },
        segments: diarized_segments,
    })
}

// Drops spacing and audio event entries, remembering where no space separated two words.
fn spoken_words(entries: Vec<UpstreamSegment>) -> Vec<UpstreamSegment> {
    let mut words = Vec::with_capacity(entries.len());
    let mut spaced = true;
    for mut entry in entries {
        match entry.kind.as_deref() {
            None => words.push(entry),
            Some("word") => {
                entry.attached = !spaced;
                spaced = false;
                words.push(entry);
            }
            _ => spaced = true,
        }
    }
    words
}

fn map_timed_text(items: &[UpstreamSegment]) -> Vec<crate::TranscriptionSegment> {
    items
        .iter()
        .map(|item| crate::TranscriptionSegment {
            start: item.start,
            end: item.end,
            text: upstream_segment_text(item),
        })
        .collect()
}

fn map_diarized_text(items: &[UpstreamSegment]) -> Vec<DiarizedSegment> {
    items
        .iter()
        .map(|item| DiarizedSegment {
            start: item.start,
            end: item.end,
            text: upstream_segment_text(item),
            speaker: item.speaker.as_ref().map(SpeakerLabel::to_id),
        })
        .collect()
}

// Merges consecutive words from the same speaker into one segment.
fn group_speaker_words(words: &[UpstreamSegment]) -> Option<Vec<DiarizedSegment>> {
    if words.iter().all(|word| word.speaker.is_none()) {
        return None;
    }
    Some(join_words(words, |last, _, speaker| {
        last.speaker != *speaker
    }))
}

const SEGMENT_PAUSE_SECONDS: f32 = 1.0;

// Timed segments for responses that only carry words: one per sentence or pause.
fn sentence_segments(words: &[UpstreamSegment]) -> Vec<crate::TranscriptionSegment> {
    join_words(words, |last, word, _| {
        last.text.ends_with(['.', '?', '!', '。', '？', '！'])
            || word.start - last.end >= SEGMENT_PAUSE_SECONDS
    })
    .into_iter()
    .map(|segment| crate::TranscriptionSegment {
        start: segment.start,
        end: segment.end,
        text: segment.text,
    })
    .collect()
}

fn join_words(
    words: &[UpstreamSegment],
    starts_segment: impl Fn(&DiarizedSegment, &UpstreamSegment, &Option<String>) -> bool,
) -> Vec<DiarizedSegment> {
    let mut segments: Vec<DiarizedSegment> = Vec::new();
    for word in words {
        let text = upstream_segment_text(word);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        let speaker = word.speaker.as_ref().map(SpeakerLabel::to_id);
        match segments.last_mut() {
            Some(last) if !starts_segment(last, word, &speaker) => {
                last.end = word.end;
                if !word.attached {
                    last.text.push(' ');
                }
                last.text.push_str(text);
            }
            _ => segments.push(DiarizedSegment {
                start: word.start,
                end: word.end,
                text: text.to_string(),
                speaker,
            }),
        }
    }
    segments
}

fn upstream_segment_text(item: &UpstreamSegment) -> String {
    if item.text.is_empty() {
        item.word.clone()
    } else {
        item.text.clone()
    }
}

fn is_verbose_unsupported(err: &RemoteError) -> bool {
    err.kind == RemoteErrorKind::InvalidRequest
        && error_mentions(
            err,
            &["verbose_json", "response_format", "timestamp_granularit"],
        )
}

fn is_diarize_unsupported(err: &RemoteError) -> bool {
    err.kind == RemoteErrorKind::InvalidRequest
        && error_mentions(err, &["diariz", "chunking_strategy"])
}

fn error_mentions(err: &RemoteError, needles: &[&str]) -> bool {
    [
        err.message.as_str(),
        err.param.as_deref().unwrap_or(""),
        err.code.as_deref().unwrap_or(""),
        err.error_type.as_deref().unwrap_or(""),
    ]
    .iter()
    .any(|field| {
        let lowered = field.to_ascii_lowercase();
        needles.iter().any(|needle| lowered.contains(needle))
    })
}

struct TempFlacFile(PathBuf);

impl Drop for TempFlacFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

struct WavSampleSource {
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
    channels: usize,
    sample_rate: usize,
    buffer: Vec<i32>,
}

impl flacenc::source::Source for WavSampleSource {
    fn channels(&self) -> usize {
        self.channels
    }

    fn bits_per_sample(&self) -> usize {
        16
    }

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn read_samples<F: flacenc::source::Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, flacenc::error::SourceError> {
        self.buffer.clear();
        for sample in self
            .reader
            .samples::<i16>()
            .take(block_size * self.channels)
        {
            let sample = sample.map_err(flacenc::error::SourceError::from_io_error)?;
            self.buffer.push(i32::from(sample));
        }
        dest.fill_interleaved(&self.buffer)?;
        Ok(self.buffer.len() / self.channels)
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.reader.duration() as usize)
    }
}

fn encode_wav_to_flac_file(audio_path: &Path) -> Option<TempFlacFile> {
    use flacenc::bitsink::ByteSink;
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let reader = hound::WavReader::open(audio_path).ok()?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int
        || spec.bits_per_sample != 16
        || reader.duration() == 0
    {
        return None;
    }

    let config = flacenc::config::Encoder::default().into_verified().ok()?;
    let source = WavSampleSource {
        channels: spec.channels as usize,
        sample_rate: spec.sample_rate as usize,
        buffer: Vec::new(),
        reader,
    };
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size).ok()?;
    let mut sink = ByteSink::new();
    stream.write(&mut sink).ok()?;

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let temp = TempFlacFile(std::env::temp_dir().join(format!(
        "glimpse-speech-upload-{}-{nanos}.flac",
        std::process::id(),
    )));
    std::fs::write(&temp.0, sink.as_slice()).ok()?;
    Some(temp)
}

fn is_flac_unsupported(err: &RemoteError) -> bool {
    if err.kind == RemoteErrorKind::InvalidRequest && err.param.as_deref() == Some("file") {
        return true;
    }
    error_mentions(
        err,
        &[
            "flac",
            "file format",
            "audio format",
            "unsupported format",
            "invalid format",
            "decod",
            "corrupt",
            "riff",
            "ffmpeg",
            "wave file",
        ],
    )
}

fn audio_mime_for_extension(extension: Option<&str>) -> &'static str {
    match extension {
        Some("wav") => "audio/wav",
        Some("mp3" | "mpga" | "mpeg") => "audio/mpeg",
        Some("m4a" | "mp4") => "audio/mp4",
        Some("aac") => "audio/aac",
        Some("flac") => "audio/flac",
        Some("ogg" | "oga") => "audio/ogg",
        Some("opus") => "audio/opus",
        Some("webm") => "audio/webm",
        _ => "application/octet-stream",
    }
}

fn api_base(endpoint: &str) -> String {
    let mut base = ensure_scheme(endpoint.trim())
        .trim_end_matches('/')
        .to_string();
    for suffix in [
        "/v1/audio/transcriptions",
        "/audio/transcriptions",
        "/v1/stt",
        "/stt",
        "/v1/speech-to-text",
        "/speech-to-text",
        "/v1/listen",
        "/listen",
    ] {
        if base.ends_with(suffix) {
            base.truncate(base.len() - suffix.len());
            break;
        }
    }
    let base = base.trim_end_matches('/').to_string();
    if base.is_empty() || ends_with_version_segment(&base) {
        base
    } else {
        format!("{base}/v1")
    }
}

fn ensure_scheme(endpoint: &str) -> String {
    let trimmed = endpoint.trim();
    if trimmed.is_empty() || trimmed.contains("://") {
        return trimmed.to_string();
    }
    let authority = trimmed.split('/').next().unwrap_or(trimmed);
    let host = authority
        .strip_prefix('[')
        .and_then(|value| value.split_once(']').map(|(host, _)| host))
        .unwrap_or_else(|| authority.split(':').next().unwrap_or(authority));
    let local = is_self_hosted_host(host);
    let scheme = if local { "http" } else { "https" };
    format!("{scheme}://{trimmed}")
}

fn ends_with_version_segment(base: &str) -> bool {
    base.rsplit('/').next().is_some_and(|segment| {
        segment.len() > 1
            && segment.starts_with('v')
            && segment[1..].bytes().all(|byte| byte.is_ascii_digit())
    })
}

#[cfg(test)]
mod tests {
    use super::{ensure_scheme, parse_transcription_body};
    use crate::remote::provider::resolve_profile;

    #[test]
    fn infers_endpoint_scheme() {
        assert_eq!(
            ensure_scheme("server.local:8000"),
            "http://server.local:8000"
        );
        assert_eq!(ensure_scheme("[::1]:8000"), "http://[::1]:8000");
        assert_eq!(ensure_scheme("api.example.com"), "https://api.example.com");
        assert_eq!(
            ensure_scheme("http://localhost:8000"),
            "http://localhost:8000"
        );
    }

    #[test]
    fn preserves_diarized_speaker_ids_separately_from_standard_segments() {
        let response = parse_transcription_body(
            r#"{
                "text": "Hello there",
                "language": "en",
                "segments": [{
                    "start": 0.25,
                    "end": 1.5,
                    "text": "Hello there",
                    "speaker_id": "speaker_0"
                }],
                "usage": { "prompt_audio_seconds": 2 }
            }"#,
            "voxtral-mini-latest",
            &resolve_profile("https://api.mistral.ai/v1"),
            true,
        )
        .expect("valid transcription response");

        assert_eq!(
            response.transcription.segments.as_deref(),
            Some(
                [crate::TranscriptionSegment {
                    start: 0.25,
                    end: 1.5,
                    text: "Hello there".to_string(),
                }]
                .as_slice()
            )
        );
        assert_eq!(
            response.segments.as_deref(),
            Some(
                [super::DiarizedSegment {
                    start: 0.25,
                    end: 1.5,
                    text: "Hello there".to_string(),
                    speaker: Some("speaker_0".to_string()),
                }]
                .as_slice()
            )
        );
    }
}
