use std::{error::Error as StdError, fmt, sync::Arc};

use anyhow::anyhow;

use crate::service::{SpeechService, TranscribeRequest};
use crate::Transcription;

#[cfg(feature = "remote")]
use crate::service::AudioInput;

#[cfg(feature = "remote")]
use reqwest::Client;

#[cfg(feature = "remote")]
use crate::remote::{RemoteEngine, RemoteError, RemoteRequestParams};

#[cfg(feature = "remote")]
pub use crate::remote::RemoteConfig;

#[derive(Debug)]
pub enum TranscribeError {
    Local(anyhow::Error),
    #[cfg(feature = "remote")]
    Remote(RemoteError),
}

impl fmt::Display for TranscribeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local(error) => write!(f, "{error}"),
            #[cfg(feature = "remote")]
            Self::Remote(error) => write!(f, "{error}"),
        }
    }
}

impl StdError for TranscribeError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Local(error) => Some(error.as_ref()),
            #[cfg(feature = "remote")]
            Self::Remote(error) => Some(error),
        }
    }
}

#[cfg(feature = "remote")]
pub fn remote_config(
    endpoint: impl Into<String>,
    api_key: impl Into<String>,
    model: Option<String>,
) -> RemoteConfig {
    RemoteConfig {
        endpoint: endpoint.into(),
        api_key: api_key.into(),
        model: model.filter(|value| !value.trim().is_empty()),
    }
}

pub enum SpeechProvider {
    Local(Arc<SpeechService>),
    #[cfg(feature = "remote")]
    Remote(RemoteUpstream),
}

#[cfg(feature = "remote")]
pub struct RemoteUpstream {
    engine: RemoteEngine,
    default_model: Option<String>,
    fallback: Option<Arc<SpeechProvider>>,
}

#[cfg(feature = "remote")]
impl RemoteUpstream {
    pub fn new(
        client: Client,
        config: RemoteConfig,
        fallback: Option<Arc<SpeechProvider>>,
    ) -> Self {
        let default_model = config.model.clone();
        Self {
            engine: RemoteEngine::new(client, config),
            default_model,
            fallback,
        }
    }
}

#[cfg(feature = "remote")]
pub fn build_remote_provider(
    client: Client,
    config: RemoteConfig,
    local: Arc<SpeechService>,
) -> SpeechProvider {
    SpeechProvider::Remote(RemoteUpstream::new(
        client,
        config,
        Some(Arc::new(SpeechProvider::Local(local))),
    ))
}

impl SpeechProvider {
    pub async fn transcribe(
        &self,
        request: TranscribeRequest,
    ) -> Result<Transcription, TranscribeError> {
        match self {
            Self::Local(service) => {
                let service = Arc::clone(service);
                tokio::task::spawn_blocking(move || service.transcribe(request))
                    .await
                    .map_err(|err| {
                        TranscribeError::Local(anyhow!("transcription task failed: {err}"))
                    })?
                    .map_err(TranscribeError::Local)
            }
            #[cfg(feature = "remote")]
            Self::Remote(upstream) => Box::pin(upstream.transcribe(request)).await,
        }
    }

    pub async fn remote_model_ids(&self) -> Option<Result<Vec<String>, TranscribeError>> {
        match self {
            Self::Local(_) => None,
            #[cfg(feature = "remote")]
            Self::Remote(upstream) => Some(upstream.remote_model_ids().await),
        }
    }
}

#[cfg(feature = "remote")]
impl RemoteUpstream {
    async fn remote_model_ids(&self) -> Result<Vec<String>, TranscribeError> {
        if let Some(model) = self
            .default_model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Ok(vec![model.to_string()]);
        }

        self.engine
            .list_models()
            .await
            .map_err(TranscribeError::Remote)
    }

    async fn transcribe(
        &self,
        request: TranscribeRequest,
    ) -> Result<Transcription, TranscribeError> {
        let audio_path = match &request.audio {
            AudioInput::WavPath(path) => path.clone(),
            _ => {
                return match &self.fallback {
                    Some(fallback) => match local_fallback_request(fallback, request) {
                        Some(request) => Box::pin(fallback.transcribe(request)).await,
                        None => Err(TranscribeError::Local(anyhow!(
                            "No local transcription model is installed for fallback"
                        ))),
                    },
                    None => Err(TranscribeError::Remote(crate::remote::config_error(
                        "Remote provider requires an audio file upload",
                    ))),
                };
            }
        };

        let model = self
            .default_model
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(request.model_id.as_str());

        let result = self
            .engine
            .transcribe_file(
                &audio_path,
                RemoteRequestParams {
                    model,
                    language: request.language.as_deref(),
                    dictionary: &request.dictionary,
                    prompt: request.prompt.as_deref(),
                    timestamps: request.timestamps,
                    timestamp_granularity: request.timestamp_granularity,
                },
            )
            .await;

        match result {
            Ok(response) => Ok(Transcription {
                model_id: format!("remote:{}", response.model_id),
                language: response.language.or(request.language),
                ..response
            }),
            Err(err) if self.fallback.is_some() && err.should_fallback() => {
                let fallback = self.fallback.as_ref().expect("checked above");
                eprintln!(
                    "Remote speech temporarily unavailable, falling back to local: {}",
                    err.user_message()
                );
                match local_fallback_request(fallback, request) {
                    Some(request) => Box::pin(fallback.transcribe(request)).await,
                    None => Err(TranscribeError::Local(anyhow!(
                        "No local transcription model is installed for fallback"
                    ))),
                }
            }
            Err(err) => Err(TranscribeError::Remote(err)),
        }
    }
}

#[cfg(feature = "remote")]
fn local_fallback_request(
    fallback: &SpeechProvider,
    mut request: TranscribeRequest,
) -> Option<TranscribeRequest> {
    let SpeechProvider::Local(service) = fallback else {
        return None;
    };
    request.model_id = installed_model_id(service, &request.model_id)?;
    Some(request)
}

#[cfg(feature = "remote")]
fn installed_model_id(service: &SpeechService, preferred: &str) -> Option<String> {
    service.resolve(preferred).ok().map(|model| model.id)
}
