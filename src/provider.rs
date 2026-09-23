use std::{error::Error as StdError, fmt, sync::Arc};

use anyhow::anyhow;

use crate::Transcription;
use crate::service::{SpeechService, TranscribeRequest};

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
        let default_model = config
            .model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
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
        if let Some(model) = &self.default_model {
            return Ok(vec![model.clone()]);
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
        let AudioInput::WavPath(audio_path) = &request.audio else {
            return match &self.fallback {
                Some(fallback) => transcribe_via_fallback(fallback, request).await,
                None => Err(TranscribeError::Remote(crate::remote::config_error(
                    "Remote provider requires an audio file upload",
                ))),
            };
        };

        let model = self
            .default_model
            .as_deref()
            .unwrap_or(request.model_id.as_str());

        let result = self
            .engine
            .transcribe_file(
                audio_path,
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

        match (result, &self.fallback) {
            (Ok(response), _) => Ok(Transcription {
                model_id: format!("remote:{}", response.model_id),
                language: response.language.or(request.language),
                ..response
            }),
            (Err(err), Some(fallback)) if err.should_fallback() => {
                match local_fallback_request(fallback, request) {
                    Some(request) => {
                        tracing::warn!(
                            "Remote speech temporarily unavailable, falling back to local: {}",
                            err.user_message()
                        );
                        Box::pin(fallback.transcribe(request))
                            .await
                            .map_err(|local| {
                                tracing::warn!("Local fallback failed: {local:?}");
                                TranscribeError::Remote(err)
                            })
                    }
                    None => Err(TranscribeError::Remote(err)),
                }
            }
            (Err(err), _) => Err(TranscribeError::Remote(err)),
        }
    }
}

#[cfg(feature = "remote")]
async fn transcribe_via_fallback(
    fallback: &SpeechProvider,
    request: TranscribeRequest,
) -> Result<Transcription, TranscribeError> {
    match local_fallback_request(fallback, request) {
        Some(request) => Box::pin(fallback.transcribe(request)).await,
        None => Err(TranscribeError::Local(anyhow!(
            "No local transcription model is installed for fallback"
        ))),
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
    request.model_id = service.resolve(&request.model_id).ok()?.id;
    Some(request)
}
