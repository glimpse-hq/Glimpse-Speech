use std::{
    fmt, fs, io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context, Result};
use reqwest::{header::RANGE, Client, StatusCode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

pub const MODEL_CAPABILITY_DICTIONARY_PROMPT: &str = "dictionary_prompt";
pub const MODEL_CAPABILITY_TIMESTAMPS: &str = "timestamps";
pub const MODEL_CAPABILITY_STREAMING: &str = "streaming";
pub const MODEL_CAPABILITY_DICTIONARY_BIASING: &str = "dictionary_biasing";

const MAX_STREAM_RETRIES: usize = 4;
const DOWNLOAD_REQUEST_TIMEOUT: Duration = Duration::from_secs(60 * 60 * 24);
const RETRY_BACKOFF_BASE_MS: u64 = 300;
const DEFAULT_APP_IDENTIFIER: &str = "com.glimpse.data";
const MODELS_DIR_NAME: &str = "models";

static MODEL_DOWNLOAD_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelEngine {
    Whisper,
    Parakeet,
    Nemotron,
}

impl fmt::Display for ModelEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelEngine::Whisper => write!(f, "whisper"),
            ModelEngine::Parakeet => write!(f, "parakeet"),
            ModelEngine::Nemotron => write!(f, "nemotron"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStorage {
    Directory,
    File { artifact: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelPlatform {
    MacosArm64,
    MacosX64,
    Windows,
    Linux,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    pub url: &'static str,
    pub path: &'static str,
    pub size_bytes: Option<u64>,
    pub sha256: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelManifest {
    pub id: &'static str,
    pub engine: ModelEngine,
    pub variant: &'static str,
    pub storage: ModelStorage,
    pub files: &'static [ModelFile],
    pub size_bytes: Option<u64>,
    pub capabilities: &'static [&'static str],
    pub supported_platforms: &'static [ModelPlatform],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStatus {
    pub id: String,
    pub installed: bool,
    pub bytes_on_disk: u64,
    pub missing_files: Vec<String>,
    pub directory: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    pub id: String,
    pub path: PathBuf,
    pub engine: ModelEngine,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDownloadProgress {
    pub model: String,
    pub file: String,
    pub downloaded: u64,
    pub total: u64,
    pub percent: f64,
}

pub type ProgressCallback<'a> = dyn Fn(ModelDownloadProgress) + Send + Sync + 'a;

#[derive(Default)]
pub struct InstallOptions<'a> {
    pub cancel_token: Option<CancellationToken>,
    pub progress: Option<&'a ProgressCallback<'a>>,
}

#[derive(Debug, Clone)]
pub struct ModelInstallManager {
    cache_dir: PathBuf,
    client: Client,
}

const SUPPORTED_ALL: &[ModelPlatform] = &[
    ModelPlatform::MacosArm64,
    ModelPlatform::MacosX64,
    ModelPlatform::Windows,
    ModelPlatform::Linux,
];

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
const SUPPORTED_NON_INTEL_MAC: &[ModelPlatform] = &[
    ModelPlatform::MacosArm64,
    ModelPlatform::Windows,
    ModelPlatform::Linux,
];

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
const PARAKEET_TDT_INT8_FILES: &[ModelFile] = &[
    ModelFile {
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx",
        path: "encoder-model.int8.onnx",
        size_bytes: Some(652_183_999),
        sha256: None,
    },
    ModelFile {
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.int8.onnx",
        path: "decoder_joint-model.int8.onnx",
        size_bytes: Some(18_202_004),
        sha256: None,
    },
    ModelFile {
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt",
        path: "vocab.txt",
        size_bytes: Some(93_939),
        sha256: None,
    },
];

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
const NEMOTRON_STREAMING_FILES: &[ModelFile] = &[
    ModelFile {
        url: "https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8/resolve/main/encoder.onnx",
        path: "encoder.onnx",
        size_bytes: Some(880_555_453),
        sha256: None,
    },
    ModelFile {
        url: "https://huggingface.co/altunenes/parakeet-rs/resolve/main/nemotron-speech-streaming-en-0.6b/encoder.onnx.data",
        path: "encoder.onnx.data",
        size_bytes: Some(2_436_567_040),
        sha256: None,
    },
    ModelFile {
        url: "https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8/resolve/main/decoder_joint.onnx",
        path: "decoder_joint.onnx",
        size_bytes: Some(10_962_697),
        sha256: None,
    },
    ModelFile {
        url: "https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8/resolve/main/tokenizer.model",
        path: "tokenizer.model",
        size_bytes: Some(251_056),
        sha256: None,
    },
];

const WHISPER_SMALL_Q5_FILES: &[ModelFile] = &[ModelFile {
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
    path: "ggml-small-q5_1.bin",
    size_bytes: Some(190_085_487),
    sha256: None,
}];

const WHISPER_LARGE_V3_TURBO_Q8_FILES: &[ModelFile] = &[ModelFile {
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
    path: "ggml-large-v3-turbo-q8_0.bin",
    size_bytes: Some(874_188_075),
    sha256: None,
}];

pub const MODEL_MANIFESTS: &[ModelManifest] = &[
    ModelManifest {
        id: "whisper_large_v3_turbo_q8",
        engine: ModelEngine::Whisper,
        variant: "Q8_0",
        storage: ModelStorage::File {
            artifact: "ggml-large-v3-turbo-q8_0.bin",
        },
        files: WHISPER_LARGE_V3_TURBO_Q8_FILES,
        size_bytes: Some(880_000_000),
        capabilities: &[
            MODEL_CAPABILITY_DICTIONARY_PROMPT,
            MODEL_CAPABILITY_TIMESTAMPS,
        ],
        supported_platforms: SUPPORTED_ALL,
    },
    #[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
    ModelManifest {
        id: "parakeet_tdt_int8",
        engine: ModelEngine::Parakeet,
        variant: "Int8",
        storage: ModelStorage::Directory,
        files: PARAKEET_TDT_INT8_FILES,
        size_bytes: Some(670_000_000),
        capabilities: &[MODEL_CAPABILITY_TIMESTAMPS],
        supported_platforms: SUPPORTED_NON_INTEL_MAC,
    },
    #[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
    ModelManifest {
        id: "nemotron_streaming_en",
        engine: ModelEngine::Nemotron,
        variant: "Int8",
        storage: ModelStorage::Directory,
        files: NEMOTRON_STREAMING_FILES,
        size_bytes: Some(895_000_000),
        capabilities: &[MODEL_CAPABILITY_STREAMING],
        supported_platforms: SUPPORTED_NON_INTEL_MAC,
    },
    ModelManifest {
        id: "whisper_small_q5",
        engine: ModelEngine::Whisper,
        variant: "Q5_1",
        storage: ModelStorage::File {
            artifact: "ggml-small-q5_1.bin",
        },
        files: WHISPER_SMALL_Q5_FILES,
        size_bytes: Some(190_000_000),
        capabilities: &[
            MODEL_CAPABILITY_DICTIONARY_PROMPT,
            MODEL_CAPABILITY_TIMESTAMPS,
        ],
        supported_platforms: SUPPORTED_ALL,
    },
];

pub fn list_models() -> &'static [ModelManifest] {
    MODEL_MANIFESTS
}

pub fn definition(id: &str) -> Option<&'static ModelManifest> {
    let id = resolve_model_alias(id);
    MODEL_MANIFESTS.iter().find(|manifest| manifest.id == id)
}

pub fn model_supports_capability(model_id: &str, capability: &str) -> bool {
    definition(model_id)
        .map(|manifest| {
            manifest
                .capabilities
                .iter()
                .any(|entry| entry.eq_ignore_ascii_case(capability))
        })
        .unwrap_or(false)
}

pub fn resolve_model_alias(id: &str) -> &str {
    match id {
        "whisper-1" | "gpt-4o-transcribe" | "gpt-4o-mini-transcribe" => "whisper_large_v3_turbo_q8",
        other => other,
    }
}

pub fn default_model_cache_dir() -> PathBuf {
    if let Ok(value) = std::env::var("GLIMPSE_SPEECH_CACHE_DIR") {
        return PathBuf::from(value);
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join(DEFAULT_APP_IDENTIFIER)
                .join(MODELS_DIR_NAME);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(app_data) = std::env::var("APPDATA").or_else(|_| std::env::var("LOCALAPPDATA")) {
            return PathBuf::from(app_data)
                .join(DEFAULT_APP_IDENTIFIER)
                .join(MODELS_DIR_NAME);
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        if let Ok(xdg_data_home) = std::env::var("XDG_DATA_HOME") {
            return PathBuf::from(xdg_data_home)
                .join(DEFAULT_APP_IDENTIFIER)
                .join(MODELS_DIR_NAME);
        }

        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join(DEFAULT_APP_IDENTIFIER)
                .join(MODELS_DIR_NAME);
        }
    }

    PathBuf::from(DEFAULT_APP_IDENTIFIER).join(MODELS_DIR_NAME)
}

impl ModelInstallManager {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            client: Client::new(),
        }
    }

    pub fn with_client(cache_dir: impl Into<PathBuf>, client: Client) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            client,
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn model_dir(&self, id: &str) -> PathBuf {
        self.cache_dir.join(resolve_model_alias(id))
    }

    pub fn artifact_path(&self, manifest: &ModelManifest) -> PathBuf {
        artifact_path(&self.model_dir(manifest.id), &manifest.storage)
    }

    pub fn model_status(&self, id: &str) -> Result<ModelStatus> {
        let manifest = definition(id).ok_or_else(|| anyhow!("Unknown model: {id}"))?;
        Ok(status_from_manifest(&self.model_dir(manifest.id), manifest))
    }

    pub fn resolve_model(&self, id: &str) -> Result<ResolvedModel> {
        let manifest = definition(id).ok_or_else(|| anyhow!("Unknown model: {id}"))?;
        let status = self.model_status(id)?;
        if !status.installed {
            return Err(anyhow!(
                "{id} is not fully installed. Missing: {}",
                status.missing_files.join(", ")
            ));
        }

        Ok(ResolvedModel {
            id: manifest.id.to_string(),
            path: self.artifact_path(manifest),
            engine: manifest.engine,
        })
    }

    pub fn verify_model(&self, id: &str) -> Result<ModelStatus> {
        let manifest = definition(id).ok_or_else(|| anyhow!("Unknown model: {id}"))?;
        let dir = self.model_dir(id);
        let status = status_from_manifest(&dir, manifest);
        if !status.missing_files.is_empty() {
            return Ok(status);
        }

        for file in manifest.files {
            let path = dir.join(file.path);
            if let Some(expected_size) = file.size_bytes {
                let actual_size = fs::metadata(&path)
                    .with_context(|| format!("read metadata for {}", file.path))?
                    .len();
                if actual_size != expected_size {
                    return Err(anyhow!(
                        "{} has unexpected size: expected {}, got {}",
                        file.path,
                        expected_size,
                        actual_size
                    ));
                }
            }

            if let Some(expected_sha256) = file.sha256 {
                let actual =
                    sha256_file(&path).with_context(|| format!("checksum {}", path.display()))?;
                if !actual.eq_ignore_ascii_case(expected_sha256) {
                    return Err(anyhow!(
                        "{} has unexpected sha256: expected {}, got {}",
                        file.path,
                        expected_sha256,
                        actual
                    ));
                }
            }
        }

        Ok(status)
    }

    pub async fn install_model(
        &self,
        id: &str,
        options: InstallOptions<'_>,
    ) -> Result<ModelStatus> {
        let manifest = definition(id).ok_or_else(|| anyhow!("Unknown model: {id}"))?;
        let dir = self.model_dir(id);
        tokio::fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("create model directory {}", dir.display()))?;

        for file in manifest.files {
            self.download_file(manifest, file, &dir, &options).await?;
        }

        self.verify_model(id)
    }

    pub fn delete_model(&self, id: &str) -> Result<ModelStatus> {
        let manifest = definition(id).ok_or_else(|| anyhow!("Unknown model: {id}"))?;
        let dir = self.model_dir(id);
        if dir.exists() {
            fs::remove_dir_all(&dir)
                .with_context(|| format!("remove model directory {}", dir.display()))?;
        }
        Ok(status_from_manifest(&dir, manifest))
    }

    async fn download_file(
        &self,
        manifest: &ModelManifest,
        file: &ModelFile,
        target_dir: &Path,
        options: &InstallOptions<'_>,
    ) -> Result<()> {
        let target_path = target_dir.join(file.path);
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        if manifest_file_ready(target_dir, file) {
            return Ok(());
        }

        let replace_existing = target_path.exists();
        let download_path = if replace_existing {
            replacement_download_path(&target_path)
        } else {
            target_path.clone()
        };
        let mut downloaded = if replace_existing {
            0
        } else {
            fs::metadata(&download_path).map(|m| m.len()).unwrap_or(0)
        };
        let mut total_size: u64 = file.size_bytes.unwrap_or(0);
        let mut retries = 0usize;
        let mut resume_supported = !replace_existing;

        loop {
            if is_cancelled(options) {
                let _ = fs::remove_file(&download_path);
                return Err(anyhow!("Download cancelled"));
            }

            if !resume_supported && downloaded > 0 {
                downloaded = 0;
                total_size = file.size_bytes.unwrap_or(0);
                let _ = fs::remove_file(&download_path);
            }

            let mut request = self.client.get(file.url).timeout(DOWNLOAD_REQUEST_TIMEOUT);
            if resume_supported && downloaded > 0 {
                request = request.header(RANGE, format!("bytes={downloaded}-"));
            }

            let mut response = request
                .send()
                .await
                .with_context(|| format!("download {}", file.path))?;

            if resume_supported && downloaded > 0 && response.status() == StatusCode::OK {
                downloaded = 0;
                total_size = file.size_bytes.unwrap_or(0);
                let _ = fs::remove_file(&download_path);
                resume_supported = false;
                continue;
            }

            if !response.status().is_success() {
                if resume_supported
                    && downloaded > 0
                    && response.status() == StatusCode::RANGE_NOT_SATISFIABLE
                {
                    downloaded = 0;
                    total_size = file.size_bytes.unwrap_or(0);
                    let _ = fs::remove_file(&download_path);
                    resume_supported = false;
                    continue;
                }

                if response.status().is_server_error()
                    || response.status() == StatusCode::TOO_MANY_REQUESTS
                {
                    if !can_retry(&mut retries) {
                        return Err(anyhow!(
                            "Download failed with status {} while fetching {}",
                            response.status(),
                            file.path
                        ));
                    }
                    wait_before_retry(retries).await;
                    continue;
                }

                return Err(anyhow!(
                    "Download failed with status {} while fetching {}",
                    response.status(),
                    file.path
                ));
            }

            let response_size = response.content_length().unwrap_or(0);
            if response_size > 0 {
                total_size = if downloaded > 0 {
                    downloaded.saturating_add(response_size)
                } else {
                    response_size
                };
            }

            let mut output = if downloaded > 0 {
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&download_path)
                    .await
                    .with_context(|| format!("open partial file {}", download_path.display()))?
            } else {
                tokio::fs::File::create(&download_path)
                    .await
                    .with_context(|| format!("create file {}", download_path.display()))?
            };

            loop {
                if is_cancelled(options) {
                    drop(output);
                    let _ = fs::remove_file(&download_path);
                    return Err(anyhow!("Download cancelled"));
                }

                match response.chunk().await {
                    Ok(Some(chunk)) => {
                        output.write_all(&chunk).await?;
                        downloaded += chunk.len() as u64;
                        emit_progress(manifest.id, file.path, downloaded, total_size, options);
                    }
                    Ok(None) => {
                        if total_size > 0 && downloaded < total_size {
                            break;
                        }
                        output.flush().await?;
                        drop(output);
                        if replace_existing {
                            replace_existing_file(&download_path, &target_path).with_context(
                                || format!("replace model file {}", target_path.display()),
                            )?;
                        }
                        return Ok(());
                    }
                    Err(err) => {
                        if !can_retry(&mut retries) {
                            let _ = fs::remove_file(&download_path);
                            return Err(anyhow!(
                                "Network interrupted while downloading {}",
                                file.path
                            )
                            .context(err));
                        }
                        wait_before_retry(retries).await;
                        break;
                    }
                }
            }
        }
    }
}

fn artifact_path(dir: &Path, storage: &ModelStorage) -> PathBuf {
    match storage {
        ModelStorage::Directory => dir.to_path_buf(),
        ModelStorage::File { artifact } => dir.join(artifact),
    }
}

fn status_from_manifest(dir: &Path, manifest: &ModelManifest) -> ModelStatus {
    let missing_files = missing_files(dir, manifest);
    let installed = missing_files.is_empty() && dir.exists();
    let bytes_on_disk = if dir.exists() {
        calculate_dir_size(dir).unwrap_or(0)
    } else {
        0
    };

    ModelStatus {
        id: manifest.id.to_string(),
        installed,
        bytes_on_disk,
        missing_files,
        directory: artifact_path(dir, &manifest.storage).display().to_string(),
    }
}

fn missing_files(dir: &Path, manifest: &ModelManifest) -> Vec<String> {
    manifest
        .files
        .iter()
        .filter_map(|file| {
            if manifest_file_ready(dir, file) {
                None
            } else {
                Some(file.path.to_string())
            }
        })
        .collect()
}

fn manifest_file_ready(dir: &Path, file: &ModelFile) -> bool {
    let path = dir.join(file.path);
    if !path.is_file() {
        return false;
    }

    if let Some(expected_size) = file.size_bytes {
        let Ok(actual_size) = fs::metadata(&path).map(|metadata| metadata.len()) else {
            return false;
        };
        if actual_size != expected_size {
            return false;
        }
    }

    if let Some(expected_sha256) = file.sha256 {
        let Ok(actual) = sha256_file(&path) else {
            return false;
        };
        if !actual.eq_ignore_ascii_case(expected_sha256) {
            return false;
        }
    }

    true
}

fn replacement_download_path(path: &Path) -> PathBuf {
    sibling_temp_path(path, "download")
}

fn replacement_backup_path(path: &Path) -> PathBuf {
    sibling_temp_path(path, "backup")
}

fn sibling_temp_path(path: &Path, purpose: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let sequence = MODEL_DOWNLOAD_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut replacement = path.to_path_buf();
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    replacement.set_file_name(format!(
        "{file_name}.{purpose}-{}-{timestamp}-{sequence}",
        std::process::id(),
    ));
    replacement
}

fn replace_existing_file(source: &Path, target: &Path) -> io::Result<()> {
    if !target.exists() {
        return fs::rename(source, target);
    }

    let backup = replacement_backup_path(target);
    fs::rename(target, &backup)?;
    match fs::rename(source, target) {
        Ok(()) => {
            let _ = fs::remove_file(&backup);
            Ok(())
        }
        Err(rename_error) => {
            if let Err(restore_error) = fs::rename(&backup, target) {
                return Err(io::Error::other(format!(
                    "failed to replace {}: {rename_error}; also failed to restore backup {}: {restore_error}",
                    target.display(),
                    backup.display()
                )));
            }
            Err(rename_error)
        }
    }
}

fn calculate_dir_size(dir: &Path) -> io::Result<u64> {
    let mut total = 0u64;
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                total += calculate_dir_size(&entry.path())?;
            } else {
                total += metadata.len();
            }
        }
    }
    Ok(total)
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let bytes = fs::read(path)?;
    let digest = Sha256::digest(bytes);
    Ok(hex_encode(&digest))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn can_retry(retries: &mut usize) -> bool {
    *retries = retries.saturating_add(1);
    *retries <= MAX_STREAM_RETRIES
}

async fn wait_before_retry(retries: usize) {
    let delay_ms = RETRY_BACKOFF_BASE_MS.saturating_mul(retries as u64);
    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
}

fn is_cancelled(options: &InstallOptions<'_>) -> bool {
    options
        .cancel_token
        .as_ref()
        .map(|token| token.is_cancelled())
        .unwrap_or(false)
}

fn emit_progress(
    model: &str,
    file: &str,
    downloaded: u64,
    total: u64,
    options: &InstallOptions<'_>,
) {
    let Some(progress) = options.progress else {
        return;
    };

    let percent = if total > 0 {
        ((downloaded as f64 / total as f64) * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };

    progress(ModelDownloadProgress {
        model: model.to_string(),
        file: file.to_string(),
        downloaded,
        total,
        percent,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_lookup_preserves_stable_ids() {
        assert_eq!(
            definition("whisper_small_q5").map(|model| model.engine),
            Some(ModelEngine::Whisper)
        );
        assert_eq!(
            definition("whisper-1").map(|model| model.id),
            Some("whisper_large_v3_turbo_q8")
        );
        assert!(definition("missing").is_none());
    }

    #[test]
    fn model_status_reports_missing_files() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-status-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let manager = ModelInstallManager::new(&root);

        let status = manager.model_status("whisper_small_q5").unwrap();
        assert!(!status.installed);
        assert_eq!(
            status.missing_files,
            vec!["ggml-small-q5_1.bin".to_string()]
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn partial_model_file_is_not_installed() {
        let root = std::env::temp_dir().join(format!(
            "glimpse-speech-partial-status-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let manager = ModelInstallManager::new(&root);
        let dir = manager.model_dir("whisper_small_q5");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("ggml-small-q5_1.bin"), b"partial").unwrap();

        let status = manager.model_status("whisper_small_q5").unwrap();
        assert!(!status.installed);
        assert_eq!(
            status.missing_files,
            vec!["ggml-small-q5_1.bin".to_string()]
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn replace_existing_file_restores_target_on_failed_swap() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-replace-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let target = root.join("model.bin");
        let missing_source = root.join("download.bin");
        fs::write(&target, b"old").unwrap();

        let result = replace_existing_file(&missing_source, &target);

        assert!(result.is_err());
        assert_eq!(fs::read(&target).unwrap(), b"old");
        assert!(fs::read_dir(&root).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".backup-")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_model_uses_file_artifact_path() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-resolve-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let manager = ModelInstallManager::new(&root);
        let dir = manager.model_dir("whisper_small_q5");
        fs::create_dir_all(&dir).unwrap();
        let artifact = dir.join("ggml-small-q5_1.bin");
        fs::File::create(&artifact)
            .unwrap()
            .set_len(190_085_487)
            .unwrap();

        let resolved = manager.resolve_model("whisper_small_q5").unwrap();
        assert_eq!(resolved.engine, ModelEngine::Whisper);
        assert_eq!(resolved.path, dir.join("ggml-small-q5_1.bin"));

        let _ = fs::remove_dir_all(&root);
    }
}
