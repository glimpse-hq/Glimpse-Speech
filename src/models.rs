use std::{
    fmt, fs, io,
    path::Component,
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

const MAX_STREAM_RETRIES: usize = 4;
const DOWNLOAD_REQUEST_TIMEOUT: Duration = Duration::from_secs(60 * 60 * 24);
const RETRY_BACKOFF_BASE_MS: u64 = 300;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStorage {
    Directory,
    File { artifact: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteFile {
    pub url: String,
    pub path: String,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstallSpec {
    pub id: String,
    pub engine: ModelEngine,
    pub storage: ModelStorage,
    pub files: Vec<RemoteFile>,
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
        self.cache_dir.join(id)
    }

    pub fn artifact_path(&self, spec: &InstallSpec) -> PathBuf {
        artifact_path(&self.model_dir(&spec.id), &spec.storage)
    }

    pub fn status(&self, spec: &InstallSpec) -> Result<ModelStatus> {
        validate_spec(spec)?;
        Ok(status_from_spec(&self.model_dir(&spec.id), spec))
    }

    pub fn resolve(&self, spec: &InstallSpec) -> Result<ResolvedModel> {
        validate_spec(spec)?;
        if spec.engine != ModelEngine::Whisper {
            let status = self.status(spec)?;
            if !status.installed {
                return Err(anyhow!(
                    "{} is not fully installed. Missing: {}",
                    spec.id,
                    status.missing_files.join(", ")
                ));
            }
        }

        Ok(ResolvedModel {
            id: spec.id.clone(),
            path: self.artifact_path(spec),
            engine: spec.engine,
        })
    }

    pub fn resolve_loose(&self, reference: &str, engine: ModelEngine) -> Result<ResolvedModel> {
        let path = match engine {
            ModelEngine::Whisper => [PathBuf::from(reference), self.cache_dir.join(reference)]
                .into_iter()
                .find(|candidate| candidate.is_file())
                .or_else(|| single_file_in_dir(&self.cache_dir.join(reference)))
                .ok_or_else(|| anyhow!("Unknown model: {reference}"))?,
            ModelEngine::Parakeet | ModelEngine::Nemotron => {
                [PathBuf::from(reference), self.cache_dir.join(reference)]
                    .into_iter()
                    .find(|candidate| candidate.is_dir())
                    .ok_or_else(|| anyhow!("{engine} models require a directory: {reference}"))?
            }
        };
        Ok(ResolvedModel {
            id: reference.to_string(),
            path,
            engine,
        })
    }

    pub fn verify(&self, spec: &InstallSpec) -> Result<ModelStatus> {
        validate_spec(spec)?;
        let dir = self.model_dir(&spec.id);
        let status = status_from_spec(&dir, spec);
        if !status.missing_files.is_empty() {
            return Ok(status);
        }

        for file in &spec.files {
            let path = dir.join(&file.path);
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

            if let Some(expected_sha256) = &file.sha256 {
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

    pub async fn install(
        &self,
        spec: &InstallSpec,
        options: InstallOptions<'_>,
    ) -> Result<ModelStatus> {
        validate_spec(spec)?;
        let dir = self.model_dir(&spec.id);
        tokio::fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("create model directory {}", dir.display()))?;

        for file in &spec.files {
            self.download_file(&spec.id, file, &dir, &options).await?;
        }

        match self.verify(spec) {
            Ok(status) => Ok(status),
            Err(err) => {
                let _ = fs::remove_dir_all(&dir);
                Err(err)
            }
        }
    }

    pub fn delete(&self, id: &str) -> Result<ModelStatus> {
        validate_model_id(id)?;
        let dir = self.model_dir(id);
        if dir.exists() {
            fs::remove_dir_all(&dir)
                .with_context(|| format!("remove model directory {}", dir.display()))?;
        }
        Ok(ModelStatus {
            id: id.to_string(),
            installed: false,
            bytes_on_disk: 0,
            missing_files: Vec::new(),
            directory: dir.display().to_string(),
        })
    }

    async fn download_file(
        &self,
        model_id: &str,
        file: &RemoteFile,
        target_dir: &Path,
        options: &InstallOptions<'_>,
    ) -> Result<()> {
        let target_path = target_dir.join(&file.path);
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        if file_ready(target_dir, file) {
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

            let mut request = self.client.get(&file.url).timeout(DOWNLOAD_REQUEST_TIMEOUT);
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
                        emit_progress(model_id, &file.path, downloaded, total_size, options);
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

fn validate_spec(spec: &InstallSpec) -> Result<()> {
    validate_model_id(&spec.id)?;
    match &spec.storage {
        ModelStorage::Directory => {}
        ModelStorage::File { artifact } => validate_relative_file_path(artifact)?,
    }
    for file in &spec.files {
        validate_relative_file_path(&file.path)?;
    }
    Ok(())
}

fn validate_model_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id == "."
        || id == ".."
        || !id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        return Err(anyhow!("Invalid model id: {id}"));
    }
    Ok(())
}

fn validate_relative_file_path(path: &str) -> Result<()> {
    if path.is_empty() {
        return Err(anyhow!("Invalid model file path: {path}"));
    }

    let path = Path::new(path);
    if path.is_absolute() {
        return Err(anyhow!("Invalid model file path: {}", path.display()));
    }

    let valid = path
        .components()
        .all(|component| matches!(component, Component::Normal(_)));
    if !valid {
        return Err(anyhow!("Invalid model file path: {}", path.display()));
    }

    Ok(())
}

fn artifact_path(dir: &Path, storage: &ModelStorage) -> PathBuf {
    match storage {
        ModelStorage::Directory => dir.to_path_buf(),
        ModelStorage::File { artifact } => dir.join(artifact),
    }
}

fn single_file_in_dir(dir: &Path) -> Option<PathBuf> {
    let mut files = fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && !path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with('.'))
        });
    let first = files.next()?;
    files.next().is_none().then_some(first)
}

fn status_from_spec(dir: &Path, spec: &InstallSpec) -> ModelStatus {
    let missing_files = missing_files(dir, spec);
    let installed = missing_files.is_empty() && dir.exists();
    let bytes_on_disk = if dir.exists() {
        calculate_dir_size(dir).unwrap_or(0)
    } else {
        0
    };

    ModelStatus {
        id: spec.id.clone(),
        installed,
        bytes_on_disk,
        missing_files,
        directory: artifact_path(dir, &spec.storage).display().to_string(),
    }
}

fn missing_files(dir: &Path, spec: &InstallSpec) -> Vec<String> {
    spec.files
        .iter()
        .filter_map(|file| {
            if file_ready(dir, file) {
                None
            } else {
                Some(file.path.clone())
            }
        })
        .collect()
}

fn file_ready(dir: &Path, file: &RemoteFile) -> bool {
    let path = dir.join(&file.path);
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

    fn whisper_spec(id: &str, artifact: &str, size: Option<u64>) -> InstallSpec {
        InstallSpec {
            id: id.to_string(),
            engine: ModelEngine::Whisper,
            storage: ModelStorage::File {
                artifact: artifact.to_string(),
            },
            files: vec![RemoteFile {
                url: format!("https://example.test/{artifact}"),
                path: artifact.to_string(),
                size_bytes: size,
                sha256: None,
            }],
        }
    }

    #[test]
    fn status_reports_missing_files() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-status-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let manager = ModelInstallManager::new(&root);
        let spec = whisper_spec("whisper_small", "ggml-small.bin", Some(190_085_487));

        let status = manager.status(&spec).unwrap();
        assert!(!status.installed);
        assert_eq!(status.missing_files, vec!["ggml-small.bin".to_string()]);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn partial_file_is_not_installed() {
        let root = std::env::temp_dir().join(format!(
            "glimpse-speech-partial-status-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let manager = ModelInstallManager::new(&root);
        let spec = whisper_spec("whisper_small", "ggml-small.bin", Some(190_085_487));
        let dir = manager.model_dir(&spec.id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("ggml-small.bin"), b"partial").unwrap();

        let status = manager.status(&spec).unwrap();
        assert!(!status.installed);
        assert_eq!(status.missing_files, vec!["ggml-small.bin".to_string()]);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_unsafe_model_ids() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-unsafe-id-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let manager = ModelInstallManager::new(&root);
        let spec = whisper_spec("../escape", "ggml.bin", None);

        assert!(manager.status(&spec).is_err());
        assert!(manager.delete("../escape").is_err());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_unsafe_model_file_paths() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-unsafe-file-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let manager = ModelInstallManager::new(&root);
        let mut spec = whisper_spec("whisper_safe", "../ggml.bin", None);

        assert!(manager.status(&spec).is_err());
        spec.storage = ModelStorage::File {
            artifact: "ggml.bin".to_string(),
        };
        spec.files[0].path = "/tmp/ggml.bin".to_string();
        assert!(manager.status(&spec).is_err());

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
    fn resolve_accepts_whisper_artifact_with_unexpected_size() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-quant-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let manager = ModelInstallManager::new(&root);
        let spec = whisper_spec("whisper_turbo", "ggml-turbo.bin", Some(874_188_075));
        let dir = manager.model_dir(&spec.id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("ggml-turbo.bin"), b"ggml not the expected size").unwrap();

        let resolved = manager.resolve(&spec).unwrap();
        assert_eq!(resolved.engine, ModelEngine::Whisper);
        assert_eq!(resolved.path, dir.join("ggml-turbo.bin"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_loose_loads_arbitrary_whisper_file_by_path() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-loose-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manager = ModelInstallManager::new(&root);

        let model_path = root.join("ggml-arbitrary.bin");
        fs::write(&model_path, b"ggml arbitrary quant").unwrap();

        let by_path = manager
            .resolve_loose(model_path.to_str().unwrap(), ModelEngine::Whisper)
            .unwrap();
        assert_eq!(by_path.engine, ModelEngine::Whisper);
        assert_eq!(by_path.path, model_path);

        let by_name = manager
            .resolve_loose("ggml-arbitrary.bin", ModelEngine::Whisper)
            .unwrap();
        assert_eq!(by_name.path, model_path);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_loose_rejects_unknown_reference() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-unknown-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manager = ModelInstallManager::new(&root);

        assert!(manager
            .resolve_loose("totally-unknown", ModelEngine::Whisper)
            .is_err());
        assert!(manager
            .resolve_loose("/no/such/file/model.bin", ModelEngine::Whisper)
            .is_err());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_loose_loads_parakeet_directory_by_name() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-parakeet-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let manager = ModelInstallManager::new(&root);
        let dir = root.join("parakeet-local");
        fs::create_dir_all(&dir).unwrap();

        let resolved = manager
            .resolve_loose("parakeet-local", ModelEngine::Parakeet)
            .unwrap();
        assert_eq!(resolved.engine, ModelEngine::Parakeet);
        assert_eq!(resolved.path, dir);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_loose_loads_nemotron_directory_by_path() {
        let root =
            std::env::temp_dir().join(format!("glimpse-speech-nemotron-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manager = ModelInstallManager::new(&root);
        let dir = root.join("nemotron-local");
        fs::create_dir_all(&dir).unwrap();

        let resolved = manager
            .resolve_loose(dir.to_str().unwrap(), ModelEngine::Nemotron)
            .unwrap();
        assert_eq!(resolved.engine, ModelEngine::Nemotron);
        assert_eq!(resolved.path, dir);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_loose_rejects_directory_engine_file() {
        let root = std::env::temp_dir().join(format!(
            "glimpse-speech-directory-engine-file-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manager = ModelInstallManager::new(&root);
        let model_path = root.join("model.bin");
        fs::write(&model_path, b"not a directory").unwrap();

        assert!(manager
            .resolve_loose(model_path.to_str().unwrap(), ModelEngine::Parakeet)
            .is_err());

        let _ = fs::remove_dir_all(&root);
    }
}
