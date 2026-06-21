// Silero neural VAD. ort provides no prebuilt onnxruntime for Intel Mac, so the
// implementation is gated off that target; there `speech_regions` returns None
// and callers keep all detected speech.

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
pub use silero::speech_regions;

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub fn speech_regions(_samples: &[i16], _sample_rate: u32) -> Option<Vec<(f32, f32)>> {
    None
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
mod silero {
    use std::sync::Mutex;

    use anyhow::{Context, Result};
    use ort::session::Session;
    use ort::value::Tensor;

    const MODEL: &[u8] = include_bytes!("silero_vad_16k_op15.onnx");
    const WINDOW: usize = 512;
    const CONTEXT: usize = 64;
    const SPEECH_THRESHOLD: f32 = 0.5;
    const FRAME_S: f32 = WINDOW as f32 / 16_000.0; // 32 ms
    const BRIDGE_FRAMES: usize = 4; // merge speech across silence gaps up to ~128 ms
    const PAD_S: f32 = 0.25; // widen each region so words adjacent to speech survive

    struct SileroVad {
        session: Session,
        state: Vec<f32>,
        context: Vec<f32>,
        input: Vec<f32>,
    }

    impl SileroVad {
        fn new() -> Result<Self> {
            let session = Session::builder()?
                .commit_from_memory(MODEL)
                .context("load silero vad model")?;
            Ok(Self {
                session,
                state: vec![0.0; 2 * 128],
                context: vec![0.0; CONTEXT],
                input: vec![0.0; CONTEXT + WINDOW],
            })
        }

        // Per-frame speech mask for 16 kHz mono audio in [-1, 1].
        fn frame_mask(&mut self, samples: &[f32]) -> Result<Vec<bool>> {
            self.state.fill(0.0);
            self.context.fill(0.0);
            let mut mask = Vec::with_capacity(samples.len() / WINDOW + 1);
            for chunk in samples.chunks_exact(WINDOW) {
                self.input[..CONTEXT].copy_from_slice(&self.context);
                self.input[CONTEXT..].copy_from_slice(chunk);

                let input = Tensor::from_array(([1usize, CONTEXT + WINDOW], self.input.clone()))?;
                let state = Tensor::from_array(([2usize, 1, 128], self.state.clone()))?;
                let sr = Tensor::from_array(((), vec![16000i64]))?;

                let outputs = self
                    .session
                    .run(ort::inputs!["input" => input, "state" => state, "sr" => sr])?;
                let (_, prob) = outputs["output"].try_extract_tensor::<f32>()?;
                mask.push(prob[0] >= SPEECH_THRESHOLD);
                let (_, new_state) = outputs["stateN"].try_extract_tensor::<f32>()?;
                self.state.copy_from_slice(new_state);
                self.context.copy_from_slice(&chunk[WINDOW - CONTEXT..]);
            }
            Ok(mask)
        }
    }

    fn mask_to_regions(mask: &[bool]) -> Vec<(f32, f32)> {
        let mut regions: Vec<(usize, usize)> = Vec::new();
        let mut start: Option<usize> = None;
        let mut gap = 0usize;
        for (i, &speech) in mask.iter().enumerate() {
            if speech {
                if start.is_none() {
                    start = Some(i);
                }
                gap = 0;
            } else if let Some(s) = start {
                gap += 1;
                if gap > BRIDGE_FRAMES {
                    regions.push((s, i - gap + 1));
                    start = None;
                    gap = 0;
                }
            }
        }
        if let Some(s) = start {
            regions.push((s, mask.len() - gap));
        }
        regions
            .into_iter()
            .map(|(s, e)| {
                (
                    (s as f32 * FRAME_S - PAD_S).max(0.0),
                    e as f32 * FRAME_S + PAD_S,
                )
            })
            .collect()
    }

    fn resample_to_16k(samples: &[i16], sample_rate: u32) -> Vec<f32> {
        if sample_rate == 16_000 {
            return samples.iter().map(|&s| s as f32 / 32768.0).collect();
        }
        let ratio = 16_000.0 / sample_rate as f32;
        let out_len = (samples.len() as f32 * ratio) as usize;
        (0..out_len)
            .map(|i| {
                let src = i as f32 / ratio;
                let idx = src as usize;
                let frac = src - idx as f32;
                let a = samples.get(idx).copied().unwrap_or(0) as f32;
                let b = samples.get(idx + 1).copied().unwrap_or(0) as f32;
                (a + (b - a) * frac) / 32768.0
            })
            .collect()
    }

    static VAD: Mutex<Option<SileroVad>> = Mutex::new(None);

    /// Speech regions in seconds (padded), detected by the Silero neural VAD.
    /// Returns `None` if the model is unavailable so callers can fall back
    /// without dropping transcript text. Empty `Vec` means no speech.
    pub fn speech_regions(samples: &[i16], sample_rate: u32) -> Option<Vec<(f32, f32)>> {
        let audio = resample_to_16k(samples, sample_rate);
        if audio.len() < WINDOW {
            return Some(Vec::new());
        }
        let mut guard = VAD.lock().ok()?;
        if guard.is_none() {
            *guard = Some(SileroVad::new().ok()?);
        }
        let mask = guard.as_mut().unwrap().frame_mask(&audio).ok()?;
        Some(mask_to_regions(&mask))
    }
}
