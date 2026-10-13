// Silero neural VAD: a std-only forward pass of the bundled ONNX model. Only
// the model's weights are read from the file; the graph is reimplemented here
// and matches onnxruntime to float rounding.

use std::sync::OnceLock;

// silero-vad v6.2, sha256 7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49.
const MODEL: &[u8] = include_bytes!("silero_vad_16k_op15.onnx");
const WINDOW: usize = 512;
const CONTEXT: usize = 64;
const N_FFT: usize = 256;
const HOP: usize = 128;
const BINS: usize = N_FFT / 2 + 1;
const HIDDEN: usize = 128;
const SPEECH_THRESHOLD: f32 = 0.5;
const FRAME_S: f32 = WINDOW as f32 / 16_000.0; // 32 ms
const BRIDGE_FRAMES: usize = 4; // merge speech across silence gaps up to ~128 ms
const PAD_S: f32 = 0.25; // widen each region so words adjacent to speech survive
const MAX_THREADS: usize = 8;
const MIN_FRAMES_PER_THREAD: usize = 32;

// Conv1d with kernel 3, padding 1 and ReLU over time-major input [t][inputs].
struct Conv {
    weight: Vec<f32>, // [outputs][3][inputs]
    bias: Vec<f32>,
    inputs: usize,
    stride: usize,
}

impl Conv {
    fn new(weight: &[f32], bias: Vec<f32>, inputs: usize, stride: usize) -> Self {
        let outputs = bias.len();
        let mut transposed = vec![0.0; weight.len()];
        for o in 0..outputs {
            for i in 0..inputs {
                for k in 0..3 {
                    transposed[(o * 3 + k) * inputs + i] = weight[(o * inputs + i) * 3 + k];
                }
            }
        }
        Self {
            weight: transposed,
            bias,
            inputs,
            stride,
        }
    }

    fn run(&self, input: &[f32], output: &mut Vec<f32>) {
        let steps = input.len() / self.inputs;
        output.clear();
        for step in 0..(steps - 1) / self.stride + 1 {
            for (o, bias) in self.bias.iter().enumerate() {
                let mut acc = *bias;
                for k in 0..3 {
                    let Some(t) = (step * self.stride + k).checked_sub(1) else {
                        continue;
                    };
                    if t >= steps {
                        continue;
                    }
                    let row = (o * 3 + k) * self.inputs;
                    acc += dot(
                        &self.weight[row..row + self.inputs],
                        &input[t * self.inputs..(t + 1) * self.inputs],
                    );
                }
                output.push(acc.max(0.0));
            }
        }
    }
}

// The ONNX STFT basis is a periodic-Hann-windowed DFT, so the magnitudes come
// from an FFT of the windowed frame. Row 0 of the basis (frequency 0) is the
// window itself.
struct Stft {
    window: Vec<f32>,
    twiddles: Vec<(f32, f32)>, // exp(-2 pi i k / N_FFT), k < N_FFT / 2
    bit_reversed: Vec<usize>,
}

impl Stft {
    fn new(basis: &[f32]) -> Self {
        let bits = N_FFT.trailing_zeros();
        Self {
            window: basis[..N_FFT].to_vec(),
            twiddles: (0..N_FFT / 2)
                .map(|k| {
                    let angle = -2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64;
                    (angle.cos() as f32, angle.sin() as f32)
                })
                .collect(),
            bit_reversed: (0..N_FFT)
                .map(|i| i.reverse_bits() >> (usize::BITS - bits))
                .collect(),
        }
    }

    // Appends |X_k| for k in 0..BINS.
    fn magnitudes(&self, frame: &[f32], out: &mut Vec<f32>) {
        let mut re = [0.0f32; N_FFT];
        let mut im = [0.0f32; N_FFT];
        for (i, &j) in self.bit_reversed.iter().enumerate() {
            re[j] = frame[i] * self.window[i];
        }
        let mut half = 1;
        while half < N_FFT {
            let stride = N_FFT / (2 * half);
            for start in (0..N_FFT).step_by(2 * half) {
                for k in 0..half {
                    let (wr, wi) = self.twiddles[k * stride];
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            half *= 2;
        }
        out.extend((0..BINS).map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt()));
    }
}

struct Silero {
    stft: Stft,
    encoder: [Conv; 4],
    w_ih: Vec<f32>, // [4 * HIDDEN][HIDDEN], LSTM gates i, f, g, o
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    w_out: Vec<f32>,
    b_out: f32,
}

impl Silero {
    fn load() -> Option<Self> {
        let tensors = initializers(MODEL)?;
        let tensor = |name: &str, len: usize| -> Option<Vec<f32>> {
            let raw = tensors
                .iter()
                .find(|(tensor, _)| *tensor == name.as_bytes())?
                .1;
            (raw.len() == len * 4).then(|| {
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            })
        };
        let conv = |layer: usize, inputs: usize, outputs: usize, stride: usize| {
            let prefix = format!("model.encoder.{layer}.reparam_conv");
            let weight = tensor(&format!("{prefix}.weight"), outputs * inputs * 3)?;
            let bias = tensor(&format!("{prefix}.bias"), outputs)?;
            Some(Conv::new(&weight, bias, inputs, stride))
        };
        Some(Self {
            stft: Stft::new(&tensor(
                "model.stft.forward_basis_buffer",
                2 * BINS * N_FFT,
            )?),
            encoder: [
                conv(0, BINS, 128, 1)?,
                conv(1, 128, 64, 2)?,
                conv(2, 64, 64, 2)?,
                conv(3, 64, HIDDEN, 1)?,
            ],
            w_ih: tensor("model.decoder.rnn.weight_ih", 4 * HIDDEN * HIDDEN)?,
            w_hh: tensor("model.decoder.rnn.weight_hh", 4 * HIDDEN * HIDDEN)?,
            b_ih: tensor("model.decoder.rnn.bias_ih", 4 * HIDDEN)?,
            b_hh: tensor("model.decoder.rnn.bias_hh", 4 * HIDDEN)?,
            w_out: tensor("model.decoder.decoder.2.weight", HIDDEN)?,
            b_out: tensor("model.decoder.decoder.2.bias", 1)?[0],
        })
    }

    // Speech probability per 512-sample frame of 16 kHz mono audio in [-1, 1].
    // A trailing partial frame is dropped.
    fn frame_probs(&self, samples: &[f32]) -> Vec<f32> {
        let frames = samples.len() / WINDOW;
        if frames == 0 {
            return Vec::new();
        }
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(MAX_THREADS)
            .min(frames.div_ceil(MIN_FRAMES_PER_THREAD));
        let per_thread = frames.div_ceil(threads);
        let mut inputs = vec![0.0f32; frames * 4 * HIDDEN];
        std::thread::scope(|scope| {
            for (t, out) in inputs.chunks_mut(per_thread * 4 * HIDDEN).enumerate() {
                scope.spawn(move || {
                    let mut x = [0.0f32; CONTEXT + WINDOW + CONTEXT];
                    let (mut a, mut b) = (Vec::new(), Vec::new());
                    for (k, gates) in out.chunks_exact_mut(4 * HIDDEN).enumerate() {
                        let start = (t * per_thread + k) * WINDOW;
                        if start >= CONTEXT {
                            x[..CONTEXT].copy_from_slice(&samples[start - CONTEXT..start]);
                        }
                        x[CONTEXT..CONTEXT + WINDOW]
                            .copy_from_slice(&samples[start..start + WINDOW]);
                        self.input_gates(&mut x, &mut a, &mut b, gates);
                    }
                });
            }
        });
        let (mut h, mut c) = ([0.0f32; HIDDEN], [0.0f32; HIDDEN]);
        inputs
            .chunks_exact(4 * HIDDEN)
            .map(|gates| self.recur(&mut h, &mut c, gates))
            .collect()
    }

    fn step(&self, state: &mut State, chunk: &[f32]) -> f32 {
        let State { h, c, x, a, b } = state;
        x.copy_within(WINDOW..WINDOW + CONTEXT, 0);
        x[CONTEXT..CONTEXT + WINDOW].copy_from_slice(chunk);
        let mut gates = [0.0f32; 4 * HIDDEN];
        self.input_gates(x, a, b, &mut gates);
        self.recur(h, c, &gates)
    }

    // `x` holds the previous frame's last 64 samples, then the frame; a 64-sample
    // right reflection pad completes the STFT input, as the ONNX graph builds it.
    fn input_gates(
        &self,
        x: &mut [f32; CONTEXT + WINDOW + CONTEXT],
        a: &mut Vec<f32>,
        b: &mut Vec<f32>,
        gates: &mut [f32],
    ) {
        let end = CONTEXT + WINDOW;
        for i in 0..CONTEXT {
            x[end + i] = x[end - 2 - i];
        }
        a.clear();
        for frame in x.windows(N_FFT).step_by(HOP) {
            self.stft.magnitudes(frame, a);
        }
        for conv in &self.encoder {
            conv.run(a, b);
            std::mem::swap(a, b);
        }
        for (r, gate) in gates.iter_mut().enumerate() {
            let row = r * HIDDEN..(r + 1) * HIDDEN;
            *gate = self.b_ih[r] + self.b_hh[r] + dot(&self.w_ih[row], a);
        }
    }

    fn recur(&self, h: &mut [f32; HIDDEN], c: &mut [f32; HIDDEN], inputs: &[f32]) -> f32 {
        let mut gates = [0.0f32; 4 * HIDDEN];
        for (r, gate) in gates.iter_mut().enumerate() {
            *gate = inputs[r] + dot(&self.w_hh[r * HIDDEN..(r + 1) * HIDDEN], &h[..]);
        }
        for j in 0..HIDDEN {
            let input = sigmoid(gates[j]);
            let forget = sigmoid(gates[HIDDEN + j]);
            let cell = gates[2 * HIDDEN + j].tanh();
            let output = sigmoid(gates[3 * HIDDEN + j]);
            c[j] = forget * c[j] + input * cell;
            h[j] = output * c[j].tanh();
        }
        let logit = self
            .w_out
            .iter()
            .zip(h.iter())
            .fold(self.b_out, |acc, (w, v)| acc + w * v.max(0.0));
        sigmoid(logit)
    }
}

struct State {
    h: [f32; HIDDEN],
    c: [f32; HIDDEN],
    x: [f32; CONTEXT + WINDOW + CONTEXT],
    a: Vec<f32>,
    b: Vec<f32>,
}

impl State {
    fn new() -> Self {
        Self {
            h: [0.0; HIDDEN],
            c: [0.0; HIDDEN],
            x: [0.0; CONTEXT + WINDOW + CONTEXT],
            a: Vec::new(),
            b: Vec::new(),
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Eight independent accumulators so the compiler vectorizes the loop.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (lanes_a, lanes_b) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = lanes_a
        .remainder()
        .iter()
        .zip(lanes_b.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in lanes_a.zip(lanes_b) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

// Name and raw little-endian data of each tensor in an ONNX file's top-level
// graph, read straight from the protobuf: ModelProto.graph (7) ->
// GraphProto.initializer (5) -> TensorProto name (8), raw_data (9).
fn initializers(model: &[u8]) -> Option<Vec<(&[u8], &[u8])>> {
    let graph = fields(model).find_map(|(field, value)| (field == 7).then_some(value))?;
    let tensors = fields(graph)
        .filter(|(field, _)| *field == 5)
        .filter_map(|(_, tensor)| {
            let mut name = None;
            let mut raw = None;
            for (field, value) in fields(tensor) {
                match field {
                    8 => name = Some(value),
                    9 => raw = Some(value),
                    _ => {}
                }
            }
            Some((name?, raw?))
        })
        .collect();
    Some(tensors)
}

// Protobuf (field number, payload) pairs; varint values are skipped and yield
// an empty payload. Stops at the first malformed field.
fn fields(mut buf: &[u8]) -> impl Iterator<Item = (u64, &[u8])> {
    std::iter::from_fn(move || {
        let key = varint(&mut buf)?;
        let len = match key & 7 {
            0 => {
                varint(&mut buf)?;
                0
            }
            1 => 8,
            2 => usize::try_from(varint(&mut buf)?).ok()?,
            5 => 4,
            _ => return None,
        };
        let payload = buf.get(..len)?;
        buf = &buf[len..];
        Some((key >> 3, payload))
    })
}

fn varint(buf: &mut &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let (&byte, rest) = buf.split_first()?;
        *buf = rest;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

fn mask_to_regions(mask: &[bool]) -> Vec<(f32, f32)> {
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    let mut gap = 0usize;
    for (i, &speech) in mask.iter().enumerate() {
        if speech {
            start.get_or_insert(i);
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

static VAD: OnceLock<Option<Silero>> = OnceLock::new();

/// Speech regions in seconds (padded), detected by the Silero neural VAD.
/// Returns `None` if the model is unavailable so callers can fall back
/// without dropping transcript text. Empty `Vec` means no speech.
pub fn speech_regions(samples: &[i16], sample_rate: u32) -> Option<Vec<(f32, f32)>> {
    if sample_rate > crate::audio::MAX_SAMPLE_RATE {
        return None;
    }
    let audio = crate::audio::resample_i16_to_f32(samples, sample_rate, 16_000);
    if audio.len() < WINDOW {
        return Some(Vec::new());
    }
    let vad = VAD.get_or_init(Silero::load).as_ref()?;
    let mask: Vec<bool> = vad
        .frame_probs(&audio)
        .into_iter()
        .map(|p| p >= SPEECH_THRESHOLD)
        .collect();
    Some(mask_to_regions(&mask))
}

const REGION_EXIT: f32 = 0.35;
const REGION_PAD_FRAMES: usize = 16; // 512 ms
const STREAM_HISTORY_FRAMES: usize = 64;

/// Silero's speech probability for each 32 ms frame of a recording.
pub struct Speech {
    probs: Vec<f32>,
    samples: usize,
}

/// Runs Silero over a whole recording. `None` if the model is unavailable.
pub fn analyze(samples: &[i16], sample_rate: u32) -> Option<Speech> {
    if sample_rate > crate::audio::MAX_SAMPLE_RATE {
        return None;
    }
    let audio = crate::audio::resample_i16_to_f32(samples, sample_rate, 16_000);
    let vad = VAD.get_or_init(Silero::load).as_ref()?;
    Some(Speech {
        probs: vad.frame_probs(&audio),
        samples: audio.len(),
    })
}

impl Speech {
    /// Share of 32 ms frames that are speech, 0 to 100.
    pub fn percentage(&self) -> f32 {
        if self.probs.is_empty() {
            return 0.0;
        }
        let speech = self
            .probs
            .iter()
            .filter(|&&p| p >= SPEECH_THRESHOLD)
            .count();
        speech as f32 * 100.0 / self.probs.len() as f32
    }

    /// Whether each whole frame of `frame` 16 kHz samples is speech.
    pub fn frames(&self, frame: usize) -> Vec<bool> {
        let mask: Vec<bool> = self.probs.iter().map(|&p| p >= SPEECH_THRESHOLD).collect();
        on_frames(&mask, 0, 0, frame, self.samples / frame)
    }

    /// Speech for cutting audio: starts at 0.5, ends below 0.35, padded by
    /// about half a second on both sides.
    pub fn regions(&self, frame: usize) -> Vec<bool> {
        on_frames(
            &regions_mask(&self.probs),
            0,
            0,
            frame,
            self.samples / frame,
        )
    }
}

/// Silero over live 16 kHz audio. Positions count samples since the stream
/// was created or last reset.
pub struct SpeechStream {
    vad: &'static Silero,
    state: State,
    pending: Vec<f32>,
    probs: Vec<f32>,
    first: usize,
}

impl SpeechStream {
    pub fn new() -> Option<Self> {
        Some(Self {
            vad: VAD.get_or_init(Silero::load).as_ref()?,
            state: State::new(),
            pending: Vec::new(),
            probs: Vec::new(),
            first: 0,
        })
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);
        let whole = self.pending.len() / WINDOW * WINDOW;
        for chunk in self.pending[..whole].chunks_exact(WINDOW) {
            self.probs.push(self.vad.step(&mut self.state, chunk));
        }
        self.pending.drain(..whole);
    }

    /// [`Speech::regions`] from sample `from` on, as far as audio has arrived.
    /// Earlier frames can turn into speech as later audio arrives.
    pub fn regions(&self, from: u64, frame: usize) -> Vec<bool> {
        let end = (self.first + self.probs.len()) * WINDOW;
        let count = (end as u64).saturating_sub(from) as usize / frame;
        on_frames(&regions_mask(&self.probs), self.first, from, frame, count)
    }

    /// Drops analysis no longer needed for frames from `sample` on.
    pub fn forget_before(&mut self, sample: u64) {
        let keep_from = (sample as usize / WINDOW).saturating_sub(STREAM_HISTORY_FRAMES);
        let drop = keep_from.saturating_sub(self.first).min(self.probs.len());
        self.probs.drain(..drop);
        self.first += drop;
    }

    pub fn reset(&mut self) {
        self.state = State::new();
        self.pending.clear();
        self.probs.clear();
        self.first = 0;
    }
}

fn regions_mask(probs: &[f32]) -> Vec<bool> {
    let mut speech = false;
    let core: Vec<bool> = probs
        .iter()
        .map(|&p| {
            speech = p
                >= if speech {
                    REGION_EXIT
                } else {
                    SPEECH_THRESHOLD
                };
            speech
        })
        .collect();
    let mut mask = core.clone();
    for (i, _) in core.iter().enumerate().filter(|(_, s)| **s) {
        let lo = i.saturating_sub(REGION_PAD_FRAMES);
        let hi = (i + REGION_PAD_FRAMES + 1).min(mask.len());
        mask[lo..hi].fill(true);
    }
    mask
}

fn on_frames(mask: &[bool], first: usize, from: u64, frame: usize, count: usize) -> Vec<bool> {
    (0..count)
        .map(|k| {
            let mid = from as usize + k * frame + frame / 2;
            (mid / WINDOW)
                .checked_sub(first)
                .and_then(|i| mask.get(i))
                .copied()
                .unwrap_or(false)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        BRIDGE_FRAMES, FRAME_S, PAD_S, REGION_PAD_FRAMES, SPEECH_THRESHOLD, Silero, SpeechStream,
        State, WINDOW, analyze, mask_to_regions, regions_mask,
    };

    fn reference_audio() -> Vec<f32> {
        include_bytes!("vad_reference.pcm")
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32_768.0)
            .collect()
    }

    #[test]
    fn stream_matches_whole_recording() {
        let audio = reference_audio().repeat(12);
        let pcm: Vec<i16> = audio.iter().map(|v| (v * 32_768.0) as i16).collect();
        let whole = analyze(&pcm, 16_000).expect("silero weights");
        let mut stream = SpeechStream::new().expect("silero weights");
        for chunk in audio.chunks(480) {
            stream.push(chunk);
        }
        assert_eq!(
            stream.regions(0, 480),
            whole.regions(480)[..stream.regions(0, 480).len()]
        );
        assert_eq!(stream.regions(4800, 480), stream.regions(0, 480)[10..]);
        stream.forget_before(4800);
        assert_eq!(
            stream.regions(4800, 480),
            whole.regions(480)[10..10 + stream.regions(4800, 480).len()]
        );
    }

    #[test]
    fn parallel_frames_match_frame_by_frame() {
        let audio = reference_audio().repeat(12);
        let vad = Silero::load().expect("silero weights");
        let mut state = State::new();
        let stepped: Vec<f32> = audio
            .chunks_exact(WINDOW)
            .map(|chunk| vad.step(&mut state, chunk))
            .collect();
        assert_eq!(vad.frame_probs(&audio), stepped);
    }

    #[test]
    fn regions_hold_through_dips_and_pad_both_sides() {
        let mut probs = vec![0.1; 40];
        probs[20] = 0.9;
        probs[21] = 0.4;
        probs[22] = 0.2;
        let mask = regions_mask(&probs);
        assert!(mask[21] && mask[20 - REGION_PAD_FRAMES] && mask[21 + REGION_PAD_FRAMES]);
        assert!(!mask[20 - REGION_PAD_FRAMES - 1] && !mask[21 + REGION_PAD_FRAMES + 1]);
    }

    #[test]
    fn bridges_short_gaps_and_pads_regions() {
        let mut mask = vec![true; 10];
        mask.extend(std::iter::repeat_n(false, BRIDGE_FRAMES));
        mask.extend(std::iter::repeat_n(true, 5));
        mask.extend(std::iter::repeat_n(false, BRIDGE_FRAMES + 2));
        mask.extend(std::iter::repeat_n(true, 3));

        let regions = mask_to_regions(&mask);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].0, 0.0);
        assert!((regions[0].1 - (19.0 * FRAME_S + PAD_S)).abs() < 1e-6);
    }

    #[test]
    fn silence_yields_no_regions() {
        assert!(mask_to_regions(&[false; 20]).is_empty());
    }

    #[test]
    fn bundled_model_loads_and_rejects_silence() {
        let vad = Silero::load().expect("silero weights");
        let probs = vad.frame_probs(&[0.0; 4 * WINDOW]);
        assert_eq!(probs.len(), 4);
        assert!(probs.iter().all(|&p| p < SPEECH_THRESHOLD));
    }

    #[test]
    fn matches_onnxruntime_on_speech() {
        // First 24 frames of whisper.cpp's jfk.wav (s16le) and onnxruntime's
        // probabilities for them from the bundled model.
        const REFERENCE: [f32; 24] = [
            0.001670, 0.084632, 0.301986, 0.133493, 0.083199, 0.043454, 0.055422, 0.052693,
            0.031314, 0.037638, 0.343341, 0.945989, 0.932934, 0.857259, 0.972762, 0.988260,
            0.990164, 0.989361, 0.995614, 0.993969, 0.982173, 0.994392, 0.994515, 0.994812,
        ];
        let probs = Silero::load()
            .expect("silero weights")
            .frame_probs(&reference_audio());
        assert_eq!(probs.len(), REFERENCE.len());
        for (p, r) in probs.iter().zip(REFERENCE) {
            assert!((p - r).abs() < 1e-4, "{p} vs {r}");
        }
    }
}
