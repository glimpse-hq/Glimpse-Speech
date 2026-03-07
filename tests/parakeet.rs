#![cfg(all(
    feature = "parakeet",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]

use glimpse_speech::engines::parakeet::{ParakeetModelParams, QuantizationType};

#[test]
fn int8_constructor_sets_int8_quantization() {
    let params = ParakeetModelParams::int8();
    assert_eq!(params.quantization, QuantizationType::Int8);
}

#[test]
fn fp32_constructor_sets_fp32_quantization() {
    let params = ParakeetModelParams::fp32();
    assert_eq!(params.quantization, QuantizationType::FP32);
}
