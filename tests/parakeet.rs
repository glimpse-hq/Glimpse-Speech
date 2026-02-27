#![cfg(feature = "parakeet")]

use glimpse_speech::engines::parakeet::{
    ParakeetArchitecture, ParakeetModelParams, QuantizationType,
};

#[test]
fn int8_constructor_sets_int8_quantization() {
    let params = ParakeetModelParams::int8();
    assert_eq!(params.architecture, ParakeetArchitecture::Tdt);
    assert_eq!(params.quantization, QuantizationType::Int8);
}

#[test]
fn fp32_constructor_sets_fp32_quantization() {
    let params = ParakeetModelParams::fp32();
    assert_eq!(params.architecture, ParakeetArchitecture::Tdt);
    assert_eq!(params.quantization, QuantizationType::FP32);
}

#[test]
fn ctc_constructor_sets_ctc_architecture() {
    let params = ParakeetModelParams::ctc();
    assert_eq!(params.architecture, ParakeetArchitecture::Ctc);
    assert_eq!(params.quantization, QuantizationType::FP32);
}
