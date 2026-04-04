use crate::{TranscriptionResult, TranscriptionSegment};

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerDiarizationSegment {
    /// Segment start time in seconds.
    pub start: f32,
    /// Segment end time in seconds.
    pub end: f32,
    pub speaker_id: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiarizedTranscriptionSegment {
    /// Segment start time in seconds.
    pub start: f32,
    /// Segment end time in seconds.
    pub end: f32,
    pub text: String,
    /// `None` indicates that no diarization segment overlapped this transcription segment.
    pub speaker_id: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiarizedTranscriptionResult {
    pub text: String,
    pub segments: Vec<DiarizedTranscriptionSegment>,
    pub speaker_segments: Vec<SpeakerDiarizationSegment>,
}

pub fn attribute_speakers(
    transcription: TranscriptionResult,
    speaker_segments: Vec<SpeakerDiarizationSegment>,
) -> Result<DiarizedTranscriptionResult, Box<dyn std::error::Error>> {
    let segments = transcription
        .segments
        .ok_or_else(|| io_error("Transcription segments are required to attribute speakers."))?;

    Ok(DiarizedTranscriptionResult {
        text: transcription.text,
        segments: segments
            .into_iter()
            .map(|segment| map_segment(segment, &speaker_segments))
            .collect(),
        speaker_segments,
    })
}

fn map_segment(
    segment: TranscriptionSegment,
    speaker_segments: &[SpeakerDiarizationSegment],
) -> DiarizedTranscriptionSegment {
    DiarizedTranscriptionSegment {
        start: segment.start,
        end: segment.end,
        text: segment.text,
        speaker_id: find_speaker_id(segment.start, segment.end, speaker_segments),
    }
}

fn find_speaker_id(
    start: f32,
    end: f32,
    speaker_segments: &[SpeakerDiarizationSegment],
) -> Option<usize> {
    speaker_segments
        .iter()
        .filter_map(|segment| {
            let overlap_start = start.max(segment.start);
            let overlap_end = end.min(segment.end);
            let overlap = (overlap_end - overlap_start).max(0.0);
            if overlap > 0.0 {
                Some((segment.speaker_id, overlap))
            } else {
                None
            }
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(speaker_id, _)| speaker_id)
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}
