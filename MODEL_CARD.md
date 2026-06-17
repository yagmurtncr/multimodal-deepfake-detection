# Model Card — Multimodal Deepfake Detection

## Model Summary

This project implements a multimodal deepfake detector for FakeAVCeleb v1.2 videos. It combines:

- visual face-frame features from Xception,
- audio features from Wav2Vec 2.0,
- lip/audio synchrony features from a lightweight 3D+2D CNN stream,
- three task heads: `video_fake`, `audio_fake`, and `any_fake`.

The model is intended for research and defensive analysis, not for biometric identification or automated moderation decisions.

## Intended Use

Appropriate uses:

- research on multimodal deepfake detection,
- educational demos,
- comparing modality contributions through ablation,
- prototyping defensive analysis workflows.

Out-of-scope uses:

- final legal, hiring, moderation, or law-enforcement decisions,
- identifying people,
- generating or improving deepfakes,
- making claims about real-world performance without cross-dataset validation.

## Training Data

The model was trained on FakeAVCeleb v1.2 using a subject-disjoint split. Video clips and trained weights are not distributed in this repository because the dataset is research-licensed and includes real people.

## Evaluation Summary

Baseline v3 test metrics:

| Head | AUC | F1 | Recall |
|---|---:|---:|---:|
| video | 0.9991 | 0.9994 | 0.9987 |
| audio | 0.9982 | 0.9875 | 0.9860 |
| any | 0.9994 | 0.9984 | 0.9972 |

The most important category result is `RealVideo-FakeAudio`, where image-only systems fail by design. The multimodal model detects this category with 0.9333 `any` recall.

## Limitations

- Results are measured on FakeAVCeleb v1.2; cross-dataset performance is not established.
- High test metrics may reflect dataset-specific artifacts from Wav2Lip/SV2TTS generation pipelines.
- The demo processes short clips and samples 8 frames, so very subtle or temporally sparse manipulations may be missed.
- Face detection failure prevents inference.
- Model confidence is not a calibrated probability of real-world truth.

## Ethical Considerations

Deepfake detection can produce false positives and false negatives. Use this model as an investigative signal, not as a final decision system. Human review, provenance checks, and cross-model validation are recommended before any sensitive use.

## Recommended Next Evaluations

- Cross-dataset tests on Celeb-DF, FaceForensics++, and DFDC samples.
- Calibration curves and threshold analysis.
- Robustness tests under compression, resizing, noise, and re-encoding.
- Explainability outputs for sampled frames, face crops, lip crops, and mel spectrograms.
