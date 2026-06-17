# Architecture

## High-Level Flow

```text
video file
  ├─ frame sampling → face crops → Xception stream
  ├─ audio extraction → Wav2Vec 2.0 stream
  └─ lip crops + mel spectrogram → sync stream

stream embeddings
  └─ late fusion MLP
       ├─ video_fake head
       ├─ audio_fake head
       └─ any_fake head
```

## Components

### VideoPreprocessor

`demo_site/inference.py` and `notebook_v2/deepfake_v3.py` use the same preprocessing logic:

- sample 8 frames,
- detect a face on the middle frame with MTCNN,
- reuse the selected bounding box across sampled frames,
- crop full face frames for the visual stream,
- crop the lower face/lip region for the sync stream,
- extract 3 seconds of 16 kHz mono audio,
- compute a normalized mel spectrogram.

### ImageStream

The visual stream runs Xception over sampled face frames and aggregates temporal information through mean + standard deviation pooling.

### AudioStream

The audio stream uses frozen Wav2Vec 2.0 features. The pooled representation is projected into the shared embedding dimension.

### SyncStream

The sync stream combines:

- a small 3D CNN over lip crops,
- a small 2D CNN over mel spectrograms.

This stream helps detect lip/audio inconsistencies and audio-only manipulations.

### MultiTaskDetector

The three stream embeddings are concatenated and passed through a shared MLP trunk. Three binary heads are trained:

- `video`: visual manipulation,
- `audio`: audio manipulation,
- `any`: any manipulation.

This is preferable to a single binary label because FakeAVCeleb has category-specific manipulations, including real video with fake audio.

## Runtime Paths

- `demo_site/app.py`: Flask web UI and upload/sample handling.
- `demo_site/inference.py`: local predictor and model architecture copy used by the demo.
- `scripts/analyze_video.py`: CLI wrapper for one-video analysis.
- `src/deepfake_detector/reporting.py`: shared report generation for demo and CLI.
- `notebook_v2/deepfake_v3.py`: training/evaluation pipeline.

## Future Refactor

The next structural improvement is to move the model architecture and preprocessing from `demo_site/inference.py` into:

```text
src/deepfake_detector/model.py
src/deepfake_detector/preprocessing.py
src/deepfake_detector/inference.py
```

The demo and CLI would then import the same package code instead of the demo owning a model copy.
