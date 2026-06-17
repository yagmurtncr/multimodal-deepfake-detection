# Roadmap

This roadmap tracks practical improvements that would move the project from a strong research demo toward a production-quality ML application.

## Phase 1 — Repository Quality

- [x] Clean project history and ownership metadata.
- [x] Remove presentation/planning artifacts from the public repo.
- [x] Add safe demo configuration with `MODEL_PATH`.
- [x] Add CLI analysis.
- [x] Add lightweight tests and CI.
- [x] Add model card and architecture documentation.

## Phase 2 — Better Demo UX

- [x] Show structured analysis findings in the web demo.
- [x] Allow downloading a JSON report.
- [ ] Show sampled face frames used by the visual stream.
- [ ] Show lip crops used by the sync stream.
- [ ] Show waveform and mel spectrogram previews.
- [ ] Add an analysis history panel.
- [ ] Add a downloadable PDF report.

## Phase 3 — Package Refactor

- [ ] Move model architecture into `src/deepfake_detector/model.py`.
- [ ] Move preprocessing into `src/deepfake_detector/preprocessing.py`.
- [ ] Move predictor logic into `src/deepfake_detector/inference.py`.
- [ ] Update Flask demo and CLI to import only from `src`.
- [ ] Add type hints to the inference and preprocessing path.

## Phase 4 — Evaluation Depth

- [ ] Add threshold sweep and calibration plots.
- [ ] Add cross-dataset evaluation on a small DFDC or Celeb-DF sample.
- [ ] Add compression/noise/re-encoding robustness tests.
- [ ] Add per-category confusion tables for all three heads.
- [ ] Add model size, latency, and memory benchmarks.

## Phase 5 — Deployment

- [ ] Add a CPU-only Dockerfile for the Flask demo.
- [ ] Add a Hugging Face Space or Gradio demo if weights can be shared.
- [ ] Add a FastAPI service variant for async analysis.
- [ ] Store reports in a lightweight SQLite or PostgreSQL backend.

## Recommended Priority

The highest-impact next step is Phase 2 explainability: face crops, lip crops, and mel spectrogram previews. It makes the project much easier to understand in demos and interviews without needing new training.
