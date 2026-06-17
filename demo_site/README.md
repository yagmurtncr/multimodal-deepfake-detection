# Flask Demo

This folder contains the local web demo for single-video multimodal deepfake analysis.

## Run

```bash
cd demo_site

# Windows PowerShell
$env:MODEL_PATH = "C:\path\to\best_model.pt"

# Linux/macOS
export MODEL_PATH=/path/to/best_model.pt

python app.py
```

Open `http://127.0.0.1:5000`.

If `MODEL_PATH` is missing or invalid, the UI still opens and shows a model warning. Analysis requests return a clear error until a valid checkpoint is configured.

## Inputs

Supported upload extensions:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`

Sample videos can be placed under `demo_site/samples/`. Files are ignored by Git to avoid redistributing licensed or private video data.

## Output

For each successful analysis, the demo shows:

- video/audio/any scores,
- a final decision label,
- timing and preprocessing metadata,
- a structured explanation report,
- downloadable JSON report.
