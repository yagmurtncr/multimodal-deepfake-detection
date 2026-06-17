# Colab Quick Start — v3 Training

Use an A100/T4 GPU runtime. The FakeAVCeleb videos are not included in this repository; mount or upload your licensed copy before running training.

## 1. Install Dependencies

```python
!pip install -q torch torchvision timm transformers facenet-pytorch librosa \
    opencv-python-headless numpy pandas scikit-learn tqdm matplotlib seaborn
```

## 2. Mount Drive And Prepare Dataset

```python
from google.colab import drive
drive.mount('/content/drive')

# Update this path to your own licensed FakeAVCeleb archive or extracted folder.
DATASET_DIR = '/content/FakeAVCeleb_v1.2'
ZIP = '/content/drive/MyDrive/FakeAVCeleb/FakeAVCeleb_v1.2.zip'

import os
if not os.path.exists(DATASET_DIR):
    !cp "{ZIP}" /content/
    !cd /content && unzip -q "FakeAVCeleb_v1.2.zip"
!ls "{DATASET_DIR}"
```

## 3. Clone This Repository

```python
!git clone https://github.com/yagmurtncr/multimodal-deepfake-detection.git
%cd /content/multimodal-deepfake-detection
```

## 4. Run The v3 Pipeline

```python
%env DATASET_ROOT=/content/FakeAVCeleb_v1.2
%env WORK_DIR=/content/work

!mkdir -p /content/work
!python notebook_v2/deepfake_v3.py --stage scan
!python notebook_v2/deepfake_v3.py --stage train --epochs 12 --batch 24 --workers 4
!python notebook_v2/deepfake_v3.py --stage eval
!python notebook_v2/deepfake_v3.py --stage ablation
```

## 5. Save Artifacts

```python
!mkdir -p /content/drive/MyDrive/MultimodalDeepfakeResults
!cp /content/work/*.png /content/work/*.csv /content/work/*.json /content/work/best_model.pt \
   /content/drive/MyDrive/MultimodalDeepfakeResults/
```

## Debug Tips

| Problem | Fix |
|---|---|
| CUDA OOM | Lower `--batch` to `8` or `16`. |
| DataLoader hangs | Use `--workers 0` for debugging. |
| Many `no_face` failures | Try lowering `min_face_size` in `VideoProcessor`. |
| First run downloads slowly | Wav2Vec2 and Xception weights are cached after the first run. |
