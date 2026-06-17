# Colab Quick Start — v2 Eğitimi

A100 runtime'da yeni notebook aç ve **5 hücre** çalıştır.

## Hücre 1 — Bağımlılıklar (~2 dk)
```python
!pip install -q facenet-pytorch transformers librosa timm scikit-learn \
    matplotlib seaborn tqdm opencv-python-headless albumentations
```

## Hücre 2 — Drive mount + Veri seti hazırlığı (~5-8 dk)
```python
from google.colab import drive
drive.mount('/content/drive')

# Drive'a 6.4 GB zip'in yüklenmiş olduğunu varsay
ZIP = '/content/drive/MyDrive/FakeAVCeleb/FakeAVCeleb_v1.2.zip'
import os
if not os.path.exists('/content/FakeAVCeleb_v1.2'):
    !cp "{ZIP}" /content/
    !cd /content && unzip -q "FakeAVCeleb_v1.2.zip"
!ls /content/FakeAVCeleb_v1.2
```

## Hücre 3 — Script'i indir / kopyala
```python
# Yöntem A: GitHub
# !git clone https://github.com/<user>/deepfake_grup11.git
# !cp deepfake_grup11/notebook_v2/deepfake_v2.py /content/

# Yöntem B: scp / upload
# Colab UI'dan dosya yükle, /content/deepfake_v2.py yoluna
```

## Hücre 4 — Tek komutla çalıştır
```python
%env DATASET_ROOT=/content/FakeAVCeleb_v1.2
%env WORK_DIR=/content/work
%env RESULTS_DIR=/content/drive/MyDrive/Grup11_Deepfake_Results

!mkdir -p /content/work /content/drive/MyDrive/Grup11_Deepfake_Results
!python /content/deepfake_v2.py --stage all --epochs 15 --batch 32 --workers 4
```

## Hücre 5 — Sonuçları Drive'a kopyala
```python
!cp /content/work/*.png /content/work/*.csv /content/work/*.json /content/work/best_model.pt \
   /content/drive/MyDrive/Grup11_Deepfake_Results/
!ls -la /content/drive/MyDrive/Grup11_Deepfake_Results/
```

---

## Aşamalı (debug için)

Önce sadece scan:
```bash
!python /content/deepfake_v2.py --stage scan
```
Sonra küçük epoch ile dene:
```bash
!python /content/deepfake_v2.py --stage train --epochs 2 --batch 16
```
Çalışınca tam koş.

---

## Beklenen sürelendirme (A100)

| Stage | Süre |
|-------|------|
| `scan` | ~30 sn |
| `train` (15 epoch, ~4.6K video) | ~70-90 dk |
| `eval` | ~5 dk |
| `ablation` (7 config) | ~25-35 dk |
| **Total** | **~2-2.5 saat** |

## Sorunlar ve Çözümler

| Hata | Çözüm |
|------|-------|
| `CUDA OOM` | `--batch 16` ya da `--workers 2` |
| `MTCNN no face` çok | `min_face_size=40` (kodda düzelt) |
| Wav2Vec download yavaş | İlk çalıştırma sonrası HF cache var, sonraki run hızlı |
| Dataloader hang | `--workers 0` ile debug |
| Sonuçlar Drive'a kopyalanmıyor | Drive quota dolmuş olabilir, `df -h` kontrol |
