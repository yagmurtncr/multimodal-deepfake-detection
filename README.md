# Multimodal Deepfake Detection — Grup 11

> Marmara Üniversitesi · Yapay Zeka dersi grup projesi · FakeAVCeleb v1.2 üzerinde **video + ses + dudak-senkron** birleşik (multimodal) deepfake tespiti.

---

## 🇬🇧 English summary (TL;DR)

A **multimodal deepfake detector** trained on the full **FakeAVCeleb v1.2** dataset (21,544 videos).
It fuses three modalities — **Xception** (visual frames), **Wav2Vec 2.0** (audio), and a lightweight
**3D+2D CNN sync stream** (lip–audio synchrony) — via **late fusion** into a shared MLP trunk with
**three task heads** (video-fake, audio-fake, any-fake).

Key idea: instead of a single naive binary label (which is heavily imbalanced, ~1:42), we use
**multi-task supervision** so each modality is scored independently. This exposes the real value of
multimodality: the *Real-Video / Fake-Audio* category (only the audio is synthetic) — invisible to an
image-only model — is still caught at **93.3% recall**.

**Test results (v3 baseline):** any-AUC **0.9994**, video-AUC 0.9991, audio-AUC 0.9982.
The repo also includes an ablation study, cross-attention fusion experiments (v4), TTA +
threshold/temperature post-processing, and a pretrained-SyncNet comparison.

> ⚠️ This repository contains **code, documentation and result artifacts only**. Trained model weights
> and the FakeAVCeleb video clips are **not included** (size + research-only license + privacy). See
> *Kurulum & Çalıştırma* and *Veri seti & Etik* below.

---

## Proje Özeti

- **Hedef:** FakeAVCeleb veri seti üzerinde Xception + Wav2Vec 2.0 + Two-Stream Sync mimarisi ile multimodal deepfake tespiti.
- **Yaklaşım:** Late Fusion + paylaşımlı MLP trunk + 3 görev başlığı (video / audio / any).
- **Süre:** Yoğun eğitim ve değerlendirme çalışması.
- **Donanım:** Google Colab Pro A100 80GB GPU.

## Mimari

```
            ┌─────────────────────┐
 video ───► │ Xception (fine-tune)│ ─┐
 kareleri   └─────────────────────┘  │
            ┌─────────────────────┐  │   ┌──────────┐   ┌─ head_video (sigmoid)
 ses ─────► │ Wav2Vec 2.0 (frozen)│ ─┼─► │ MLP trunk│ ─►├─ head_audio (sigmoid)
            └─────────────────────┘  │   └──────────┘   └─ head_any   (sigmoid)
            ┌─────────────────────┐  │      late
 dudak+mel► │ 3D+2D CNN Sync      │ ─┘     fusion
            └─────────────────────┘
```

- **ImageStream:** 8 kareden Xception özniteliği → mean+std pooling → 512-d.
- **AudioStream:** 3 sn ses → Wav2Vec 2.0 (donuk) → 512-d.
- **SyncStream:** dudak kırpıntıları (3D CNN) + mel-spektrogram (2D CNN) → dudak-ses uyumu.
- **Trunk + 3 head:** birleştirilmiş 3×512 → 512 → 256 → {video, audio, any}.

## 📊 Test Sonuçları (v3 baseline)

```
Task     │  AUC    │  F1     │  Precision │  Recall
─────────┼─────────┼─────────┼────────────┼─────────
video    │ 0.9991  │ 0.9994  │   1.0000   │ 0.9987
audio    │ 0.9982  │ 0.9875  │   0.9889   │ 0.9860
any      │ 0.9994  │ 0.9984  │   0.9997   │ 0.9972
```

### Kategori-bazlı recall (multimodalin somut değeri)
| Kategori | n | any_recall |
|---|---:|---:|
| Real-Real | 75 | (kontrol) |
| Fake-Real | 1485 | 0.9973 |
| **Real-Fake** | **75** | **0.9333** ⭐ |
| Fake-Fake | 1641 | 1.0000 |

→ R-F (yalnızca ses sahte): 70/75 yakalandı. Image-only model bu kategoride ~%0 verir.

### Ablation (any AUC)
```
Full multimodal   0.9994  ← baseline
No image          0.7618  ← image kritik
No audio          0.9993
Only image        0.9921  ← yanıltıcı yüksek (R-F kategorisi atlanıyor!)
Only audio        0.6393  ← çok zayıf
Only sync         0.7596
```

## Kurulum & Çalıştırma

### 1. Bağımlılıklar
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```
> Python 3.10+ önerilir. GPU (CUDA) eğitim için gereklidir; demo CPU'da da çalışır (yavaş).

### 2. Model ağırlığı
Eğitilmiş `best_model.pt` **repoya dahil değildir.** İki seçenek:
- **Eğit:** `notebook_v2/deepfake_v3.py` ile (FakeAVCeleb gerekir — aşağıya bakın).
- **Hazır checkpoint kullan:** kendi `best_model.pt` yolunuzu çevre değişkeniyle verin.

### 3. Demo web arayüzü
```bash
cd demo_site
# Model yolunu ortam değişkeniyle verin:
#   Windows PowerShell:  $env:MODEL_PATH = "C:\path\to\best_model.pt"
#   Linux/Mac:           export MODEL_PATH=/path/to/best_model.pt
python app.py
# → http://127.0.0.1:5000
```
Demo, `MODEL_PATH` verilmemişse yine açılır ve arayüzde model uyarısı gösterir; analiz endpoint'i
model yüklenene kadar çalışmaz. Kendi videonuzu yükleyebilir veya hazır örnekler için
`demo_site/samples/` klasörüne `.mp4`, `.mov`, `.avi` ya da `.mkv` ekleyebilirsiniz
(bkz. [demo_site/samples/README.md](demo_site/samples/README.md)).

### 4. Eğitim
Ana eğitim scripti `notebook_v2/deepfake_v3.py` içindedir. FakeAVCeleb verisine erişim ve GPU ortamı
gereklidir.

```bash
set DATASET_ROOT=C:\path\to\FakeAVCeleb_v1.2
set WORK_DIR=C:\path\to\work
python notebook_v2/deepfake_v3.py --stage scan
python notebook_v2/deepfake_v3.py --stage train --epochs 12 --batch 24 --workers 4
python notebook_v2/deepfake_v3.py --stage eval
python notebook_v2/deepfake_v3.py --stage ablation
```

Colab için kısa rehber: [notebook_v2/colab_quickstart.md](notebook_v2/colab_quickstart.md).

### 5. Tek video CLI analizi

```bash
python scripts/analyze_video.py --model C:\path\to\best_model.pt --video C:\path\to\sample.mp4
python scripts/analyze_video.py --model C:\path\to\best_model.pt --video C:\path\to\sample.mp4 --json --output report.json
```

CLI çıktısı model skorlarını, karar özetini ve demo arayüzündekiyle aynı açıklayıcı raporu üretir.

### 6. Hafif testler

```bash
pytest tests
```

Bu testler ağır model ağırlığı indirmez; raporlama ve yardımcı mantıkların bozulmadığını kontrol eder.

## Veri seti & Etik

- **Veri seti:** [FakeAVCeleb v1.2](https://github.com/DASH-Lab/FakeAVCeleb) — 21.544 video. Subject-disjoint split (350/75/75 ünlü; train/val/test kesişimi sıfır), böylece model yüz tanımayı değil sahteleme imzalarını öğrenir.
- **Lisans/dağıtım:** FakeAVCeleb yalnızca araştırma amaçlı lisanslıdır ve **yeniden dağıtımı yasaktır.** Bu nedenle veri klipleri ve eğitilmiş ağırlıklar bu repoda yer almaz. Veri setine erişim için resmi EULA'yı imzalamanız gerekir.
- **Etik:** Bu çalışma **savunma amaçlıdır** — deepfake'leri *üretmek* için değil, *tespit etmek* için. Yüksek skorlar (~%99.94) "çözüldü" anlamına gelmez: Wav2Lip ve SV2TTS imzaları belirgindir ve cross-dataset (farklı veri seti) performansı bilinmemektedir.

## Repo yapısı

| Yol | İçerik |
|-----|--------|
| `src/deepfake_detector/` | Demo ve CLI tarafından paylaşılan yardımcı raporlama kodu |
| `scripts/` | Komut satırı araçları, ör. `analyze_video.py` |
| `tests/` | Hafif unit testler |
| `notebook_v2/` | Eğitim ve analiz scriptleri: `deepfake_v3.py`, `v4_cross_attention.py`, `v3_post_processing.py`, `dataset_audit.py`, `syncnet_*.py`, `smoke_test.py`, `live_monitor.py` |
| `demo_site/` | Flask demo: `app.py`, `inference.py`, `static/`, `templates/` |
| `results/` | Eğitim çıktıları: ROC/confusion/training PNG'leri, `test_results.json`, `ablation.csv`, `per_category.csv` |
| `requirements.txt` | Python bağımlılıkları |
| `.env.example` | Demo için örnek `MODEL_PATH` ayarı |
| `LICENSE` | MIT lisansı |

## Ekip

- **Nur Yağmur Tuncer** — [@yagmurtncr](https://github.com/yagmurtncr)
- Grup 11 ekip projesi · Yapay Zeka dersi · Marmara Üniversitesi
- Tarih: 2026-06

> Not: Repoda yer almayanlar — eğitilmiş model ağırlıkları (`*.pt`), FakeAVCeleb video klipleri (`demo_site/samples/*`), demo runtime yüklemeleri (`demo_site/uploads/`). Ayrıntılar için [.gitignore](.gitignore).
