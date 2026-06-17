"""
Grup 11 — Multimodal Deepfake Detection v2
==========================================
Demo notebook'tan iyileştirilmiş, tek dosya halinde script.
Colab'da:  !python deepfake_v2.py --stage <scan|preprocess|train|eval|ablation|all>

İyileştirmeler:
- WeightedRandomSampler (sınıf dengesizliği)
- Multi-frame Xception embedding (mean+std pool)
- Wav2Vec FULLY FROZEN
- Two-stream sync (lip 3D-CNN + mel-spec 2D-CNN)
- On-the-fly preprocessing
- MTCNN tek-detect + 16 frame'de reuse
- Augmentation: albumentations face + audio multi-aug
- AMP + cosine + early stopping
- Detailed ablation + per-category report
"""

import os, sys, time, glob, random, json, gc, argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import cv2
import librosa
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import timm
from transformers import Wav2Vec2Model
from facenet_pytorch import MTCNN

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             balanced_accuracy_score)

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Config
# ============================================================
SEED = 42
N_FRAMES = 16
AUDIO_SR = 16000
AUDIO_LEN = 3.0
IMG_SIZE = 299
LIP_SIZE = 112
EMBED_DIM = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_ROOT = os.environ.get('DATASET_ROOT', '/content/FakeAVCeleb_v1.2')
WORK_DIR     = os.environ.get('WORK_DIR',     '/content/work')
RESULTS_DIR  = os.environ.get('RESULTS_DIR',  '/content/drive/MyDrive/Grup11_Deepfake_Results')

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ============================================================
# 1) Dataset scan + subject-disjoint split + balanced subset
# ============================================================
def scan_dataset(root):
    records = []
    for cat in os.listdir(root):
        cat_path = os.path.join(root, cat)
        if not os.path.isdir(cat_path): continue
        if cat == 'RealVideo-RealAudio':
            label, vf, af = 0, False, False
        elif cat == 'FakeVideo-RealAudio':
            label, vf, af = 1, True, False
        elif cat == 'RealVideo-FakeAudio':
            label, vf, af = 1, False, True
        elif cat == 'FakeVideo-FakeAudio':
            label, vf, af = 1, True, True
        else: continue
        for mp4 in glob.glob(os.path.join(cat_path, '**', '*.mp4'), recursive=True):
            rel = mp4.replace(cat_path, '').strip(os.sep)
            parts = rel.split(os.sep)
            # ethnicity / gender / id / file
            if cat == 'RealVideo-RealAudio':
                eth, gen, cid = parts[0], parts[1], parts[2]
            else:
                # bazı sahte dosyalar method dir içermiyor; id sondan ikinci
                cid = parts[-2]
                eth = parts[0] if len(parts) >= 4 else 'unk'
                gen = parts[1] if len(parts) >= 4 else 'unk'
            records.append({'path': mp4, 'category': cat, 'label': label,
                            'video_fake': vf, 'audio_fake': af,
                            'celeb_id': cid, 'ethnicity': eth, 'gender': gen,
                            'filename': os.path.basename(mp4)})
    return pd.DataFrame(records)


def subject_disjoint_split(df, seed=SEED):
    # %70 train / %15 val / %15 test (celeb based)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, tmp = next(gss1.split(df, df['label'], groups=df['celeb_id']))
    df_tr, df_tmp = df.iloc[tr].reset_index(drop=True), df.iloc[tmp].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    v, t = next(gss2.split(df_tmp, df_tmp['label'], groups=df_tmp['celeb_id']))
    df_v, df_t = df_tmp.iloc[v].reset_index(drop=True), df_tmp.iloc[t].reset_index(drop=True)
    # leakage assertion
    a, b, c = set(df_tr.celeb_id), set(df_v.celeb_id), set(df_t.celeb_id)
    assert not (a & b) and not (a & c) and not (b & c), 'identity leakage!'
    return df_tr, df_v, df_t


def balanced_subset(df, n_real, n_fake_per_cat, seed=SEED):
    rng = np.random.RandomState(seed)
    real = df[df.label == 0]
    real_pick = real.sample(n=min(n_real, len(real)), random_state=seed)
    picks = [real_pick]
    for fc in ['FakeVideo-RealAudio', 'FakeVideo-FakeAudio', 'RealVideo-FakeAudio']:
        f = df[df.category == fc]
        if len(f) == 0: continue
        picks.append(f.sample(n=min(n_fake_per_cat, len(f)), random_state=seed))
    out = pd.concat(picks).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


# ============================================================
# 2) Video processing — single MTCNN detect, reuse box
# ============================================================
class VideoProcessor:
    """Per-worker MTCNN; tek detect, tüm frame'de aynı box."""
    def __init__(self, device='cpu'):
        self.mtcnn = MTCNN(image_size=IMG_SIZE, margin=40, min_face_size=60,
                           thresholds=[0.6, 0.7, 0.7], post_process=False, device=device)

    def process(self, video_path, n_frames=N_FRAMES):
        try:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < n_frames:
                cap.release(); return None
            idxs = np.linspace(0, total - 1, n_frames, dtype=int)

            # Detect on middle frame
            mid = idxs[len(idxs)//2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, mf = cap.read()
            if not ret:
                cap.release(); return None
            mrgb = cv2.cvtColor(mf, cv2.COLOR_BGR2RGB)
            boxes, probs = self.mtcnn.detect(mrgb)
            if boxes is None or len(boxes) == 0:
                cap.release(); return None
            box = boxes[int(np.argmax(probs))]

            faces, lips = [], []
            for ix in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
                ret, fr = cap.read()
                if not ret:
                    if faces:
                        faces.append(faces[-1]); lips.append(lips[-1])
                    continue
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                face = self._crop_face(rgb, box)
                lip  = self._crop_lip(rgb, box)
                if face is None or lip is None:
                    if faces:
                        faces.append(faces[-1]); lips.append(lips[-1])
                    continue
                faces.append(face); lips.append(lip)
            cap.release()
            if len(faces) < n_frames // 2: return None
            while len(faces) < n_frames: faces.append(faces[-1])
            while len(lips)  < n_frames: lips.append(lips[-1])
            faces, lips = faces[:n_frames], lips[:n_frames]

            # Audio
            try:
                audio, _ = librosa.load(video_path, sr=AUDIO_SR, mono=True, duration=AUDIO_LEN)
                tlen = int(AUDIO_SR * AUDIO_LEN)
                if len(audio) < tlen:
                    audio = np.pad(audio, (0, tlen - len(audio)))
                else:
                    audio = audio[:tlen]
            except Exception:
                audio = np.zeros(int(AUDIO_SR * AUDIO_LEN), dtype=np.float32)

            # Mel-spec (Sync stream için)
            mel = librosa.feature.melspectrogram(y=audio.astype(np.float32),
                                                  sr=AUDIO_SR, n_mels=64, n_fft=512, hop_length=160)
            mel = librosa.power_to_db(mel, ref=np.max)  # (64, T_mel)
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)

            return {
                'faces': np.stack(faces).astype(np.uint8),   # (N, 299, 299, 3)
                'lips':  np.stack(lips).astype(np.uint8),    # (N, 112, 112, 3)
                'audio': audio.astype(np.float32),           # (T,)
                'mel':   mel.astype(np.float32)              # (64, T_mel)
            }
        except Exception:
            return None

    def _crop_face(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        # margin
        m = 40
        x1 = max(0, x1-m); y1 = max(0, y1-m); x2 = min(w, x2+m); y2 = min(h, y2+m)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    def _crop_lip(self, frame, box, margin=10):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        lh = y2 - y1
        ly1 = y1 + int(lh * 0.55)
        ly1 = max(0, ly1 - margin); y2 = min(h, y2 + margin)
        lx1 = max(0, x1 - margin);  lx2 = min(w, x2 + margin)
        crop = frame[ly1:y2, lx1:lx2]
        if crop.size == 0: return None
        return cv2.resize(crop, (LIP_SIZE, LIP_SIZE))


# ============================================================
# 3) Dataset
# ============================================================
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class FakeAVDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.processor = None   # her worker'da init

    def _ensure_processor(self):
        if self.processor is None:
            self.processor = VideoProcessor(device='cpu')

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        self._ensure_processor()
        row = self.df.iloc[idx]
        res = self.processor.process(row['path'])
        if res is None:
            # dummy zero sample, label korunur
            return {
                'faces': torch.zeros(N_FRAMES, 3, IMG_SIZE, IMG_SIZE),
                'lips':  torch.zeros(N_FRAMES, 3, LIP_SIZE, LIP_SIZE),
                'audio': torch.zeros(int(AUDIO_SR * AUDIO_LEN)),
                'mel':   torch.zeros(1, 64, 301),
                'label': torch.tensor(row['label'], dtype=torch.float32),
                'category': row['category'],
                'valid':   torch.tensor(0, dtype=torch.float32),
            }
        faces = res['faces']  # (N, 299, 299, 3) uint8
        lips  = res['lips']   # (N, 112, 112, 3) uint8
        audio = res['audio']
        mel   = res['mel']    # (64, T_mel)

        # ---- Augmentation (train only) ----
        if self.augment:
            faces, lips, audio = self._augment(faces, lips, audio)

        # HWC->CHW, normalize
        faces = faces.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        faces = (faces - IMG_MEAN) / IMG_STD
        lips  = lips.astype(np.float32).transpose(0, 3, 1, 2) / 255.0

        # pad/truncate mel'i sabit boyuta — 301 ≈ 3s @ hop_length=160
        target_T = 301
        if mel.shape[1] < target_T:
            mel = np.pad(mel, ((0, 0), (0, target_T - mel.shape[1])))
        else:
            mel = mel[:, :target_T]
        mel = mel[None, :, :]  # (1, 64, 301)

        return {
            'faces': torch.from_numpy(faces).float(),
            'lips':  torch.from_numpy(lips).float(),
            'audio': torch.from_numpy(audio).float(),
            'mel':   torch.from_numpy(mel).float(),
            'label': torch.tensor(row['label'], dtype=torch.float32),
            'category': row['category'],
            'valid': torch.tensor(1, dtype=torch.float32),
        }

    def _augment(self, faces, lips, audio):
        # Faces & Lips: aynı flip ve color jitter tüm clip'e
        flip = random.random() < 0.5
        bright = 1.0 + (random.random() - 0.5) * 0.4
        if flip:
            faces = faces[:, :, ::-1, :].copy()
            lips  = lips[:, :, ::-1, :].copy()
        faces = np.clip(faces.astype(np.float32) * bright, 0, 255).astype(np.uint8)
        # Audio: noise + time shift
        if random.random() < 0.5:
            audio = audio + np.random.normal(0, 0.005, audio.shape).astype(np.float32)
        if random.random() < 0.3:
            shift = random.randint(-int(0.2*AUDIO_SR), int(0.2*AUDIO_SR))
            audio = np.roll(audio, shift)
        return faces, lips, audio


def collate_fn(batch):
    out = {}
    for k in ['faces', 'lips', 'audio', 'mel', 'label', 'valid']:
        out[k] = torch.stack([b[k] for b in batch])
    out['category'] = [b['category'] for b in batch]
    return out


# ============================================================
# 4) Model
# ============================================================
class ImageStream(nn.Module):
    """Multi-frame Xception with mean+std temporal pooling."""
    def __init__(self, embed_dim=EMBED_DIM, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        bdim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(2 * bdim, embed_dim),
            nn.ReLU(), nn.Dropout(0.3),
        )

    def forward(self, x):
        # x: (B, N, 3, 299, 299)
        B, N = x.shape[:2]
        f = self.backbone(x.view(B * N, 3, IMG_SIZE, IMG_SIZE))   # (B*N, bdim)
        f = f.view(B, N, -1)
        pooled = torch.cat([f.mean(1), f.std(1)], dim=1)           # (B, 2*bdim)
        return self.proj(pooled)


class AudioStream(nn.Module):
    """Frozen Wav2Vec 2.0 + projector."""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        for p in self.wav2vec.parameters(): p.requires_grad = False
        self.wav2vec.eval()
        self.proj = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            self.wav2vec.eval()
            out = self.wav2vec(x).last_hidden_state.mean(dim=1)    # (B, 768)
        return self.proj(out.float())


class SyncStream(nn.Module):
    """Two-stream: lip 3D-CNN + mel-spec 2D-CNN."""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.audio = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (1, 2), (1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3), (1, 2), (1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(64 + 64, embed_dim), nn.ReLU(), nn.Dropout(0.3),
        )

    def forward(self, lips, mel):
        # lips: (B, N, 3, 112, 112) -> (B, 3, N, 112, 112)
        v = self.visual(lips.permute(0, 2, 1, 3, 4)).flatten(1)   # (B, 64)
        a = self.audio(mel).flatten(1)                             # (B, 64)
        return self.proj(torch.cat([v, a], dim=1))                 # (B, embed_dim)


class MultimodalDetector(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.image = ImageStream(embed_dim)
        self.audio = AudioStream(embed_dim)
        self.sync  = SyncStream(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * embed_dim, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, faces, audio, lips, mel,
                mask_image=False, mask_audio=False, mask_sync=False):
        i = self.image(faces)
        a = self.audio(audio)
        s = self.sync(lips, mel)
        if mask_image: i = torch.zeros_like(i)
        if mask_audio: a = torch.zeros_like(a)
        if mask_sync:  s = torch.zeros_like(s)
        return self.classifier(torch.cat([i, a, s], dim=1)).squeeze(1)


# ============================================================
# 5) Train / Eval
# ============================================================
def make_loader(df, augment, batch_size, num_workers, balanced=False):
    ds = FakeAVDataset(df, augment=augment)
    if balanced:
        labels = df['label'].values
        cls_cnt = np.bincount(labels, minlength=2)
        w_per_cls = 1.0 / np.maximum(cls_cnt, 1)
        sample_w = w_per_cls[labels]
        sampler = WeightedRandomSampler(weights=sample_w,
                                        num_samples=len(labels), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)


def train_epoch(model, loader, criterion, optim, scaler, device, label_smooth=0.05):
    model.train()
    total_loss, all_p, all_y = 0.0, [], []
    for batch in tqdm(loader, desc='train', leave=False):
        faces = batch['faces'].to(device, non_blocking=True)
        audio = batch['audio'].to(device, non_blocking=True)
        lips  = batch['lips'].to(device, non_blocking=True)
        mel   = batch['mel'].to(device, non_blocking=True)
        y     = batch['label'].to(device, non_blocking=True)
        y_smooth = y * (1 - label_smooth) + 0.5 * label_smooth

        optim.zero_grad()
        with autocast():
            logits = model(faces, audio, lips, mel)
            loss = criterion(logits, y_smooth)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim); scaler.update()

        total_loss += loss.item() * faces.size(0)
        all_p.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, np.array(all_p) > 0.5)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             mask_image=False, mask_audio=False, mask_sync=False):
    model.eval()
    total_loss, all_p, all_y, all_c = 0.0, [], [], []
    for batch in tqdm(loader, desc='eval', leave=False):
        faces = batch['faces'].to(device)
        audio = batch['audio'].to(device)
        lips  = batch['lips'].to(device)
        mel   = batch['mel'].to(device)
        y     = batch['label'].to(device)
        with autocast():
            logits = model(faces, audio, lips, mel,
                           mask_image=mask_image, mask_audio=mask_audio, mask_sync=mask_sync)
            loss = criterion(logits, y)
        total_loss += loss.item() * faces.size(0)
        all_p.extend(torch.sigmoid(logits).cpu().numpy())
        all_y.extend(y.cpu().numpy())
        all_c.extend(batch['category'])
    p = np.array(all_p); yy = np.array(all_y); pred = (p > 0.5).astype(int)
    metrics = {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(yy, pred),
        'balanced_accuracy': balanced_accuracy_score(yy, pred),
        'precision': precision_score(yy, pred, zero_division=0),
        'recall': recall_score(yy, pred, zero_division=0),
        'f1': f1_score(yy, pred, zero_division=0),
        'auc': roc_auc_score(yy, p) if len(np.unique(yy)) > 1 else float('nan'),
    }
    return metrics, p, yy, pred, all_c


# ============================================================
# 6) Stage runners
# ============================================================
def stage_scan():
    os.makedirs(WORK_DIR, exist_ok=True)
    print(f'Scanning {DATASET_ROOT} ...')
    df_all = scan_dataset(DATASET_ROOT)
    print('Toplam video:', len(df_all))
    print(df_all['category'].value_counts())
    print('Label:', df_all['label'].value_counts().to_dict())

    df_tr, df_v, df_t = subject_disjoint_split(df_all)
    # Balanced subset
    df_tr = balanced_subset(df_tr, n_real=1200, n_fake_per_cat=700)
    df_v  = balanced_subset(df_v,  n_real=200,  n_fake_per_cat=140)
    df_t  = balanced_subset(df_t,  n_real=200,  n_fake_per_cat=200)
    print(f'Subset → train {len(df_tr)}, val {len(df_v)}, test {len(df_t)}')

    df_tr.to_csv(os.path.join(WORK_DIR, 'meta_train.csv'), index=False)
    df_v.to_csv (os.path.join(WORK_DIR, 'meta_val.csv'),   index=False)
    df_t.to_csv (os.path.join(WORK_DIR, 'meta_test.csv'),  index=False)

    # Quick distribution plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for axi, dfx, name in zip(ax, [df_tr, df_t], ['Train', 'Test']):
        dfx['category'].value_counts().plot.bar(ax=axi)
        axi.set_title(f'{name} category dist (n={len(dfx)})')
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, 'split_distribution.png'), dpi=160, bbox_inches='tight')
    print('Saved split_distribution.png')


def stage_train(num_epochs=15, batch_size=32, lr=2e-4, wd=1e-4,
                num_workers=4, patience=3):
    os.makedirs(WORK_DIR, exist_ok=True)
    df_tr = pd.read_csv(os.path.join(WORK_DIR, 'meta_train.csv'))
    df_v  = pd.read_csv(os.path.join(WORK_DIR, 'meta_val.csv'))

    train_loader = make_loader(df_tr, augment=True, batch_size=batch_size,
                               num_workers=num_workers, balanced=True)
    val_loader   = make_loader(df_v,  augment=False, batch_size=batch_size,
                               num_workers=num_workers, balanced=False)

    model = MultimodalDetector(EMBED_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params/1e6:.2f}M')

    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)
    scaler = GradScaler()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'val_f1': [], 'val_auc': [], 'lr': []}
    best_auc, pcounter = 0.0, 0
    ckpt = os.path.join(WORK_DIR, 'best_model.pt')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, criterion, optim, scaler, DEVICE)
        vm, _, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        sched.step()
        cur_lr = optim.param_groups[0]['lr']
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vm['loss']); history['val_acc'].append(vm['accuracy'])
        history['val_f1'].append(vm['f1']); history['val_auc'].append(vm['auc'])
        history['lr'].append(cur_lr)
        print(f"Ep {epoch:02d}/{num_epochs} ({time.time()-t0:.0f}s) "
              f"tl {tl:.4f} ta {ta:.4f} | vl {vm['loss']:.4f} va {vm['accuracy']:.4f} "
              f"f1 {vm['f1']:.4f} auc {vm['auc']:.4f} | lr {cur_lr:.2e}")
        if vm['auc'] > best_auc:
            best_auc = vm['auc']; pcounter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_metrics': vm, 'history': history}, ckpt)
            print(f'  ⭐ best AUC {best_auc:.4f} saved')
        else:
            pcounter += 1
            if pcounter >= patience:
                print(f'⏹ early stop @ epoch {epoch}')
                break
    print(f'best val AUC = {best_auc:.4f}')


def stage_eval():
    df_t = pd.read_csv(os.path.join(WORK_DIR, 'meta_test.csv'))
    test_loader = make_loader(df_t, augment=False, batch_size=32, num_workers=4)
    model = MultimodalDetector(EMBED_DIM).to(DEVICE)
    ck = torch.load(os.path.join(WORK_DIR, 'best_model.pt'), map_location=DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    print(f"Loaded best model (epoch {ck['epoch']})")

    crit = nn.BCEWithLogitsLoss()
    m, probs, y, pred, cats = evaluate(model, test_loader, crit, DEVICE)
    print('TEST METRICS:'); [print(f'  {k}: {v:.4f}') for k, v in m.items()]
    print('\n', classification_report(y, pred, target_names=['Gerçek', 'Sahte'], digits=4))

    # Category breakdown
    cat_rows = []
    cats_arr = np.array(cats)
    for c in ['RealVideo-RealAudio', 'FakeVideo-RealAudio',
              'RealVideo-FakeAudio', 'FakeVideo-FakeAudio']:
        msk = cats_arr == c
        if msk.sum() == 0: continue
        cat_rows.append({
            'category': c,
            'n': int(msk.sum()),
            'accuracy': float(accuracy_score(y[msk], pred[msk])),
            'recall':   float(recall_score(y[msk], pred[msk], zero_division=0)),
            'f1':       float(f1_score(y[msk], pred[msk], zero_division=0)),
        })

    # Plots
    _plot_curves(ck['history'])
    _plot_confusion(y, pred, m['accuracy'])
    _plot_roc(y, probs, m['auc'])
    _plot_error_analysis(y, pred, probs)

    # JSON
    out = {'test_metrics': m, 'category': cat_rows, 'epoch': ck['epoch']}
    with open(os.path.join(WORK_DIR, 'test_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('Saved test_results.json')


def stage_ablation():
    df_t = pd.read_csv(os.path.join(WORK_DIR, 'meta_test.csv'))
    test_loader = make_loader(df_t, augment=False, batch_size=32, num_workers=4)
    model = MultimodalDetector(EMBED_DIM).to(DEVICE)
    ck = torch.load(os.path.join(WORK_DIR, 'best_model.pt'), map_location=DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    crit = nn.BCEWithLogitsLoss()

    configs = [
        ('Full multimodal', dict()),
        ('No image (Audio+Sync)', dict(mask_image=True)),
        ('No audio (Image+Sync)', dict(mask_audio=True)),
        ('No sync (Image+Audio)', dict(mask_sync=True)),
        ('Only image', dict(mask_audio=True, mask_sync=True)),
        ('Only audio', dict(mask_image=True, mask_sync=True)),
        ('Only sync',  dict(mask_image=True, mask_audio=True)),
    ]
    rows = []
    for name, kw in configs:
        m, _, _, _, _ = evaluate(model, test_loader, crit, DEVICE, **kw)
        rows.append({'config': name, **{k: float(v) for k, v in m.items()}})
        print(f'{name:30s} acc {m["accuracy"]:.4f} f1 {m["f1"]:.4f} auc {m["auc"]:.4f}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(WORK_DIR, 'ablation.csv'), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    df.set_index('config')[['accuracy', 'f1', 'auc']].plot.bar(ax=ax)
    ax.set_ylim(0, 1); ax.set_title('Ablation Study'); ax.legend(loc='lower right')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, 'ablation.png'), dpi=160, bbox_inches='tight')
    print('Saved ablation.csv + ablation.png')


# ============================================================
# Plots
# ============================================================
def _plot_curves(h):
    e = range(1, len(h['train_loss']) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(e, h['train_loss'], 'b-o', label='train'); ax[0].plot(e, h['val_loss'], 'r-s', label='val')
    ax[0].set_title('Loss'); ax[0].legend()
    ax[1].plot(e, h['train_acc'], 'b-o', label='train'); ax[1].plot(e, h['val_acc'], 'r-s', label='val')
    ax[1].set_title('Accuracy'); ax[1].legend()
    ax[2].plot(e, h['val_auc'], 'g-o', label='AUC'); ax[2].plot(e, h['val_f1'], 'm-s', label='F1')
    ax[2].set_title('Val AUC/F1'); ax[2].legend()
    plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, 'training_curves.png'), dpi=160)


def _plot_confusion(y, pred, acc):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Gerçek', 'Sahte'], yticklabels=['Gerçek', 'Sahte'])
    ax.set_xlabel('Tahmin'); ax.set_ylabel('Gerçek'); ax.set_title(f'Confusion (acc={acc:.4f})')
    plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, 'confusion_matrix.png'), dpi=160)


def _plot_roc(y, probs, auc):
    fpr, tpr, _ = roc_curve(y, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'Multimodal AUC={auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC')
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, 'roc_curve.png'), dpi=160)


def _plot_error_analysis(y, pred, probs):
    wrong = pred != y
    fp = (pred == 1) & (y == 0)
    fn = (pred == 0) & (y == 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(probs[~wrong], bins=30, alpha=0.6, color='g', label='correct')
    ax[0].hist(probs[wrong],  bins=30, alpha=0.6, color='r', label='wrong')
    ax[0].axvline(0.5, color='k', ls='--', alpha=0.4)
    ax[0].set_xlabel('P(fake)'); ax[0].set_title('Score distribution'); ax[0].legend()
    ax[1].bar(['FP (Gerçek→Sahte)', 'FN (Sahte→Gerçek)'], [fp.sum(), fn.sum()], color=['#3b82f6', '#ef4444'])
    ax[1].set_title('Error breakdown')
    for i, v in enumerate([fp.sum(), fn.sum()]):
        ax[1].text(i, v + 0.3, str(int(v)), ha='center', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, 'error_analysis.png'), dpi=160)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['scan', 'train', 'eval', 'ablation', 'all'], default='all')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    if args.stage in ('scan', 'all'):     stage_scan()
    if args.stage in ('train', 'all'):    stage_train(args.epochs, args.batch, args.lr, num_workers=args.workers)
    if args.stage in ('eval', 'all'):     stage_eval()
    if args.stage in ('ablation', 'all'): stage_ablation()
    print('done.')
