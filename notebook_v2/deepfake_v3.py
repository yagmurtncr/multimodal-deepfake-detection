"""
Grup 11 — Multimodal Deepfake Detection v3 (MULTI-TASK)
========================================================
v2'den farkları:
  - 3 HEAD: video_fake, audio_fake, any_fake
  - 8 frame (16 yerine) → 2-2.5x hızlı
  - Tüm veri seti kullanılır (subset YOK)
  - 4×4 kategori-bazlı confusion matrix
  - Audit'ten gelen "temiz" örnekler kullanılır
"""

import os, sys, time, glob, random, json, argparse
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
SEED = 42
N_FRAMES = 8       # ← v3: 16 → 8
AUDIO_SR = 16000
AUDIO_LEN = 3.0
IMG_SIZE = 299
LIP_SIZE = 112
EMBED_DIM = 512
MEL_T = 301        # ~3s @ hop_length=160

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_ROOT = os.environ.get('DATASET_ROOT', '/content/FakeAVCeleb_v1.2')
WORK_DIR     = os.environ.get('WORK_DIR',     '/content/work')

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ============================================================
# 1) Dataset scan + audit filter + subject-disjoint split
# ============================================================
def scan_dataset(root):
    records = []
    for cat in os.listdir(root):
        cp = os.path.join(root, cat)
        if not os.path.isdir(cp) or cat not in (
            'RealVideo-RealAudio', 'FakeVideo-RealAudio',
            'RealVideo-FakeAudio', 'FakeVideo-FakeAudio'): continue
        for mp4 in glob.glob(os.path.join(cp, '**', '*.mp4'), recursive=True):
            rel = mp4.replace(cp, '').strip(os.sep)
            parts = rel.split(os.sep)
            if cat == 'RealVideo-RealAudio':
                eth, gen, cid = parts[0], parts[1], parts[2]
            else:
                cid = parts[-2]
                eth = parts[0] if len(parts) >= 4 else 'unk'
                gen = parts[1] if len(parts) >= 4 else 'unk'
            vf = cat in ('FakeVideo-RealAudio', 'FakeVideo-FakeAudio')
            af = cat in ('RealVideo-FakeAudio', 'FakeVideo-FakeAudio')
            records.append({
                'path': mp4, 'category': cat,
                'video_fake': int(vf), 'audio_fake': int(af),
                'any_fake':   int(vf or af),
                'celeb_id': cid, 'ethnicity': eth, 'gender': gen,
            })
    return pd.DataFrame(records)


def subject_disjoint_split(df, val_size=0.15, test_size=0.15, seed=SEED):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed)
    tr, tmp = next(gss1.split(df, df['any_fake'], groups=df['celeb_id']))
    df_tr, df_tmp = df.iloc[tr].reset_index(drop=True), df.iloc[tmp].reset_index(drop=True)
    inner = test_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=inner, random_state=seed)
    v, t = next(gss2.split(df_tmp, df_tmp['any_fake'], groups=df_tmp['celeb_id']))
    df_v, df_t = df_tmp.iloc[v].reset_index(drop=True), df_tmp.iloc[t].reset_index(drop=True)
    a, b, c = set(df_tr.celeb_id), set(df_v.celeb_id), set(df_t.celeb_id)
    assert not (a & b) and not (a & c) and not (b & c)
    return df_tr, df_v, df_t


# ============================================================
# 2) Video processing
# ============================================================
class VideoProcessor:
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

            mid = idxs[len(idxs) // 2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, mf = cap.read()
            if not ret: cap.release(); return None
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

            try:
                audio, _ = librosa.load(video_path, sr=AUDIO_SR, mono=True, duration=AUDIO_LEN)
                tlen = int(AUDIO_SR * AUDIO_LEN)
                if len(audio) < tlen:
                    audio = np.pad(audio, (0, tlen - len(audio)))
                else:
                    audio = audio[:tlen]
            except Exception:
                audio = np.zeros(int(AUDIO_SR * AUDIO_LEN), dtype=np.float32)

            mel = librosa.feature.melspectrogram(
                y=audio.astype(np.float32), sr=AUDIO_SR,
                n_mels=64, n_fft=512, hop_length=160)
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)

            return {
                'faces': np.stack(faces).astype(np.uint8),
                'lips':  np.stack(lips).astype(np.uint8),
                'audio': audio.astype(np.float32),
                'mel':   mel.astype(np.float32),
            }
        except Exception:
            return None

    def _crop_face(self, frame, box, margin=40):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
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


IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class FakeAVDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.processor = None

    def _ensure(self):
        if self.processor is None:
            self.processor = VideoProcessor(device='cpu')

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        self._ensure()
        row = self.df.iloc[idx]
        res = self.processor.process(row['path'])
        if res is None:
            return self._zero(row)
        faces = res['faces']; lips = res['lips']; audio = res['audio']; mel = res['mel']
        if self.augment:
            faces, lips, audio = self._augment(faces, lips, audio)
        faces = faces.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        faces = (faces - IMG_MEAN) / IMG_STD
        lips  = lips.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        if mel.shape[1] < MEL_T:
            mel = np.pad(mel, ((0, 0), (0, MEL_T - mel.shape[1])))
        else:
            mel = mel[:, :MEL_T]
        mel = mel[None]
        return {
            'faces': torch.from_numpy(faces).float(),
            'lips':  torch.from_numpy(lips).float(),
            'audio': torch.from_numpy(audio).float(),
            'mel':   torch.from_numpy(mel).float(),
            'y_video': torch.tensor(row['video_fake'], dtype=torch.float32),
            'y_audio': torch.tensor(row['audio_fake'], dtype=torch.float32),
            'y_any':   torch.tensor(row['any_fake'],   dtype=torch.float32),
            'category': row['category'],
            'valid': torch.tensor(1, dtype=torch.float32),
        }

    def _zero(self, row):
        return {
            'faces': torch.zeros(N_FRAMES, 3, IMG_SIZE, IMG_SIZE),
            'lips':  torch.zeros(N_FRAMES, 3, LIP_SIZE, LIP_SIZE),
            'audio': torch.zeros(int(AUDIO_SR * AUDIO_LEN)),
            'mel':   torch.zeros(1, 64, MEL_T),
            'y_video': torch.tensor(row['video_fake'], dtype=torch.float32),
            'y_audio': torch.tensor(row['audio_fake'], dtype=torch.float32),
            'y_any':   torch.tensor(row['any_fake'],   dtype=torch.float32),
            'category': row['category'],
            'valid': torch.tensor(0, dtype=torch.float32),
        }

    def _augment(self, faces, lips, audio):
        if random.random() < 0.5:
            faces = faces[:, :, ::-1, :].copy()
            lips  = lips[:, :, ::-1, :].copy()
        bright = 1.0 + (random.random() - 0.5) * 0.4
        faces = np.clip(faces.astype(np.float32) * bright, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            audio = audio + np.random.normal(0, 0.005, audio.shape).astype(np.float32)
        if random.random() < 0.3:
            shift = random.randint(-int(0.2 * AUDIO_SR), int(0.2 * AUDIO_SR))
            audio = np.roll(audio, shift)
        return faces, lips, audio


def collate_fn(batch):
    out = {}
    for k in ['faces', 'lips', 'audio', 'mel', 'y_video', 'y_audio', 'y_any', 'valid']:
        out[k] = torch.stack([b[k] for b in batch])
    out['category'] = [b['category'] for b in batch]
    return out


# ============================================================
# 3) Model with 3 heads
# ============================================================
class ImageStream(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, pretrained=True):
        super().__init__()
        self.bb = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        bdim = self.bb.num_features
        self.proj = nn.Sequential(
            nn.Linear(2 * bdim, embed_dim), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, x):
        B, N = x.shape[:2]
        f = self.bb(x.view(B * N, 3, IMG_SIZE, IMG_SIZE))
        f = f.view(B, N, -1)
        pooled = torch.cat([f.mean(1), f.std(1)], dim=1)
        return self.proj(pooled)


class AudioStream(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        for p in self.w2v.parameters(): p.requires_grad = False
        self.w2v.eval()
        self.proj = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, embed_dim))

    def forward(self, x):
        with torch.no_grad():
            self.w2v.eval()
            h = self.w2v(x).last_hidden_state.mean(dim=1)
        return self.proj(h.float())


class SyncStream(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.audio = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (1, 2), (1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3), (1, 2), (1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, lips, mel):
        v = self.visual(lips.permute(0, 2, 1, 3, 4)).flatten(1)
        a = self.audio(mel).flatten(1)
        return self.proj(torch.cat([v, a], dim=1))


class MultiTaskDetector(nn.Module):
    """3 head: video_fake, audio_fake, any_fake."""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.image = ImageStream(embed_dim)
        self.audio = AudioStream(embed_dim)
        self.sync  = SyncStream(embed_dim)
        D = 3 * embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(D, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.head_video = nn.Linear(256, 1)
        self.head_audio = nn.Linear(256, 1)
        self.head_any   = nn.Linear(256, 1)

    def forward(self, faces, audio, lips, mel,
                mask_image=False, mask_audio=False, mask_sync=False):
        i = self.image(faces)
        a = self.audio(audio)
        s = self.sync(lips, mel)
        if mask_image: i = torch.zeros_like(i)
        if mask_audio: a = torch.zeros_like(a)
        if mask_sync:  s = torch.zeros_like(s)
        h = self.trunk(torch.cat([i, a, s], dim=1))
        return {
            'video': self.head_video(h).squeeze(1),
            'audio': self.head_audio(h).squeeze(1),
            'any':   self.head_any(h).squeeze(1),
        }


# ============================================================
# 4) Training
# ============================================================
def make_loader(df, augment, batch_size, num_workers, balanced=False):
    ds = FakeAVDataset(df, augment=augment)
    if balanced:
        # 4-kategori dengeli sampler
        cat_arr = df['category'].values
        cat_unique = np.unique(cat_arr)
        cat_cnt = {c: int((cat_arr == c).sum()) for c in cat_unique}
        cat_w = {c: 1.0 / cat_cnt[c] for c in cat_unique}
        sample_w = np.array([cat_w[c] for c in cat_arr], dtype=np.float64)
        sampler = WeightedRandomSampler(weights=sample_w,
                                         num_samples=len(sample_w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
                          persistent_workers=(num_workers > 0))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
                      persistent_workers=(num_workers > 0))


def multitask_loss(preds, batch, w_video=0.5, w_audio=1.0, w_any=1.0, smooth=0.05):
    def bce(p, y):
        ys = y * (1 - smooth) + 0.5 * smooth
        return F.binary_cross_entropy_with_logits(p, ys)
    lv = bce(preds['video'], batch['y_video'])
    la = bce(preds['audio'], batch['y_audio'])
    ly = bce(preds['any'],   batch['y_any'])
    return w_video * lv + w_audio * la + w_any * ly, dict(lv=lv.item(), la=la.item(), ly=ly.item())


def train_epoch(model, loader, optim, scaler, device, log_every=50):
    model.train()
    total, n = 0.0, 0
    for i, batch in enumerate(tqdm(loader, desc='train', leave=False)):
        for k in ['faces', 'lips', 'audio', 'mel', 'y_video', 'y_audio', 'y_any']:
            batch[k] = batch[k].to(device, non_blocking=True)
        optim.zero_grad()
        with autocast():
            preds = model(batch['faces'], batch['audio'], batch['lips'], batch['mel'])
            loss, _ = multitask_loss(preds, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim); scaler.update()
        total += loss.item() * batch['faces'].size(0); n += batch['faces'].size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, mask_image=False, mask_audio=False, mask_sync=False):
    model.eval()
    ys = {'video': [], 'audio': [], 'any': []}
    ps = {'video': [], 'audio': [], 'any': []}
    cats = []
    for batch in tqdm(loader, desc='eval', leave=False):
        for k in ['faces', 'lips', 'audio', 'mel', 'y_video', 'y_audio', 'y_any']:
            batch[k] = batch[k].to(device, non_blocking=True)
        with autocast():
            preds = model(batch['faces'], batch['audio'], batch['lips'], batch['mel'],
                          mask_image=mask_image, mask_audio=mask_audio, mask_sync=mask_sync)
        for task in ['video', 'audio', 'any']:
            ys[task].extend(batch[f'y_{task}'].cpu().numpy())
            ps[task].extend(torch.sigmoid(preds[task]).cpu().numpy())
        cats.extend(batch['category'])
    metrics = {}
    for task in ['video', 'audio', 'any']:
        y = np.array(ys[task]); p = np.array(ps[task]); pred = (p > 0.5).astype(int)
        m = {
            'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall':    float(recall_score(y, pred, zero_division=0)),
            'f1':        float(f1_score(y, pred, zero_division=0)),
            'auc':       float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan'),
        }
        metrics[task] = m
    return metrics, ps, ys, cats


# ============================================================
# 5) Stage runners
# ============================================================
def filter_by_audit(df, audit_csv):
    """Audit'te ok=0 olanları çıkar."""
    if not os.path.exists(audit_csv):
        print(f'(audit not found, skipping filter)')
        return df
    a = pd.read_csv(audit_csv)
    bad = set(a[a['ok'] == 0]['path'])
    out = df[~df['path'].isin(bad)].reset_index(drop=True)
    print(f'Audit filter: {len(df)} → {len(out)} (-{len(df)-len(out)})')
    return out


def stage_scan():
    os.makedirs(WORK_DIR, exist_ok=True)
    df_all = scan_dataset(DATASET_ROOT)
    print(f'Total videos: {len(df_all)}')
    print('Category:', df_all['category'].value_counts().to_dict())
    print('video_fake:', df_all['video_fake'].value_counts().to_dict())
    print('audio_fake:', df_all['audio_fake'].value_counts().to_dict())
    print('any_fake:',   df_all['any_fake'].value_counts().to_dict())

    audit_csv = os.path.join(WORK_DIR, 'audit_report.csv')
    df_all = filter_by_audit(df_all, audit_csv)

    df_tr, df_v, df_t = subject_disjoint_split(df_all, val_size=0.15, test_size=0.15)
    print(f'Split (full): train {len(df_tr)} | val {len(df_v)} | test {len(df_t)}')
    print('  train cats:', df_tr['category'].value_counts().to_dict())
    print('  val cats:',   df_v['category'].value_counts().to_dict())
    print('  test cats:',  df_t['category'].value_counts().to_dict())

    df_tr.to_csv(os.path.join(WORK_DIR, 'meta_train.csv'), index=False)
    df_v.to_csv (os.path.join(WORK_DIR, 'meta_val.csv'),   index=False)
    df_t.to_csv (os.path.join(WORK_DIR, 'meta_test.csv'),  index=False)

    # Dist plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, dfx, name in zip(axs, [df_tr, df_v, df_t], ['Train', 'Val', 'Test']):
        dfx['category'].value_counts().sort_index().plot.bar(ax=ax)
        ax.set_title(f'{name} (n={len(dfx)})')
        ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, 'split_distribution.png'), dpi=140, bbox_inches='tight')
    print('Saved split_distribution.png')


def stage_train(num_epochs=12, batch_size=24, lr=2e-4, wd=1e-4,
                num_workers=6, patience=3):
    os.makedirs(WORK_DIR, exist_ok=True)
    df_tr = pd.read_csv(os.path.join(WORK_DIR, 'meta_train.csv'))
    df_v  = pd.read_csv(os.path.join(WORK_DIR, 'meta_val.csv'))

    train_loader = make_loader(df_tr, augment=True, batch_size=batch_size,
                               num_workers=num_workers, balanced=True)
    val_loader   = make_loader(df_v,  augment=False, batch_size=batch_size,
                               num_workers=num_workers, balanced=False)

    model = MultiTaskDetector(EMBED_DIM).to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_train/1e6:.2f}M')

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)
    scaler = GradScaler()

    history = {'train_loss': [], 'val': []}
    best_auc, pcounter = 0.0, 0
    ckpt = os.path.join(WORK_DIR, 'best_model.pt')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tl = train_epoch(model, train_loader, optim, scaler, DEVICE)
        m, _, _, _ = evaluate(model, val_loader, DEVICE)
        sched.step()
        any_auc = m['any']['auc']
        history['train_loss'].append(tl)
        history['val'].append(m)
        et = time.time() - t0
        print(f"Ep {epoch:02d}/{num_epochs} ({et:.0f}s) "
              f"tl={tl:.4f} | video auc={m['video']['auc']:.4f} f1={m['video']['f1']:.4f} | "
              f"audio auc={m['audio']['auc']:.4f} f1={m['audio']['f1']:.4f} | "
              f"any auc={any_auc:.4f} f1={m['any']['f1']:.4f}")
        if any_auc > best_auc:
            best_auc = any_auc; pcounter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_metrics': m, 'history': history}, ckpt)
            print(f'  ⭐ best any-AUC {best_auc:.4f} saved')
        else:
            pcounter += 1
            if pcounter >= patience:
                print(f'⏹ early stop'); break
    print(f'best val any-AUC = {best_auc:.4f}')


def stage_eval():
    df_t = pd.read_csv(os.path.join(WORK_DIR, 'meta_test.csv'))
    test_loader = make_loader(df_t, augment=False, batch_size=24, num_workers=6)
    model = MultiTaskDetector(EMBED_DIM).to(DEVICE)
    ck = torch.load(os.path.join(WORK_DIR, 'best_model.pt'), map_location=DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    print(f"Loaded epoch {ck['epoch']}")

    m, ps, ys, cats = evaluate(model, test_loader, DEVICE)
    print('=== TEST METRICS ===')
    for task in ['video', 'audio', 'any']:
        print(f'  [{task}]', {k: f'{v:.4f}' for k, v in m[task].items()})

    cats_arr = np.array(cats)
    cat_rows = []
    for c in ['RealVideo-RealAudio', 'FakeVideo-RealAudio',
              'RealVideo-FakeAudio', 'FakeVideo-FakeAudio']:
        msk = cats_arr == c
        if msk.sum() == 0: continue
        row = {'category': c, 'n': int(msk.sum())}
        for task in ['video', 'audio', 'any']:
            y = np.array(ys[task])[msk]; p = np.array(ps[task])[msk]; pred = (p > 0.5).astype(int)
            row[f'{task}_acc'] = float(accuracy_score(y, pred))
            row[f'{task}_recall'] = float(recall_score(y, pred, zero_division=0))
            row[f'{task}_f1'] = float(f1_score(y, pred, zero_division=0))
        cat_rows.append(row)
    pd.DataFrame(cat_rows).to_csv(os.path.join(WORK_DIR, 'per_category.csv'), index=False)
    print('Saved per_category.csv')

    # 4×4 category-aware confusion
    # rows: gerçek kategori, cols: model "any" tahmin
    _plot_4x4_confusion(np.array(ys['any']), np.array(ps['any']), cats_arr)
    _plot_curves(ck['history'])
    _plot_roc_3tasks(ys, ps)

    with open(os.path.join(WORK_DIR, 'test_results.json'), 'w') as f:
        json.dump({'metrics': m, 'per_category': cat_rows, 'epoch': ck['epoch']},
                  f, indent=2)


def stage_ablation():
    df_t = pd.read_csv(os.path.join(WORK_DIR, 'meta_test.csv'))
    test_loader = make_loader(df_t, augment=False, batch_size=24, num_workers=6)
    model = MultiTaskDetector(EMBED_DIM).to(DEVICE)
    ck = torch.load(os.path.join(WORK_DIR, 'best_model.pt'), map_location=DEVICE)
    model.load_state_dict(ck['model_state_dict'])

    configs = [
        ('Full multimodal', dict()),
        ('No image',        dict(mask_image=True)),
        ('No audio',        dict(mask_audio=True)),
        ('No sync',         dict(mask_sync=True)),
        ('Only image',      dict(mask_audio=True, mask_sync=True)),
        ('Only audio',      dict(mask_image=True, mask_sync=True)),
        ('Only sync',       dict(mask_image=True, mask_audio=True)),
    ]
    rows = []
    for name, kw in configs:
        m, _, _, _ = evaluate(model, test_loader, DEVICE, **kw)
        any_m = m['any']
        rows.append({'config': name, **{f'any_{k}': v for k, v in any_m.items()},
                     'video_auc': m['video']['auc'], 'audio_auc': m['audio']['auc']})
        print(f"{name:18s} any_auc={any_m['auc']:.4f} any_f1={any_m['f1']:.4f} "
              f"video_auc={m['video']['auc']:.4f} audio_auc={m['audio']['auc']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(WORK_DIR, 'ablation.csv'), index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    df.set_index('config')[['any_auc', 'video_auc', 'audio_auc']].plot.bar(ax=ax)
    ax.set_ylim(0, 1); ax.set_title('Ablation Study — 3 task heads')
    plt.xticks(rotation=20, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, 'ablation.png'), dpi=140, bbox_inches='tight')


# ============================================================
# Plots
# ============================================================
def _plot_curves(h):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    e = range(1, len(h['train_loss']) + 1)
    ax[0].plot(e, h['train_loss'], 'b-o'); ax[0].set_title('Train loss')
    for task, color in zip(['video', 'audio', 'any'], ['C0', 'C1', 'C2']):
        ax[1].plot(e, [v[task]['auc'] for v in h['val']], '-o', color=color, label=task)
        ax[2].plot(e, [v[task]['f1'] for v in h['val']],  '-s', color=color, label=task)
    ax[1].set_title('Val AUC'); ax[1].legend()
    ax[2].set_title('Val F1');  ax[2].legend()
    plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, 'training_curves.png'), dpi=140)


def _plot_4x4_confusion(y_any, p_any, cats):
    """rows: gerçek kategori, cols: model tahmini (Gerçek vs Sahte) ile binary."""
    pred = (p_any > 0.5).astype(int)
    cat_order = ['RealVideo-RealAudio', 'FakeVideo-RealAudio',
                 'RealVideo-FakeAudio', 'FakeVideo-FakeAudio']
    mat = np.zeros((4, 2), dtype=int)
    for i, c in enumerate(cat_order):
        msk = cats == c
        if msk.sum() == 0: continue
        mat[i, 0] = int((pred[msk] == 0).sum())
        mat[i, 1] = int((pred[msk] == 1).sum())
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: Gerçek', 'Pred: Sahte'],
                yticklabels=[c.replace('Video-', 'V-').replace('Audio', 'A') for c in cat_order],
                ax=ax, cbar=False)
    ax.set_title('Kategori-bazlı tahmin (any_fake head)')
    plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, 'confusion_4x2.png'), dpi=140)

    # Also 4×4: rows = gerçek kategori, cols = tahmin edilen kategori
    # (video_fake_pred, audio_fake_pred kombinasyonu → 4 kategori)
    # bunu sonra eval'de yapacağız


def _plot_roc_3tasks(ys, ps):
    fig, ax = plt.subplots(figsize=(7, 6))
    for task in ['video', 'audio', 'any']:
        y = np.array(ys[task]); p = np.array(ps[task])
        if len(np.unique(y)) < 2: continue
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, label=f'{task}  AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC — 3 tasks')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, 'roc_3tasks.png'), dpi=140)


# ============================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['scan', 'train', 'eval', 'ablation', 'all'], default='all')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    if a.stage in ('scan', 'all'):     stage_scan()
    if a.stage in ('train', 'all'):    stage_train(a.epochs, a.batch, a.lr, num_workers=a.workers)
    if a.stage in ('eval', 'all'):     stage_eval()
    if a.stage in ('ablation', 'all'): stage_ablation()
    print('done.')
