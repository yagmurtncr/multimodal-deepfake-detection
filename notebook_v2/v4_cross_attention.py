"""
v4 — Cross-attention fusion + Optional SyncNet feature

v3'ün üzerine inşa edilir:
- v3 checkpoint yüklenir
- Backbone donmuş (Xception + Wav2Vec)
- Yeni cross-attention katmanı + yeni trunk + yeni 3 head eğitilir
- Opsiyonel: SyncNet conf (skaler) eklenmesi

Eğitim: 3-5 epoch warm-start, lr=5e-5, AdamW
"""
import os, sys, time, json, math, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from deepfake_v3 import (FakeAVDataset, collate_fn, make_loader,
                          ImageStream, AudioStream, SyncStream,
                          MultiTaskDetector, EMBED_DIM, DEVICE,
                          N_FRAMES, IMG_SIZE, LIP_SIZE)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
WORK_DIR = os.environ.get('WORK_DIR', '/content/work')


# ============================================================
# Cross-Attention Fusion module
# ============================================================
class CrossModalAttention(nn.Module):
    """Her modaliteyi diğer ikisinin context'iyle yeniden inşa eder."""
    def __init__(self, embed_dim=EMBED_DIM, num_heads=4, dropout=0.2):
        super().__init__()
        self.attn_img2av = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_aud2is = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_snc2ia = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_i = nn.LayerNorm(embed_dim)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_s = nn.LayerNorm(embed_dim)

    def forward(self, i, a, s):
        # Each is (B, D); convert to (B, 1, D)
        I = i.unsqueeze(1); A = a.unsqueeze(1); S = s.unsqueeze(1)
        # image attends to [audio, sync]
        i2, _ = self.attn_img2av(I, torch.cat([A, S], dim=1), torch.cat([A, S], dim=1))
        a2, _ = self.attn_aud2is(A, torch.cat([I, S], dim=1), torch.cat([I, S], dim=1))
        s2, _ = self.attn_snc2ia(S, torch.cat([I, A], dim=1), torch.cat([I, A], dim=1))
        # Residual + LayerNorm
        i_out = self.norm_i(i + i2.squeeze(1))
        a_out = self.norm_a(a + a2.squeeze(1))
        s_out = self.norm_s(s + s2.squeeze(1))
        return i_out, a_out, s_out


class CrossAttentionDetector(nn.Module):
    """v3 model üzerine cross-attention sarar."""
    def __init__(self, v3_state_dict=None, embed_dim=EMBED_DIM, extra_features=0):
        super().__init__()
        self.image = ImageStream(embed_dim, pretrained=False)
        self.audio = AudioStream(embed_dim)
        self.sync  = SyncStream(embed_dim)
        # Load v3 streams
        if v3_state_dict is not None:
            self.load_streams_from_v3(v3_state_dict)
        # Backbone'ları dondur
        for p in self.image.parameters(): p.requires_grad = False
        for p in self.audio.parameters(): p.requires_grad = False
        for p in self.sync.parameters():  p.requires_grad = False
        # Yeni katmanlar
        self.xattn = CrossModalAttention(embed_dim, num_heads=4, dropout=0.2)
        D = 3 * embed_dim + extra_features
        self.trunk = nn.Sequential(
            nn.Linear(D, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head_video = nn.Linear(256, 1)
        self.head_audio = nn.Linear(256, 1)
        self.head_any   = nn.Linear(256, 1)
        self.extra_features = extra_features

    def load_streams_from_v3(self, sd):
        # Filter keys for each stream
        for prefix in ['image.', 'audio.', 'sync.']:
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not sub: continue
            mod = getattr(self, prefix.strip('.'))
            missing, unexpected = mod.load_state_dict(sub, strict=False)
            print(f'  loaded {prefix}: missing={len(missing)} unexpected={len(unexpected)}')

    def forward(self, faces, audio, lips, mel, extra=None,
                mask_image=False, mask_audio=False, mask_sync=False):
        with torch.no_grad():
            i = self.image(faces)
            a = self.audio(audio)
            s = self.sync(lips, mel)
        if mask_image: i = torch.zeros_like(i)
        if mask_audio: a = torch.zeros_like(a)
        if mask_sync:  s = torch.zeros_like(s)
        i, a, s = self.xattn(i, a, s)
        feats = torch.cat([i, a, s], dim=1)
        if self.extra_features > 0 and extra is not None:
            feats = torch.cat([feats, extra], dim=1)
        h = self.trunk(feats)
        return {
            'video': self.head_video(h).squeeze(1),
            'audio': self.head_audio(h).squeeze(1),
            'any':   self.head_any(h).squeeze(1),
        }


def multitask_loss(preds, batch, w_video=0.5, w_audio=1.0, w_any=1.0, smooth=0.05):
    def bce(p, y):
        ys = y * (1 - smooth) + 0.5 * smooth
        return F.binary_cross_entropy_with_logits(p, ys)
    return (w_video * bce(preds['video'], batch['y_video']) +
            w_audio * bce(preds['audio'], batch['y_audio']) +
            w_any   * bce(preds['any'],   batch['y_any']))


def train_v4(num_epochs=5, batch_size=32, lr=5e-5, num_workers=6, patience=2):
    print(f'[v4] Loading v3 checkpoint...')
    v3_ck = torch.load(os.path.join(WORK_DIR, 'best_model.pt'), map_location=DEVICE)
    model = CrossAttentionDetector(v3_state_dict=v3_ck['model_state_dict'])
    model = model.to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[v4] Trainable (xattn+trunk+heads): {n_train/1e6:.2f}M')

    df_tr = pd.read_csv(os.path.join(WORK_DIR, 'meta_train.csv'))
    df_v  = pd.read_csv(os.path.join(WORK_DIR, 'meta_val.csv'))

    train_loader = make_loader(df_tr, augment=True, batch_size=batch_size,
                               num_workers=num_workers, balanced=True)
    val_loader   = make_loader(df_v,  augment=False, batch_size=batch_size,
                               num_workers=num_workers, balanced=False)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)
    scaler = GradScaler()

    best_auc, pcounter = 0.0, 0
    history = {'train_loss': [], 'val': []}
    ckpt = os.path.join(WORK_DIR, 'v4_best_model.pt')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        # Backbone'ları eval'da tut (BN istatistikleri)
        model.image.eval(); model.audio.eval(); model.sync.eval()
        total, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f'v4 train ep{epoch}', leave=False):
            for k in ['faces', 'lips', 'audio', 'mel', 'y_video', 'y_audio', 'y_any']:
                batch[k] = batch[k].to(DEVICE, non_blocking=True)
            optim.zero_grad()
            with autocast():
                preds = model(batch['faces'], batch['audio'], batch['lips'], batch['mel'])
                loss = multitask_loss(preds, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optim); scaler.update()
            total += loss.item() * batch['faces'].size(0); n += batch['faces'].size(0)
        tl = total / max(n, 1)

        # Val
        model.eval()
        ys = {'video': [], 'audio': [], 'any': []}; ps = {'video': [], 'audio': [], 'any': []}
        for batch in tqdm(val_loader, desc=f'v4 val ep{epoch}', leave=False):
            for k in ['faces', 'lips', 'audio', 'mel', 'y_video', 'y_audio', 'y_any']:
                batch[k] = batch[k].to(DEVICE, non_blocking=True)
            with torch.no_grad(), autocast():
                preds = model(batch['faces'], batch['audio'], batch['lips'], batch['mel'])
            for task in ['video', 'audio', 'any']:
                ys[task].extend(batch[f'y_{task}'].cpu().numpy())
                ps[task].extend(torch.sigmoid(preds[task]).cpu().numpy())
        vm = {}
        for task in ['video', 'audio', 'any']:
            y = np.array(ys[task]); p = np.array(ps[task]); pred = (p > 0.5).astype(int)
            vm[task] = {
                'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan'),
                'f1':  float(f1_score(y, pred, zero_division=0)),
                'acc': float(accuracy_score(y, pred)),
                'recall': float(recall_score(y, pred, zero_division=0)),
            }
        sched.step()
        history['train_loss'].append(tl); history['val'].append(vm)
        any_auc = vm['any']['auc']
        print(f"v4 Ep {epoch:02d}/{num_epochs} ({time.time()-t0:.0f}s) tl={tl:.4f} | "
              f"video auc={vm['video']['auc']:.4f} | audio auc={vm['audio']['auc']:.4f} | "
              f"any auc={any_auc:.4f} f1={vm['any']['f1']:.4f}")
        if any_auc > best_auc:
            best_auc = any_auc; pcounter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_metrics': vm, 'history': history,
                        'v3_source': 'best_model.pt'}, ckpt)
            print(f'  ⭐ v4 best any-AUC {best_auc:.4f} saved')
        else:
            pcounter += 1
            if pcounter >= patience:
                print('⏹ v4 early stop'); break
    print(f'[v4] best val any-AUC = {best_auc:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    train_v4(a.epochs, a.batch, a.lr, a.workers)
    print('v4 done.')
