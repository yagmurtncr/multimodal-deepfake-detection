"""
Eğitim canlıyken eğrileri çizen monitor.
train.log dosyasından Ep lines'ı parse eder, PNG üretir.

Usage:  python3 live_monitor.py
Output: /content/work/live_curves.png
"""
import re, os, json
import matplotlib.pyplot as plt
import numpy as np

LOG = '/content/work/train.log'
OUT = '/content/work/live_curves.png'

pat = re.compile(
    r"Ep\s+(\d+)/(\d+)\s+\((\d+)s\)\s+tl=([\d.]+)\s+\|\s+"
    r"video auc=([\d.]+)\s+f1=([\d.]+)\s+\|\s+"
    r"audio auc=([\d.]+)\s+f1=([\d.]+)\s+\|\s+"
    r"any auc=([\d.]+)\s+f1=([\d.]+)"
)

rows = []
with open(LOG, 'r', errors='ignore') as f:
    for line in f:
        m = pat.search(line)
        if m:
            rows.append({
                'epoch': int(m.group(1)),
                'total': int(m.group(2)),
                'sec':   int(m.group(3)),
                'train_loss':  float(m.group(4)),
                'video_auc':   float(m.group(5)),
                'video_f1':    float(m.group(6)),
                'audio_auc':   float(m.group(7)),
                'audio_f1':    float(m.group(8)),
                'any_auc':     float(m.group(9)),
                'any_f1':      float(m.group(10)),
            })
if not rows:
    print('Hiç epoch satırı bulunamadı.')
    raise SystemExit(0)

epochs = [r['epoch'] for r in rows]
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# --- Train loss ---
ax[0].plot(epochs, [r['train_loss'] for r in rows], 'b-o', linewidth=2, markersize=8)
ax[0].set_title('Train Loss', fontsize=14, fontweight='bold')
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('BCE Loss (multi-task sum)')
ax[0].grid(alpha=0.3)

# --- Val AUC per task ---
for task, color, marker in [('video', '#3b82f6', 'o'),
                             ('audio', '#f59e0b', 's'),
                             ('any',   '#10b981', '^')]:
    ax[1].plot(epochs, [r[f'{task}_auc'] for r in rows],
               color=color, marker=marker, linewidth=2, label=task)
ax[1].set_title('Val AUC (per task)', fontsize=14, fontweight='bold')
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('AUC')
ax[1].set_ylim(0.5, 1.001); ax[1].axhline(1.0, color='gray', linestyle=':', alpha=0.5)
ax[1].legend(); ax[1].grid(alpha=0.3)

# --- Val F1 per task ---
for task, color, marker in [('video', '#3b82f6', 'o'),
                             ('audio', '#f59e0b', 's'),
                             ('any',   '#10b981', '^')]:
    ax[2].plot(epochs, [r[f'{task}_f1'] for r in rows],
               color=color, marker=marker, linewidth=2, label=task)
ax[2].set_title('Val F1 (per task)', fontsize=14, fontweight='bold')
ax[2].set_xlabel('Epoch'); ax[2].set_ylabel('F1')
ax[2].set_ylim(0.5, 1.001); ax[2].axhline(1.0, color='gray', linestyle=':', alpha=0.5)
ax[2].legend(); ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches='tight')
print(f'saved {OUT}')

# Summary table
print('\nEpoch | Train Loss | Val AUC (video/audio/any) | Val F1 (video/audio/any)')
print('-' * 88)
for r in rows:
    print(f'  {r["epoch"]:2d}  |   {r["train_loss"]:.4f}   |  '
          f'{r["video_auc"]:.4f} / {r["audio_auc"]:.4f} / {r["any_auc"]:.4f}  |  '
          f'{r["video_f1"]:.4f} / {r["audio_f1"]:.4f} / {r["any_f1"]:.4f}')

# Sinyal yakala: aşırı uyum başlangıcı?
if len(rows) >= 3:
    last3_any = [r['any_auc'] for r in rows[-3:]]
    last3_tl  = [r['train_loss'] for r in rows[-3:]]
    if last3_tl[-1] < last3_tl[0] and last3_any[-1] < last3_any[0] - 0.001:
        print('\n⚠️  Train loss düşerken val AUC düştü → potansiyel overfit sinyali')
    elif max(last3_any) - min(last3_any) < 0.001:
        print('\nℹ️  Son 3 epoch val AUC plato (Δ < 0.001) → early stop yakın')
    else:
        print('\n✓ Hala iyi yönde ilerliyor (overfit sinyali yok)')
