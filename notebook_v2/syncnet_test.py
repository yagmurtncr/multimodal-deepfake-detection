"""
SyncNet pretrained weights üzerinde 4 kategori discriminative testi.

Mantık:
  Gerçek video → lip ile audio embeddingleri "yakın" olmalı (low distance)
  Sahte video  → mismatch → "uzak"

SyncNet input formatı: 25 FPS video, 5 frame penceresi, 224x224 yüz crop
"""

import os, sys, time, glob, random
import cv2
import numpy as np
import pandas as pd
import torch
import subprocess

sys.path.insert(0, '/content/syncnet_python')
from SyncNetModel import S

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_ROOT = '/content/FakeAVCeleb_v1.2'
TMP = '/content/syncnet_tmp'
os.makedirs(TMP, exist_ok=True)

# Load pretrained
model = S(num_layers_in_fc_layers=1024).to(DEVICE)
sd = torch.load('/content/syncnet_python/data/syncnet_v2.model',
                map_location=DEVICE, weights_only=False)
# weight dict has prefix __S__ — handle both cases
new_sd = {}
for k, v in sd.items():
    new_sd[k.replace('__S__.', '')] = v
model.load_state_dict(new_sd)
model.eval()
print('SyncNet loaded.')

import python_speech_features


def extract_audio_mfcc(video_path, tmp):
    """Video → 16kHz WAV → MFCC tensor"""
    wav = os.path.join(tmp, 'a.wav')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video_path,
                    '-async', '1', '-ac', '1', '-vn',
                    '-acodec', 'pcm_s16le', '-ar', '16000', wav],
                   check=True, capture_output=True)
    from scipy.io import wavfile
    sr, audio = wavfile.read(wav)
    mfcc = python_speech_features.mfcc(audio, sr)        # (T, 13)
    mfcc = np.stack([np.array(c) for c in zip(*mfcc)])    # (13, T)
    return mfcc, len(audio) / sr


def extract_face_frames(video_path, target_size=224, max_frames=200):
    """Video → frame'lerin orta yüz bölgesini 224x224 kırp."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Frame'leri 25 FPS hıza yeniden örnekle
    target_fps = 25.0
    step = fps / target_fps
    frames = []
    idx = 0.0
    while int(idx) < n and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if not ret: break
        h, w = fr.shape[:2]
        # Basit: orta crop (face detection yerine — hızlı test için)
        sz = min(h, w)
        y0, x0 = (h - sz) // 2, (w - sz) // 2
        face = fr[y0:y0+sz, x0:x0+sz]
        face = cv2.resize(face, (target_size, target_size))
        frames.append(face)
        idx += step
    cap.release()
    return frames, target_fps


def sync_score(video_path, tmp):
    """Return (min_dist, conf, n_windows). Lower min_dist = more synced."""
    frames, fps = extract_face_frames(video_path)
    if len(frames) < 6:
        return None, None, 0
    mfcc, audio_dur = extract_audio_mfcc(video_path, tmp)
    n_audio_frames = mfcc.shape[1]
    # SyncNet: 4 audio frames per 1 video frame (160ms hop on 25fps)
    last = min(len(frames), n_audio_frames // 4) - 5
    if last <= 0:
        return None, None, 0

    # Build batches
    im = np.stack(frames, axis=3).astype(np.float32)         # (224, 224, 3, T)
    im = np.expand_dims(im, 0)                                # (1, ...)
    im = np.transpose(im, (0, 3, 4, 1, 2))                    # (1, 3, T, 224, 224)
    imtv = torch.from_numpy(im).float()
    cct  = torch.from_numpy(np.expand_dims(np.expand_dims(mfcc, 0), 0).astype(np.float32)).float()

    BATCH = 32
    im_feats, cc_feats = [], []
    with torch.no_grad():
        for i in range(0, last, BATCH):
            im_batch = torch.cat([imtv[:, :, v:v+5, :, :] for v in range(i, min(last, i+BATCH))], 0)
            cc_batch = torch.cat([cct[:, :, :, v*4:v*4+20] for v in range(i, min(last, i+BATCH))], 0)
            im_feats.append(model.forward_lip(im_batch.to(DEVICE)).cpu())
            cc_feats.append(model.forward_aud(cc_batch.to(DEVICE)).cpu())
    im_feat = torch.cat(im_feats, 0)
    cc_feat = torch.cat(cc_feats, 0)

    # Compute distance over offsets (-10..10)
    vshift = 10
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(cc_feat, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(im_feat)):
        d = torch.nn.functional.pairwise_distance(
            im_feat[[i], :].repeat(win_size, 1),
            feat2p[i:i+win_size, :])
        dists.append(d)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    median = torch.median(mdist)
    return float(minval), float(median - minval), int(last)


# Sample 5 video from each category
def sample_videos(per_cat=5):
    cats = ['RealVideo-RealAudio', 'FakeVideo-RealAudio',
            'RealVideo-FakeAudio', 'FakeVideo-FakeAudio']
    out = []
    for c in cats:
        all_v = glob.glob(f'{DATASET_ROOT}/{c}/**/*.mp4', recursive=True)
        random.seed(42 + hash(c) % 1000)
        for v in random.sample(all_v, min(per_cat, len(all_v))):
            out.append((c, v))
    return out


def main():
    samples = sample_videos(per_cat=5)
    rows = []
    for cat, path in samples:
        t0 = time.time()
        min_d, conf, nw = sync_score(path, TMP)
        dt = time.time() - t0
        if min_d is None:
            print(f'[{cat}] {os.path.basename(path):40s} FAILED')
            continue
        rows.append({'category': cat, 'min_dist': min_d, 'conf': conf,
                     'n_windows': nw, 'time_s': dt, 'path': os.path.basename(path)})
        print(f'[{cat}] {os.path.basename(path):40s} '
              f'min_dist={min_d:6.3f}  conf={conf:6.3f}  n={nw:3d}  t={dt:.2f}s')

    df = pd.DataFrame(rows)
    print('\n=== AGGREGATE ===')
    agg = df.groupby('category')[['min_dist', 'conf']].agg(['mean', 'std']).round(3)
    print(agg)
    df.to_csv('/content/work/syncnet_test_results.csv', index=False)
    print('\nSaved syncnet_test_results.csv')


if __name__ == '__main__':
    main()
