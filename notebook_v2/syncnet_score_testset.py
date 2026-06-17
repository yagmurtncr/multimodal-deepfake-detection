"""
Test set'in tümü için SyncNet conf skorları hesapla.
Çıktı: /content/work/syncnet_test_scores.csv  (path, conf, min_dist)
"""
import os, sys, time, glob, random, subprocess, shutil
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import python_speech_features
from scipy.io import wavfile

sys.path.insert(0, '/content/syncnet_python')
from SyncNetModel import S

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WORK_DIR = os.environ.get('WORK_DIR', '/content/work')
TMP_ROOT = '/content/syncnet_tmp_parallel'
os.makedirs(TMP_ROOT, exist_ok=True)

# Load model
model = S(num_layers_in_fc_layers=1024).to(DEVICE)
sd = torch.load('/content/syncnet_python/data/syncnet_v2.model',
                 map_location=DEVICE, weights_only=False)
new_sd = {k.replace('__S__.', ''): v for k, v in sd.items()}
model.load_state_dict(new_sd)
model.eval()
print('SyncNet loaded.')


def extract_audio_mfcc(video_path, tmp):
    wav = os.path.join(tmp, 'a.wav')
    try:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video_path,
                        '-async', '1', '-ac', '1', '-vn',
                        '-acodec', 'pcm_s16le', '-ar', '16000', wav],
                       check=True, capture_output=True, timeout=30)
    except Exception:
        return None
    if not os.path.exists(wav):
        return None
    sr, audio = wavfile.read(wav)
    mfcc = python_speech_features.mfcc(audio, sr)
    mfcc = np.stack([np.array(c) for c in zip(*mfcc)])
    return mfcc


def extract_face_frames(video_path, target_size=224, max_frames=125):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_fps = 25.0
    step = fps / target_fps
    frames = []
    idx = 0.0
    while int(idx) < n and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if not ret: break
        h, w = fr.shape[:2]
        sz = min(h, w)
        y0, x0 = (h - sz) // 2, (w - sz) // 2
        face = fr[y0:y0+sz, x0:x0+sz]
        face = cv2.resize(face, (target_size, target_size))
        frames.append(face)
        idx += step
    cap.release()
    return frames


def score_video(video_path, worker_id=0):
    tmp = os.path.join(TMP_ROOT, f'w{worker_id}')
    os.makedirs(tmp, exist_ok=True)
    try:
        frames = extract_face_frames(video_path)
        if len(frames) < 6:
            return None, None
        mfcc = extract_audio_mfcc(video_path, tmp)
        if mfcc is None:
            return None, None
        last = min(len(frames), mfcc.shape[1] // 4) - 5
        if last <= 0:
            return None, None

        im = np.stack(frames, axis=3).astype(np.float32)
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0, 3, 4, 1, 2))
        imtv = torch.from_numpy(im).float()
        cct  = torch.from_numpy(np.expand_dims(np.expand_dims(mfcc, 0), 0).astype(np.float32)).float()

        BATCH = 64
        im_feats, cc_feats = [], []
        with torch.no_grad():
            for i in range(0, last, BATCH):
                im_batch = torch.cat([imtv[:, :, v:v+5, :, :] for v in range(i, min(last, i+BATCH))], 0)
                cc_batch = torch.cat([cct[:, :, :, v*4:v*4+20] for v in range(i, min(last, i+BATCH))], 0)
                im_feats.append(model.forward_lip(im_batch.to(DEVICE)).cpu())
                cc_feats.append(model.forward_aud(cc_batch.to(DEVICE)).cpu())
        im_feat = torch.cat(im_feats, 0)
        cc_feat = torch.cat(cc_feats, 0)

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
        minval, _ = torch.min(mdist, 0)
        median = torch.median(mdist)
        return float(minval), float(median - minval)
    except Exception as e:
        return None, None


def main():
    df_t = pd.read_csv(os.path.join(WORK_DIR, 'meta_test.csv'))
    print(f'Scoring {len(df_t)} test videos with SyncNet...')

    rows = []
    t0 = time.time()
    for i, row in tqdm(df_t.iterrows(), total=len(df_t), desc='SyncNet'):
        min_d, conf = score_video(row['path'], worker_id=0)
        rows.append({
            'path': row['path'],
            'category': row['category'],
            'video_fake': row['video_fake'],
            'audio_fake': row['audio_fake'],
            'any_fake':   row['any_fake'],
            'syncnet_min_dist': min_d,
            'syncnet_conf':     conf,
        })
        if i % 200 == 199:
            pd.DataFrame(rows).to_csv(os.path.join(WORK_DIR, 'syncnet_test_scores.csv'), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(WORK_DIR, 'syncnet_test_scores.csv'), index=False)
    elapsed = time.time() - t0
    print(f'Done in {elapsed:.0f}s ({elapsed/len(df_t):.2f}s/video)')


if __name__ == '__main__':
    main()
