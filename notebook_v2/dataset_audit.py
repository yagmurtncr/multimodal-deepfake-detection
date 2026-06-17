"""
Dataset audit — videoların gerçekten içerdiği şeyleri kontrol eder.

Kontrol edilenler:
- Süre (saniye)
- Frame count + FPS
- Çözünürlük
- Ses var mı? (audio stream + first-3s RMS)
- MTCNN yüzü buluyor mu? (orta frame'de)

Çıktı:
- /content/work/audit_report.csv (her video için 1 satır)
- /content/work/audit_summary.json (özet istatistik)
- /content/work/audit_problems.csv (eliminate edilmesi gerekenler)
"""
import os, sys, json, time, glob, random
import numpy as np
import pandas as pd
import cv2
import librosa
from tqdm import tqdm
from collections import defaultdict

import torch
from facenet_pytorch import MTCNN

# ============================================================
DATASET_ROOT = os.environ.get('DATASET_ROOT', '/content/FakeAVCeleb_v1.2')
WORK_DIR     = os.environ.get('WORK_DIR',     '/content/work')
SAMPLE_PER_CAT = int(os.environ.get('SAMPLE_PER_CAT', '500'))  # her kategoriden N
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(WORK_DIR, exist_ok=True)


def scan_dataset(root):
    records = []
    for cat in os.listdir(root):
        cat_path = os.path.join(root, cat)
        if not os.path.isdir(cat_path): continue
        if cat not in ('RealVideo-RealAudio', 'FakeVideo-RealAudio',
                       'RealVideo-FakeAudio', 'FakeVideo-FakeAudio'):
            continue
        for mp4 in glob.glob(os.path.join(cat_path, '**', '*.mp4'), recursive=True):
            rel = mp4.replace(cat_path, '').strip(os.sep)
            parts = rel.split(os.sep)
            if cat == 'RealVideo-RealAudio':
                eth, gen, cid = parts[0], parts[1], parts[2]
            else:
                cid = parts[-2]
                eth = parts[0] if len(parts) >= 4 else 'unk'
                gen = parts[1] if len(parts) >= 4 else 'unk'
            label = 0 if cat == 'RealVideo-RealAudio' else 1
            video_fake = cat in ('FakeVideo-RealAudio', 'FakeVideo-FakeAudio')
            audio_fake = cat in ('RealVideo-FakeAudio', 'FakeVideo-FakeAudio')
            records.append({'path': mp4, 'category': cat, 'label': label,
                            'video_fake': int(video_fake), 'audio_fake': int(audio_fake),
                            'celeb_id': cid, 'ethnicity': eth, 'gender': gen})
    return pd.DataFrame(records)


def audit_one(path, mtcnn):
    info = {'path': path, 'ok': 1, 'reason': ''}
    try:
        cap = cv2.VideoCapture(path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info.update({'n_frames': n_frames, 'fps': fps, 'width': w, 'height': h,
                     'duration_s': n_frames / fps if fps else 0.0})

        if n_frames < 8 or fps == 0:
            cap.release()
            info['ok'] = 0; info['reason'] = 'too_short_or_no_fps'
            return info

        # MTCNN on middle frame
        mid = n_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, fr = cap.read()
        cap.release()
        if not ret:
            info['ok'] = 0; info['reason'] = 'cannot_read_mid'; return info
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb)
        info['face_detected'] = int(boxes is not None and len(boxes) > 0)
        info['face_prob'] = float(np.max(probs)) if boxes is not None and len(probs) > 0 and probs[0] is not None else 0.0
        if boxes is not None and len(boxes) > 0:
            box = boxes[int(np.argmax(probs))]
            info['face_w'] = float(box[2] - box[0])
            info['face_h'] = float(box[3] - box[1])
        else:
            info['face_w'] = 0.0; info['face_h'] = 0.0

        # Audio audit
        try:
            audio, sr = librosa.load(path, sr=16000, mono=True, duration=3.0)
            if len(audio) == 0:
                info['has_audio'] = 0; info['audio_rms'] = 0.0; info['audio_duration_loaded'] = 0.0
            else:
                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                info['has_audio'] = 1
                info['audio_rms'] = rms
                info['audio_duration_loaded'] = float(len(audio) / 16000.0)
                info['audio_max_abs'] = float(np.max(np.abs(audio)))
        except Exception as e:
            info['has_audio'] = 0; info['audio_rms'] = 0.0
            info['audio_duration_loaded'] = 0.0; info['audio_max_abs'] = 0.0
            info['reason'] = (info['reason'] + ';audio_err:' + str(e)[:30]).strip(';')

        # silence check: RMS < 0.005 = likely silent
        if info.get('has_audio', 0) and info.get('audio_rms', 0) < 0.005:
            info['silent_audio'] = 1
        else:
            info['silent_audio'] = 0

        # ok flag
        if info['face_detected'] == 0:
            info['ok'] = 0; info['reason'] = (info['reason'] + ';no_face').strip(';')
        if info.get('has_audio', 0) == 0:
            info['ok'] = 0; info['reason'] = (info['reason'] + ';no_audio').strip(';')

        return info
    except Exception as e:
        info['ok'] = 0; info['reason'] = f'exc:{str(e)[:40]}'
        return info


def main():
    print(f'Scanning {DATASET_ROOT}...')
    df_all = scan_dataset(DATASET_ROOT)
    print(f'Total videos: {len(df_all)}')
    print(df_all['category'].value_counts())

    # Stratified sample
    rng = np.random.RandomState(42)
    samples = []
    for cat, sub in df_all.groupby('category'):
        n = min(SAMPLE_PER_CAT, len(sub))
        samples.append(sub.sample(n=n, random_state=42))
    df_audit = pd.concat(samples).reset_index(drop=True)
    print(f'Auditing {len(df_audit)} videos ({SAMPLE_PER_CAT}/category)...')

    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40,
                  thresholds=[0.6, 0.7, 0.7], post_process=False, device=DEVICE)

    results = []
    t0 = time.time()
    for _, row in tqdm(df_audit.iterrows(), total=len(df_audit)):
        r = audit_one(row['path'], mtcnn)
        r['category'] = row['category']
        r['celeb_id'] = row['celeb_id']
        results.append(r)
    elapsed = time.time() - t0
    print(f'Audit took {elapsed:.0f}s ({elapsed/len(df_audit):.2f}s/video)')

    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(WORK_DIR, 'audit_report.csv'), index=False)
    print(f'Saved audit_report.csv: {len(df_r)} rows')

    # Summary by category
    print('\n=== SUMMARY ===')
    summary = {}
    for cat, sub in df_r.groupby('category'):
        s = {
            'n': len(sub),
            'ok': int(sub['ok'].sum()),
            'ok_rate': float(sub['ok'].mean()),
            'face_detected_rate': float(sub['face_detected'].mean()) if 'face_detected' in sub else 0.0,
            'has_audio_rate': float(sub['has_audio'].mean()) if 'has_audio' in sub else 0.0,
            'silent_audio_rate': float(sub['silent_audio'].mean()) if 'silent_audio' in sub else 0.0,
            'mean_duration_s': float(sub['duration_s'].mean()),
            'mean_n_frames': float(sub['n_frames'].mean()),
            'mean_fps': float(sub['fps'].mean()),
            'mean_audio_rms': float(sub['audio_rms'].mean()) if 'audio_rms' in sub else 0.0,
            'mean_face_w': float(sub['face_w'].mean()) if 'face_w' in sub else 0.0,
        }
        summary[cat] = s
        print(f'\n[{cat}]  n={s["n"]}  ok={s["ok"]} ({s["ok_rate"]*100:.1f}%)')
        print(f'  face_detected={s["face_detected_rate"]*100:.1f}%  '
              f'has_audio={s["has_audio_rate"]*100:.1f}%  '
              f'silent_audio={s["silent_audio_rate"]*100:.1f}%')
        print(f'  mean_dur={s["mean_duration_s"]:.2f}s  mean_frames={s["mean_n_frames"]:.0f}  '
              f'mean_fps={s["mean_fps"]:.1f}  mean_rms={s["mean_audio_rms"]:.4f}')

    with open(os.path.join(WORK_DIR, 'audit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    problems = df_r[df_r['ok'] == 0]
    problems.to_csv(os.path.join(WORK_DIR, 'audit_problems.csv'), index=False)
    print(f'\nProblems: {len(problems)} videos eliminate edilecek')
    if len(problems):
        print(problems['reason'].value_counts().head(10))


if __name__ == '__main__':
    main()
