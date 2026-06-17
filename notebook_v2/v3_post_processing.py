"""
Post-processing: TTA + threshold optimization + temperature scaling.

Çıktılar:
- /content/work/post_tta_results.json
- /content/work/post_threshold.json
- /content/work/post_calibration.json
- Bir kümülatif ablation tablosu

Kullanım:
   python3 v3_post_processing.py [--model v3|v4]
"""
import os, sys, json, time, copy, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              recall_score, precision_score, balanced_accuracy_score,
                              brier_score_loss)
import random

sys.path.insert(0, '/content')
from deepfake_v3 import (FakeAVDataset, collate_fn, MultiTaskDetector,
                          EMBED_DIM, DEVICE, N_FRAMES)

WORK_DIR = os.environ.get('WORK_DIR', '/content/work')


def load_model(which='v3'):
    if which == 'v3':
        ck_path = os.path.join(WORK_DIR, 'best_model.pt')
        model = MultiTaskDetector(EMBED_DIM).to(DEVICE)
        sd = torch.load(ck_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(sd['model_state_dict'])
        return model, sd
    else:
        from v4_cross_attention import CrossAttentionDetector
        v3_sd = torch.load(os.path.join(WORK_DIR, 'best_model.pt'),
                            map_location=DEVICE, weights_only=False)['model_state_dict']
        v4_sd = torch.load(os.path.join(WORK_DIR, 'v4_best_model.pt'),
                            map_location=DEVICE, weights_only=False)
        model = CrossAttentionDetector(v3_state_dict=v3_sd).to(DEVICE)
        model.load_state_dict(v4_sd['model_state_dict'])
        return model, v4_sd


def evaluate_pass(model, loader, hflip=False, frame_shift=0, audio_shift_ms=0):
    """Tek pass evaluation, return dict of probs per task + labels + cats."""
    model.eval()
    ys = {'video': [], 'audio': [], 'any': []}
    ls = {'video': [], 'audio': [], 'any': []}  # logits
    cats = []
    for batch in tqdm(loader, desc='eval', leave=False):
        faces = batch['faces'].to(DEVICE, non_blocking=True)
        lips  = batch['lips'].to(DEVICE, non_blocking=True)
        audio = batch['audio'].to(DEVICE, non_blocking=True)
        mel   = batch['mel'].to(DEVICE, non_blocking=True)
        # TTA transformations
        if hflip:
            faces = torch.flip(faces, dims=[-1])
            lips  = torch.flip(lips,  dims=[-1])
        if frame_shift != 0:
            faces = torch.roll(faces, shifts=frame_shift, dims=1)
            lips  = torch.roll(lips,  shifts=frame_shift, dims=1)
        if audio_shift_ms != 0:
            samples = int(audio_shift_ms * 16)
            audio = torch.roll(audio, shifts=samples, dims=-1)
        with torch.no_grad(), autocast():
            preds = model(faces, audio, lips, mel)
        for task in ['video', 'audio', 'any']:
            ys[task].extend(batch[f'y_{task}'].cpu().numpy())
            ls[task].extend(preds[task].cpu().numpy())
        cats.extend(batch['category'])
    return {task: np.array(ls[task]) for task in ls}, \
           {task: np.array(ys[task]) for task in ys}, cats


def metrics_from(logits, y, thresh=0.5):
    p = 1 / (1 + np.exp(-logits))  # sigmoid
    pred = (p > thresh).astype(int)
    return {
        'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan'),
        'f1':  float(f1_score(y, pred, zero_division=0)),
        'acc': float(accuracy_score(y, pred)),
        'bal_acc': float(balanced_accuracy_score(y, pred)),
        'recall':    float(recall_score(y, pred, zero_division=0)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'brier': float(brier_score_loss(y, p)),
    }


def tta_average(model, loader, configs):
    """TTA: configs listesi üzerinde average logits."""
    all_logits = None
    ys_ref = None; cats_ref = None
    for i, cfg in enumerate(configs):
        print(f'  TTA pass {i+1}/{len(configs)}: {cfg}')
        ls, ys, cats = evaluate_pass(model, loader, **cfg)
        if all_logits is None:
            all_logits = {k: np.zeros_like(v) for k, v in ls.items()}
            ys_ref = ys; cats_ref = cats
        for k in ls:
            all_logits[k] += ls[k]
    for k in all_logits:
        all_logits[k] /= len(configs)
    return all_logits, ys_ref, cats_ref


def optimize_threshold(probs, y, metric_fn=f1_score):
    best_t, best_m = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pred = (probs > t).astype(int)
        m = metric_fn(y, pred, zero_division=0)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)


def fit_temperature(val_logits, val_y, lr=0.01, n_iter=200):
    """T parameter for sigmoid: minimize NLL."""
    logits = torch.tensor(val_logits, dtype=torch.float32)
    y = torch.tensor(val_y, dtype=torch.float32)
    T = nn.Parameter(torch.ones(1) * 1.0)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=n_iter)
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / T, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.item())


# ============================================================
# Main
# ============================================================
def main(which='v3'):
    model, _ = load_model(which)
    print(f'Loaded {which} model.')

    from deepfake_v3 import make_loader
    df_v = pd.read_csv(os.path.join(WORK_DIR, 'meta_val.csv'))
    df_t = pd.read_csv(os.path.join(WORK_DIR, 'meta_test.csv'))
    val_loader  = make_loader(df_v, augment=False, batch_size=32, num_workers=6)
    test_loader = make_loader(df_t, augment=False, batch_size=32, num_workers=6)

    # ----- Baseline (1 pass) -----
    print('\n=== BASELINE (single pass) ===')
    base_logits, y_test, cats = evaluate_pass(model, test_loader)
    base_m = {task: metrics_from(base_logits[task], y_test[task]) for task in base_logits}
    for t, m in base_m.items():
        print(f'  [{t}] {m}')

    # ----- TTA -----
    print('\n=== TTA (5 passes) ===')
    tta_configs = [
        {},
        {'hflip': True},
        {'frame_shift': 1},
        {'frame_shift': -1},
        {'audio_shift_ms': 100},
    ]
    tta_logits, _, _ = tta_average(model, test_loader, tta_configs)
    tta_m = {task: metrics_from(tta_logits[task], y_test[task]) for task in tta_logits}
    for t, m in tta_m.items():
        print(f'  [{t}] {m}')

    # ----- Threshold optimization (on val) -----
    print('\n=== THRESHOLD OPT (val) ===')
    val_logits, y_val, _ = evaluate_pass(model, val_loader)
    thresholds, val_f1 = {}, {}
    for task in val_logits:
        p_val = 1 / (1 + np.exp(-val_logits[task]))
        best_t, best_f1 = optimize_threshold(p_val, y_val[task])
        thresholds[task] = best_t; val_f1[task] = best_f1
        print(f'  [{task}] optimal_thresh={best_t:.3f} val_f1={best_f1:.4f}')

    # Apply optimal thresholds to test
    opt_thr_m = {}
    for task in base_logits:
        p_test = 1 / (1 + np.exp(-base_logits[task]))
        opt_thr_m[task] = metrics_from(base_logits[task], y_test[task], thresh=thresholds[task])
    print('\n=== TEST with optimal thresholds ===')
    for t, m in opt_thr_m.items():
        print(f'  [{t}] {m}')

    # ----- Temperature scaling -----
    print('\n=== TEMPERATURE SCALING ===')
    T = {}
    cal_m = {}
    for task in val_logits:
        T[task] = fit_temperature(val_logits[task], y_val[task])
        print(f'  [{task}] T={T[task]:.4f}')
        # apply to test
        scaled = base_logits[task] / T[task]
        cal_m[task] = metrics_from(scaled, y_test[task])

    # ----- Combined: TTA + opt threshold + calibration -----
    print('\n=== COMBINED (TTA + opt-thresh + calibration) ===')
    comb_m = {}
    for task in tta_logits:
        scaled = tta_logits[task] / T[task]
        comb_m[task] = metrics_from(scaled, y_test[task], thresh=thresholds[task])
        print(f'  [{task}] {comb_m[task]}')

    # ----- Save all -----
    out = {
        'model': which,
        'baseline':    base_m,
        'tta':         tta_m,
        'opt_thresh':  opt_thr_m,
        'calibration': cal_m,
        'combined':    comb_m,
        'thresholds':  thresholds,
        'temperatures': T,
    }
    with open(os.path.join(WORK_DIR, f'post_processing_{which}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved post_processing_{which}.json')

    # Print summary table
    print('\n=== SUMMARY ===')
    print(f'{"Variant":20s}{"video AUC":12s}{"audio AUC":12s}{"any AUC":12s}{"any F1":10s}')
    for name, m in [('baseline', base_m), ('TTA', tta_m),
                    ('opt-thresh', opt_thr_m), ('calibration', cal_m),
                    ('combined', comb_m)]:
        print(f'{name:20s}'
              f'{m["video"]["auc"]:<12.4f}'
              f'{m["audio"]["auc"]:<12.4f}'
              f'{m["any"]["auc"]:<12.4f}'
              f'{m["any"]["f1"]:<10.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['v3', 'v4'], default='v3')
    a = p.parse_args()
    main(a.model)
