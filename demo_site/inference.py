"""
Local inference wrapper — loads best_model.pt, takes mp4 path → returns scores.

Mimari: deepfake_v3.py'den birebir kopya (eğitilen modelle uyumlu olmak için).
"""
import os, sys, time
import numpy as np
import cv2
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import timm
from transformers import Wav2Vec2Model
from facenet_pytorch import MTCNN

# Config — eğitimle aynı
N_FRAMES   = 8
AUDIO_SR   = 16000
AUDIO_LEN  = 3.0
IMG_SIZE   = 299
LIP_SIZE   = 112
EMBED_DIM  = 512
MEL_T      = 301

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Mimari (deepfake_v3.py birebir)
# ============================================================
class ImageStream(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, pretrained=False):
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
        for p in self.w2v.parameters():
            p.requires_grad = False
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
        self.proj = nn.Sequential(nn.Linear(128, embed_dim), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, lips, mel):
        v = self.visual(lips.permute(0, 2, 1, 3, 4)).flatten(1)
        a = self.audio(mel).flatten(1)
        return self.proj(torch.cat([v, a], dim=1))


class MultiTaskDetector(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.image = ImageStream(embed_dim, pretrained=False)
        self.audio = AudioStream(embed_dim)
        self.sync  = SyncStream(embed_dim)
        D = 3 * embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(D, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.head_video = nn.Linear(256, 1)
        self.head_audio = nn.Linear(256, 1)
        self.head_any   = nn.Linear(256, 1)

    def forward(self, faces, audio, lips, mel):
        i = self.image(faces); a = self.audio(audio); s = self.sync(lips, mel)
        h = self.trunk(torch.cat([i, a, s], dim=1))
        return {
            'video': self.head_video(h).squeeze(1),
            'audio': self.head_audio(h).squeeze(1),
            'any':   self.head_any(h).squeeze(1),
        }


# ============================================================
# Video → tensor pipeline (eğitimle aynı önişleme)
# ============================================================
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class VideoPreprocessor:
    def __init__(self, device=None):
        self.device = device or DEVICE
        self.mtcnn = MTCNN(image_size=IMG_SIZE, margin=40, min_face_size=60,
                            thresholds=[0.6, 0.7, 0.7], post_process=False,
                            device=self.device)

    def crop_face(self, frame, box, margin=40):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    def crop_lip(self, frame, box, margin=10):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        lh = y2 - y1
        ly1 = max(0, y1 + int(lh * 0.55) - margin)
        y2  = min(h, y2 + margin)
        lx1 = max(0, x1 - margin); lx2 = min(w, x2 + margin)
        crop = frame[ly1:y2, lx1:lx2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (LIP_SIZE, LIP_SIZE))

    def process(self, video_path, n_frames=N_FRAMES):
        info = {'video_path': video_path, 'fail_reason': None}
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info['total_frames'] = total
        info['fps'] = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total < n_frames:
            cap.release()
            info['fail_reason'] = 'too_few_frames'
            return None, info
        idxs = np.linspace(0, total - 1, n_frames, dtype=int)

        # Detect on middle frame
        mid = idxs[len(idxs) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, mf = cap.read()
        if not ret:
            cap.release()
            info['fail_reason'] = 'cannot_read_mid'
            return None, info
        rgb = cv2.cvtColor(mf, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            cap.release()
            info['fail_reason'] = 'no_face'
            return None, info
        box = boxes[int(np.argmax(probs))]
        info['face_prob'] = float(np.max(probs))
        info['face_box'] = box.tolist()

        faces, lips = [], []
        for ix in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
            ret, fr = cap.read()
            if not ret:
                if faces:
                    faces.append(faces[-1]); lips.append(lips[-1])
                continue
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            face = self.crop_face(rgb, box)
            lip  = self.crop_lip(rgb, box)
            if face is None or lip is None:
                if faces:
                    faces.append(faces[-1]); lips.append(lips[-1])
                continue
            faces.append(face); lips.append(lip)
        cap.release()
        if len(faces) < n_frames // 2:
            info['fail_reason'] = 'too_few_face_crops'
            return None, info
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
            info['audio_rms'] = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        except Exception as e:
            audio = np.zeros(int(AUDIO_SR * AUDIO_LEN), dtype=np.float32)
            info['audio_rms'] = 0.0
            info['fail_reason'] = f'audio_err:{str(e)[:30]}'

        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float32), sr=AUDIO_SR,
            n_mels=64, n_fft=512, hop_length=160)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        if mel.shape[1] < MEL_T:
            mel = np.pad(mel, ((0, 0), (0, MEL_T - mel.shape[1])))
        else:
            mel = mel[:, :MEL_T]

        return {
            'faces': np.stack(faces).astype(np.uint8),
            'lips':  np.stack(lips).astype(np.uint8),
            'audio': audio.astype(np.float32),
            'mel':   mel.astype(np.float32),
        }, info


# ============================================================
# Main predictor (singleton)
# ============================================================
class DeepfakePredictor:
    def __init__(self, model_path, device=None):
        self.device = device or DEVICE
        print(f'[predictor] device={self.device}')
        self.model = MultiTaskDetector(EMBED_DIM).to(self.device)
        sd = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        self.model.load_state_dict(sd)
        self.model.eval()
        print(f'[predictor] model loaded from {model_path}')
        self.preprocessor = VideoPreprocessor(device=self.device)

    @torch.no_grad()
    def predict(self, video_path):
        t0 = time.time()
        data, info = self.preprocessor.process(video_path)
        info['preprocess_s'] = round(time.time() - t0, 2)
        if data is None:
            return None, info

        faces = data['faces'].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        faces = (faces - IMG_MEAN) / IMG_STD
        lips  = data['lips'].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        mel   = data['mel'][None]  # (1, 64, MEL_T)

        faces_t = torch.from_numpy(faces).float().unsqueeze(0).to(self.device)  # (1, N, 3, H, W)
        lips_t  = torch.from_numpy(lips).float().unsqueeze(0).to(self.device)
        audio_t = torch.from_numpy(data['audio']).float().unsqueeze(0).to(self.device)
        mel_t   = torch.from_numpy(mel).float().unsqueeze(0).to(self.device)

        t1 = time.time()
        if self.device.type == 'cuda':
            with autocast():
                preds = self.model(faces_t, audio_t, lips_t, mel_t)
        else:
            preds = self.model(faces_t, audio_t, lips_t, mel_t)

        scores = {
            'video': float(torch.sigmoid(preds['video']).item()),
            'audio': float(torch.sigmoid(preds['audio']).item()),
            'any':   float(torch.sigmoid(preds['any']).item()),
        }
        info['inference_s'] = round(time.time() - t1, 2)
        info['total_s'] = round(time.time() - t0, 2)
        return scores, info


def interpret(scores, threshold=0.5):
    """3 head skorundan insan dilinde karar üret."""
    v = scores['video']; a = scores['audio']; y = scores['any']
    is_video_fake = v > threshold
    is_audio_fake = a > threshold
    is_any_fake   = y > threshold

    if not is_any_fake:
        label = 'GERÇEK'
        sub   = 'Modeller hem video hem ses akışlarını doğal buldu.'
        sev   = 'safe'
    else:
        if is_video_fake and is_audio_fake:
            label = 'SAHTE (ikisi)'
            sub   = 'Hem görüntü hem ses manipülasyonu tespit edildi.'
        elif is_video_fake:
            label = 'SAHTE (görüntü)'
            sub   = 'Görüntü manipülasyonu tespit edildi (örn. dudak/yüz değişimi).'
        elif is_audio_fake:
            label = 'SAHTE (ses)'
            sub   = 'Sentetik ses tespit edildi (örn. ses klonu / TTS).'
        else:
            label = 'SAHTE (toplam karar)'
            sub   = 'Toplam karar verici sahte buldu, modaliteler arası tutarsızlık.'
        sev = 'danger'

    return {'label': label, 'sub': sub, 'severity': sev}


def category_from_path(video_path):
    """Dosya yolunda kategori varsa çıkar (ground truth)."""
    p = video_path.replace('\\', '/').lower()
    if 'realvideo-realaudio' in p: return ('R-R', 'gerçek', 'gerçek', False)
    if 'fakevideo-realaudio' in p: return ('F-R', 'sahte',  'gerçek', True)
    if 'realvideo-fakeaudio' in p: return ('R-F', 'gerçek', 'sahte',  True)
    if 'fakevideo-fakeaudio' in p: return ('F-F', 'sahte',  'sahte',  True)
    return (None, None, None, None)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('model_path')
    p.add_argument('video_path')
    a = p.parse_args()
    pred = DeepfakePredictor(a.model_path)
    scores, info = pred.predict(a.video_path)
    print('SCORES:', scores)
    print('INFO:', info)
    if scores:
        print('DECISION:', interpret(scores))
        print('GROUND TRUTH (path-derived):', category_from_path(a.video_path))
