"""
Flask demo — Grup 11 Multimodal Deepfake Tespit
Local, Colab gerekmez, model checkpoint'inden direkt yükler.

Çalıştırmak için:
    python app.py
Açılacak adres: http://127.0.0.1:5000
"""
import os, sys, json, time, uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

from inference import DeepfakePredictor, interpret, category_from_path

HERE = Path(__file__).parent.resolve()
MODEL_PATH   = os.environ.get('MODEL_PATH', r'C:\Users\Tombulteke\Downloads\best_model.pt')
SAMPLES_DIR  = HERE / 'samples'
UPLOADS_DIR  = HERE / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

# Test set aggregate metrics (gösterim için)
AGGREGATE = {
    'test_n': 3276,
    'video_auc': 0.9991, 'video_f1': 0.9994,
    'audio_auc': 0.9982, 'audio_f1': 0.9875,
    'any_auc':   0.9994, 'any_f1':   0.9984,
    'per_category': {
        'R-R': {'n': 75,   'any_recall': None, 'label': 'Real-Real (kontrol)'},
        'F-R': {'n': 1485, 'any_recall': 0.9973, 'label': 'Fake video, gerçek ses (Wav2Lip)'},
        'R-F': {'n': 75,   'any_recall': 0.9333, 'label': 'Gerçek video, sahte ses (TTS)'},
        'F-F': {'n': 1641, 'any_recall': 1.0000, 'label': 'Hem video hem ses sahte'},
    },
}

print('[app] Model yükleniyor (~10-15 sn)...')
predictor = DeepfakePredictor(MODEL_PATH)
print('[app] hazır.')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024   # 200 MB upload limit


def list_samples():
    """samples/ klasöründeki mp4'leri kategorisiyle birlikte listele."""
    out = []
    for p in sorted(SAMPLES_DIR.glob('**/*.mp4')):
        rel = p.relative_to(SAMPLES_DIR).as_posix()
        cat, vfake, afake, any_fake = category_from_path(str(p))
        out.append({
            'name': p.name,
            'rel':  rel,
            'category': cat,
            'video_truth': vfake,
            'audio_truth': afake,
            'any_truth':   any_fake,
        })
    return out


@app.route('/')
def index():
    return render_template('index.html',
                            samples=list_samples(),
                            aggregate=AGGREGATE,
                            model_device=str(predictor.device))


@app.route('/samples/<path:filename>')
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOADS_DIR, filename)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Body:
       - mode='sample' + sample_name → samples/<name> üzerinde inference
       - mode='upload' + uploaded file → /uploads'a kaydet, inference
    """
    mode = request.form.get('mode', 'sample')
    if mode == 'sample':
        sample_name = request.form.get('sample', '').strip()
        video_path = SAMPLES_DIR / sample_name
        if not video_path.exists():
            return jsonify({'ok': False, 'error': f'sample not found: {sample_name}'}), 404
        ground = category_from_path(str(video_path))
        video_url = f'/samples/{sample_name}'
    elif mode == 'upload':
        f = request.files.get('file')
        if not f or not f.filename:
            return jsonify({'ok': False, 'error': 'no file uploaded'}), 400
        # Safe name
        safe = f"{uuid.uuid4().hex[:8]}_{Path(f.filename).name}"
        video_path = UPLOADS_DIR / safe
        f.save(video_path)
        # Try to detect ground truth from original filename
        ground = category_from_path(f.filename)
        video_url = f'/uploads/{safe}'
    else:
        return jsonify({'ok': False, 'error': 'unknown mode'}), 400

    t0 = time.time()
    scores, info = predictor.predict(str(video_path))
    if scores is None:
        return jsonify({'ok': False, 'error': 'preprocess failed: ' + str(info.get('fail_reason', '?')),
                         'info': info}), 422
    decision = interpret(scores)
    # Match with ground truth if known
    truth = None
    correct = None
    if ground[0]:
        truth = {
            'category': ground[0],
            'video_fake': bool(ground[3]) and (ground[1] == 'sahte'),
            'audio_fake': bool(ground[3]) and (ground[2] == 'sahte'),
            'any_fake':   bool(ground[3]),
        }
        # check video / audio decision against truth
        v_pred = scores['video'] > 0.5
        a_pred = scores['audio'] > 0.5
        y_pred = scores['any']   > 0.5
        correct = {
            'video': v_pred == truth['video_fake'],
            'audio': a_pred == truth['audio_fake'],
            'any':   y_pred == truth['any_fake'],
        }

    return jsonify({
        'ok': True,
        'video_url': video_url,
        'scores': scores,
        'decision': decision,
        'info': info,
        'ground_truth': truth,
        'correct': correct,
        'elapsed_total': round(time.time() - t0, 2),
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
