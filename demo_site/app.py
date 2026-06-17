"""Flask demo for the multimodal deepfake detector."""
import os, sys, time, uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from deepfake_detector.reporting import build_report
from inference import DeepfakePredictor, interpret, category_from_path

MODEL_PATH   = os.environ.get('MODEL_PATH', '').strip()
SAMPLES_DIR  = HERE / 'samples'
UPLOADS_DIR  = HERE / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}

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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024   # 200 MB upload limit
predictor = None
predictor_error = None


def allowed_video(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def safe_child(base, user_path):
    """Resolve a user-controlled path and ensure it stays under base."""
    candidate = (base / user_path).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def load_predictor():
    global predictor, predictor_error
    if predictor is not None or predictor_error is not None:
        return predictor
    if not MODEL_PATH:
        predictor_error = 'MODEL_PATH environment variable is not set.'
        return None
    model_file = Path(MODEL_PATH).expanduser()
    if not model_file.exists():
        predictor_error = f'Model checkpoint not found: {model_file}'
        return None
    try:
        print('[app] Loading model (~10-15 sec)...')
        predictor = DeepfakePredictor(str(model_file))
        print('[app] ready.')
    except Exception as exc:
        predictor_error = f'Model load failed: {exc}'
        print(f'[app] {predictor_error}')
    return predictor


def list_samples():
    """List sample videos with path-derived categories."""
    out = []
    for p in sorted(SAMPLES_DIR.glob('**/*')):
        if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
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
    pred = load_predictor()
    return render_template('index.html',
                            samples=list_samples(),
                            aggregate=AGGREGATE,
                            model_ready=pred is not None,
                            model_error=predictor_error,
                            model_path=MODEL_PATH or None,
                            model_device=str(pred.device) if pred else 'not loaded')


@app.route('/samples/<path:filename>')
def serve_sample(filename):
    path = safe_child(SAMPLES_DIR, filename)
    if path is None or not path.exists() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({'ok': False, 'error': 'sample not found'}), 404
    return send_from_directory(SAMPLES_DIR, path.relative_to(SAMPLES_DIR).as_posix())


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    path = safe_child(UPLOADS_DIR, filename)
    if path is None or not path.exists() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({'ok': False, 'error': 'upload not found'}), 404
    return send_from_directory(UPLOADS_DIR, path.relative_to(UPLOADS_DIR).as_posix())


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Body:
       - mode='sample' + sample_name → samples/<name> üzerinde inference
       - mode='upload' + uploaded file → /uploads'a kaydet, inference
    """
    pred = load_predictor()
    if pred is None:
        return jsonify({'ok': False, 'error': predictor_error or 'model is not available'}), 503

    mode = request.form.get('mode', 'sample')
    if mode == 'sample':
        sample_name = request.form.get('sample', '').strip()
        if not allowed_video(sample_name):
            return jsonify({'ok': False, 'error': 'unsupported sample type'}), 400
        video_path = safe_child(SAMPLES_DIR, sample_name)
        if video_path is None or not video_path.exists():
            return jsonify({'ok': False, 'error': f'sample not found: {sample_name}'}), 404
        ground = category_from_path(str(video_path))
        video_url = f'/samples/{sample_name}'
    elif mode == 'upload':
        f = request.files.get('file')
        if not f or not f.filename:
            return jsonify({'ok': False, 'error': 'no file uploaded'}), 400
        if not allowed_video(f.filename):
            return jsonify({'ok': False, 'error': 'unsupported file type'}), 400
        clean_name = secure_filename(Path(f.filename).name)
        if not clean_name:
            return jsonify({'ok': False, 'error': 'invalid filename'}), 400
        safe = f"{uuid.uuid4().hex[:8]}_{clean_name}"
        video_path = UPLOADS_DIR / safe
        f.save(video_path)
        ground = category_from_path(f.filename)
        video_url = f'/uploads/{safe}'
    else:
        return jsonify({'ok': False, 'error': 'unknown mode'}), 400

    t0 = time.time()
    scores, info = pred.predict(str(video_path))
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
    report = build_report(scores, decision, info, truth)

    return jsonify({
        'ok': True,
        'video_url': video_url,
        'scores': scores,
        'decision': decision,
        'report': report,
        'info': info,
        'ground_truth': truth,
        'correct': correct,
        'elapsed_total': round(time.time() - t0, 2),
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
