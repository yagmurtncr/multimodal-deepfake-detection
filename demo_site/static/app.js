// Demo client logic
(function() {
    const $ = (id) => document.getElementById(id);
    const resultSection = $('result-section');
    const loader = $('loader');
    const resultBody = $('result-body');
    const errorBox = $('result-error');

    function showLoading() {
        resultSection.classList.remove('result-hidden');
        loader.classList.remove('hidden');
        resultBody.classList.add('hidden');
        errorBox.classList.add('hidden');
        // scroll into view
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function showError(msg) {
        loader.classList.add('hidden');
        resultBody.classList.add('hidden');
        errorBox.classList.remove('hidden');
        errorBox.textContent = '⚠️  ' + msg;
    }

    function severityFor(score) {
        if (score >= 0.85) return 'high';
        if (score >= 0.5)  return 'mid';
        return 'low';
    }

    function showResults(data) {
        loader.classList.add('hidden');
        errorBox.classList.add('hidden');
        resultBody.classList.remove('hidden');

        // Video
        const v = $('result-video');
        v.src = data.video_url;
        v.load();

        // Decision box
        const decBox = $('decision-box');
        decBox.classList.remove('severity-safe', 'severity-danger');
        decBox.classList.add('severity-' + data.decision.severity);
        $('decision-label').textContent = data.decision.label;
        $('decision-sub').textContent = data.decision.sub;

        // Meta — ground truth if known
        let metaText = '';
        if (data.ground_truth && data.ground_truth.category) {
            const gt = data.ground_truth;
            const dt = data.decision.label.includes('SAHTE') === gt.any_fake;
            const tag = dt ? '✓ doğru' : '✗ yanlış';
            metaText = `Ground truth: ${gt.category} (any_fake=${gt.any_fake ? 'evet' : 'hayır'}) · ${tag}`;
        } else {
            metaText = `Ground truth: bilinmiyor (kategori dosya adından çıkarılamadı)`;
        }
        $('decision-meta').textContent = metaText;

        // 3 head scores with animated bars
        ['video', 'audio', 'any'].forEach((task, i) => {
            const sc = data.scores[task];
            const pct = Math.round(sc * 1000) / 10;
            const bar = $('bar-' + task);
            bar.style.width = '0%';
            bar.classList.remove('high', 'mid', 'low');
            bar.classList.add(severityFor(sc));
            // Animate in
            setTimeout(() => { bar.style.width = pct + '%'; }, 80 * (i + 1));

            $('score-' + task).textContent = sc.toFixed(4);

            // Truth match indicator
            const tEl = $('truth-' + task);
            if (data.ground_truth && data.correct) {
                const gtKey = task === 'any' ? 'any_fake' : (task + '_fake');
                const gt = data.ground_truth[gtKey];
                const ok = data.correct[task];
                tEl.textContent = `Gerçek: ${gt ? 'sahte' : 'gerçek'} · ${ok ? '✓' : '✗'}`;
                tEl.className = 'head-truth ' + (ok ? 'ok' : 'bad');
            } else {
                tEl.textContent = '';
                tEl.className = 'head-truth';
            }
        });

        // Info row
        const info = data.info || {};
        $('info-face').textContent  = info.face_prob != null ? info.face_prob.toFixed(3) : '—';
        $('info-rms').textContent   = info.audio_rms != null ? info.audio_rms.toFixed(4) : '—';
        $('info-pre').textContent   = (info.preprocess_s ?? '—') + ' s';
        $('info-inf').textContent   = (info.inference_s  ?? '—') + ' s';
        $('info-total').textContent = (data.elapsed_total ?? '—') + ' s';
    }

    async function analyze(payload) {
        showLoading();
        try {
            const res = await fetch('/api/analyze', { method: 'POST', body: payload });
            const data = await res.json();
            if (!data.ok) {
                showError(data.error || 'Bilinmeyen hata');
                return;
            }
            showResults(data);
        } catch (e) {
            showError('İstek hatası: ' + e.message);
        }
    }

    // Sample card click
    document.querySelectorAll('.sample-card').forEach(btn => {
        btn.addEventListener('click', () => {
            const fd = new FormData();
            fd.append('mode', 'sample');
            fd.append('sample', btn.dataset.sample);
            analyze(fd);
        });
    });

    // Upload form
    const form = $('upload-form');
    if (form) {
        form.addEventListener('submit', (ev) => {
            ev.preventDefault();
            const fileInput = $('file-input');
            if (!fileInput.files.length) {
                showError('Önce bir video seçin');
                return;
            }
            const fd = new FormData();
            fd.append('mode', 'upload');
            fd.append('file', fileInput.files[0]);
            analyze(fd);
        });
    }
})();
