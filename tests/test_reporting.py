import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.reporting import build_report, risk_level


def test_risk_level_low_when_any_below_threshold():
    assert risk_level({"video": 0.9, "audio": 0.1, "any": 0.2}) == "low"


def test_risk_level_high_when_any_crosses_and_signal_is_strong():
    assert risk_level({"video": 0.2, "audio": 0.91, "any": 0.92}) == "high"


def test_build_report_includes_findings_and_scores():
    report = build_report(
        {"video": 0.2, "audio": 0.8, "any": 0.85},
        {"label": "SAHTE (ses)", "sub": "Sentetik ses tespit edildi.", "severity": "danger"},
        {"face_prob": 0.95, "audio_rms": 0.03, "total_s": 4.2},
    )
    assert report["risk"] == "high"
    assert report["scores"]["audio"] == 0.8
    assert any("audio stream" in item for item in report["findings"])
