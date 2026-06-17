"""Human-readable report helpers shared by the CLI and Flask demo."""

from __future__ import annotations


def risk_level(scores: dict[str, float], threshold: float = 0.5) -> str:
    """Return a compact risk label from the model scores."""
    any_score = float(scores.get("any", 0.0))
    video_score = float(scores.get("video", 0.0))
    audio_score = float(scores.get("audio", 0.0))
    if any_score < threshold:
        return "low"
    if max(video_score, audio_score, any_score) >= 0.85:
        return "high"
    return "medium"


def _top_signal(scores: dict[str, float]) -> str:
    labels = {
        "video": "visual stream",
        "audio": "audio stream",
        "any": "fusion head",
    }
    key = max(scores, key=lambda item: float(scores[item]))
    return labels.get(key, key)


def build_report(
    scores: dict[str, float],
    decision: dict[str, str],
    info: dict | None = None,
    ground_truth: dict | None = None,
    threshold: float = 0.5,
) -> dict:
    """Build a structured explanation payload for a single analyzed video."""
    info = info or {}
    level = risk_level(scores, threshold=threshold)
    top_signal = _top_signal(scores)
    video_fake = float(scores.get("video", 0.0)) >= threshold
    audio_fake = float(scores.get("audio", 0.0)) >= threshold
    any_fake = float(scores.get("any", 0.0)) >= threshold

    findings = []
    if video_fake:
        findings.append("The visual stream crossed the fake threshold.")
    if audio_fake:
        findings.append("The audio stream crossed the fake threshold.")
    if any_fake and not findings:
        findings.append("The fusion head crossed the fake threshold despite weaker single-stream scores.")
    if not findings:
        findings.append("All heads stayed below the fake threshold.")

    if info.get("face_prob") is not None:
        findings.append(f"Face detection confidence was {float(info['face_prob']):.3f}.")
    if info.get("audio_rms") is not None:
        findings.append(f"Audio RMS was {float(info['audio_rms']):.4f}.")

    summary = (
        f"{decision.get('label', 'UNKNOWN')} decision with {level} risk. "
        f"Strongest signal: {top_signal}."
    )

    report = {
        "summary": summary,
        "risk": level,
        "threshold": threshold,
        "scores": {
            "video": round(float(scores.get("video", 0.0)), 4),
            "audio": round(float(scores.get("audio", 0.0)), 4),
            "any": round(float(scores.get("any", 0.0)), 4),
        },
        "decision": decision,
        "findings": findings,
        "timing": {
            "preprocess_s": info.get("preprocess_s"),
            "inference_s": info.get("inference_s"),
            "total_s": info.get("total_s"),
        },
    }
    if ground_truth:
        report["ground_truth"] = ground_truth
    return report
