"""Load MER-Prefer pairs from the official CSV files (name, a1, a2[, preference]) or JSONL manifests.

Media paths come from the video_path / audio_path / transcript_path columns, or from the
video/, audio/ and transcripts/ folders next to the manifest.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LABELS = {"a1": 0, "a2": 1}
MEDIA = (("video_path", "video", ".mp4"), ("audio_path", "audio", ".wav"), ("transcript_path", "transcripts", ".json"))


@dataclass
class Pair:
    pair_id: int
    gold: int  # 0 = a1 preferred, 1 = a2 preferred, -1 = unlabeled
    len0: float
    len1: float
    bt0: float
    bt1: float
    sample_id: str = ""
    cand0_text: str = ""
    cand1_text: str = ""
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None
    asr_text: str = ""


def _media_path(row: dict, key: str, base_dir: Path, folder: str, suffix: str, name: str) -> Optional[str]:
    value = row.get(key)
    if value:
        raw = Path(str(value))
        candidates = [raw] if raw.is_absolute() else [base_dir / raw, base_dir / folder / raw.name]
        for path in candidates:
            if path.exists():
                return str(path.resolve())
        return str(candidates[0].resolve())
    path = base_dir / folder / f"{name}{suffix}"
    return str(path.resolve()) if name and path.exists() else None


def _transcript_text(path: Optional[str]) -> str:
    if not path or not Path(path).exists():
        return ""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    text = obj.get("text", "")
    if text:
        return str(text).strip()
    return " ".join(str(seg.get("text", "")).strip() for seg in obj.get("segments") or []).strip()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        if path.suffix.lower() == ".csv":
            return list(csv.DictReader(f))
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(path: str) -> list[Pair]:
    path = Path(path)
    base_dir = path.resolve().parent
    pairs = []
    for index, row in enumerate(read_rows(path)):
        name = str(row.get("name", "")).strip()
        cand0, cand1 = str(row.get("a1", "")), str(row.get("a2", ""))
        media = {key: _media_path(row, key, base_dir, folder, suffix, name) for key, folder, suffix in MEDIA}
        pairs.append(Pair(
            pair_id=index,
            gold=LABELS.get(str(row.get("preference", "")).strip().lower(), -1),
            len0=float(len(cand0)),
            len1=float(len(cand1)),
            bt0=float(row.get("bt_a", 0.0)),
            bt1=float(row.get("bt_b", 0.0)),
            sample_id=name,
            cand0_text=cand0,
            cand1_text=cand1,
            asr_text=_transcript_text(media["transcript_path"]),
            **media,
        ))
    return pairs
