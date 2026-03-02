"""
Caricamento e validazione del video di input.

Rifiuta immediatamente con messaggio esplicativo se:
- Frame rate < MIN_FPS
- Corpo non completamente visibile (valutato dopo pose estimation)
- Qualità/contrasto insufficiente (TODO: implementare)
- Prospettiva non laterale (TODO: implementare euristica)
"""

import cv2
from dataclasses import dataclass
from pathlib import Path

from src.config import MIN_FPS


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


def load_video(path: str) -> tuple[cv2.VideoCapture, VideoMeta]:
    """Apre il video e ne estrae i metadati. Lancia ValueError se non conforme."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video: {path}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps < MIN_FPS:
        cap.release()
        raise ValueError(
            f"Frame rate insufficiente: {fps:.1f} fps rilevati, minimo richiesto {MIN_FPS} fps.\n"
            f"Riprendi il video con frame rate più alto."
        )

    meta = VideoMeta(path=p, fps=fps, frame_count=frame_count, width=width, height=height)
    return cap, meta


def iter_frames(cap: cv2.VideoCapture):
    """Generatore che restituisce i frame uno alla volta (BGR numpy array)."""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
