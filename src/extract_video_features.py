"""Paso 3: convierto los videos en una tabla de EAR/MAR por fotograma.

Cada dataset nombra los videos de forma distinta, así que infer_subject() e
infer_label() son solo un ejemplo. Antes de procesar hay que mirar cómo vienen
las carpetas y ajustarlas.

Uso: python src/extract_video_features.py
"""
import cv2
import pandas as pd
import mediapipe as mp
from pathlib import Path
from config import VIDEOS_RAW, VIDEO_FEATURES
from features import compute_ear_mar, landmarks_to_array

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def infer_subject(path: Path) -> str:
    """TODO: ajustar. Por ahora uso carpeta + prefijo del nombre del archivo."""
    return f"{path.parent.name}_{path.stem.split('_')[0]}"


def infer_label(path: Path) -> int:
    """TODO: ajustar. 0 = alerta, 1 = fatiga/somnolencia.

    El ejemplo supone que los videos están ordenados en carpetas como
    'alert/' y 'drowsy/'. Si no es así, esto etiqueta mal y las métricas no
    sirven.
    """
    name = str(path).lower()
    if "drowsy" in name or "yawn" in name:
        return 1
    return 0


def process_video(path: Path, face_mesh):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            pts = landmarks_to_array(res.multi_face_landmarks[0], w, h)
            ear, mar = compute_ear_mar(pts)
            rows.append({"frame": i, "t": i / fps, "ear": ear, "mar": mar})
        i += 1
    cap.release()
    return rows


def main():
    videos = [p for p in VIDEOS_RAW.rglob("*") if p.suffix.lower() in VIDEO_EXT]
    if not videos:
        raise SystemExit(f"No hay videos en {VIDEOS_RAW}")
    out = []
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
        for v in videos:
            print("Procesando", v.name)
            for r in process_video(v, fm):
                r.update(video=v.name, subject=infer_subject(v), label=infer_label(v))
                out.append(r)
    VIDEO_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(VIDEO_FEATURES, index=False)
    print("Guardado:", VIDEO_FEATURES)


if __name__ == "__main__":
    main()
