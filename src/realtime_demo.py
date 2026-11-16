"""Paso final: demo en vivo con la webcam.

Combino EAR, MAR y PERCLOS. Si le paso --cnn, también uso la CNN de ojos y
pido que las dos cosas coincidan para marcar el ojo como cerrado.

Uso:
    python src/realtime_demo.py
    python src/realtime_demo.py --cnn models/eye_cnn.keras

Teclas: q = salir, r = recalibrar.
"""
import argparse
import time
import cv2
import numpy as np
import mediapipe as mp

from config import *  # noqa
from features import (compute_ear_mar, landmarks_to_array, calibrate_threshold,
                      eye_roi, Perclos, RIGHT_EYE)


def load_cnn(path):
    if not path:
        return None
    from tensorflow import keras
    return keras.models.load_model(path)


def cnn_p_closed(model, frame_bgr, pts):
    roi = eye_roi(frame_bgr, pts, RIGHT_EYE)
    if roi is None:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img = np.repeat(img[..., None], 3, axis=-1).astype("float32")[None]
    return float(model.predict(img, verbose=0)[0, 0])


def main(args):
    model = load_cnn(args.cnn)
    cap = cv2.VideoCapture(args.camera)
    perclos = Perclos(PERCLOS_WINDOW_SECONDS)
    threshold, cal_start, cal_ears = None, time.monotonic(), []
    closed_since = yawn_since = None

    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.monotonic()
            h, w = frame.shape[:2]
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            msgs = []

            if not res.multi_face_landmarks:
                closed_since = yawn_since = None
                msgs.append("SIN ROSTRO")
            else:
                pts = landmarks_to_array(res.multi_face_landmarks[0], w, h)
                ear, mar = compute_ear_mar(pts)
                now = time.monotonic()

                if threshold is None:                      # calibración inicial
                    cal_ears.append(ear)
                    left = CALIBRATION_SECONDS - (now - cal_start)
                    msgs.append(f"CALIBRANDO: ojos abiertos {left:0.0f}s")
                    if left <= 0 and cal_ears:
                        threshold = calibrate_threshold(cal_ears, EAR_THRESHOLD_FACTOR)
                else:
                    closed = ear < threshold
                    if model is not None:                  # pido acuerdo CNN + EAR para no marcar falsos positivos
                        p = cnn_p_closed(model, frame, pts)
                        if p is not None:
                            closed = closed and p > 0.5
                    closed_since = (closed_since or now) if closed else None
                    p_clos = perclos.update(now, closed)

                    if closed_since and now - closed_since >= MICROSLEEP_SECONDS:
                        msgs.append("!!! MICROSUENO !!!")
                    if p_clos >= PERCLOS_ALERT:
                        msgs.append(f"FATIGA (PERCLOS {p_clos:.0%})")
                    yawn_since = (yawn_since or now) if mar > MAR_YAWN_THRESHOLD else None
                    if yawn_since and now - yawn_since >= YAWN_SECONDS:
                        msgs.append("BOSTEZO")
                    cv2.putText(frame, f"EAR {ear:.2f} (umbral {threshold:.2f})  MAR {mar:.2f}  PERCLOS {p_clos:.0%}",
                                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            for i, m in enumerate(msgs):
                color = (0, 0, 255) if "!" in m or "FATIGA" in m or "BOSTEZO" in m else (0, 200, 255)
                cv2.putText(frame, m, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            fps = 1.0 / max(time.monotonic() - t0, 1e-6)
            cv2.putText(frame, f"{fps:.0f} FPS", (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Detector de fatiga", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                threshold, cal_start, cal_ears = None, time.monotonic(), []
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cnn", default=None, help="ruta a models/eye_cnn.keras")
    main(ap.parse_args())
