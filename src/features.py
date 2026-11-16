"""Métricas geométricas del rostro: EAR, MAR y PERCLOS.

Aquí solo uso numpy, así puedo probar estas funciones sin instalar TensorFlow
ni MediaPipe. Los índices son los del Face Mesh de MediaPipe (468 puntos).

El orden de los 6 puntos del ojo es:
p1 = esquina externa, p2/p3 = párpado superior, p4 = esquina interna,
p5/p6 = párpado inferior.
"""
from collections import deque
import numpy as np

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_H = (78, 308)                              # comisuras de la boca
MOUTH_V = ((81, 178), (13, 14), (311, 402))      # pares labio superior/inferior


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def eye_aspect_ratio(eye_pts) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|).

    Con el ojo abierto suele dar entre 0.25 y 0.35, y cerrado baja de 0.2,
    más o menos según la persona.
    """
    p1, p2, p3, p4, p5, p6 = eye_pts
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-9)


def mouth_aspect_ratio(pts) -> float:
    """MAR = suma de las 3 aperturas verticales / (2 * ancho de la boca).

    La boca cerrada suele estar entre 0.0 y 0.2, y en un bostezo pasa de 0.6.
    """
    vert = sum(_dist(pts[a], pts[b]) for a, b in MOUTH_V)
    horiz = _dist(pts[MOUTH_H[0]], pts[MOUTH_H[1]])
    return vert / (2.0 * horiz + 1e-9)


def compute_ear_mar(pts):
    """Promedio el EAR de los dos ojos y calculo el MAR.

    pts tiene que venir en píxeles (468, 2) y no normalizado, porque si no se
    pierde la proporción real del rostro.
    """
    ear_r = eye_aspect_ratio(pts[RIGHT_EYE])
    ear_l = eye_aspect_ratio(pts[LEFT_EYE])
    return (ear_r + ear_l) / 2.0, mouth_aspect_ratio(pts)


def landmarks_to_array(face_landmarks, width: int, height: int) -> np.ndarray:
    return np.array([(lm.x * width, lm.y * height) for lm in face_landmarks.landmark], dtype=np.float32)


def calibrate_threshold(open_ears, factor: float = 0.75) -> float:
    """Saco un umbral propio a partir del EAR con los ojos abiertos.

    Uso una fracción de la mediana porque no todas las personas tenemos el
    mismo ojo ni la misma cámara.
    """
    return float(factor * np.median(open_ears))


def eye_roi(gray_or_bgr, pts, eye_idx=RIGHT_EYE, scale: float = 1.8):
    """Recorte cuadrado alrededor del ojo, pensado para la CNN.

    Devuelve None si el recorte queda demasiado pequeño (por ejemplo, si el
    ojo está muy cerca del borde de la imagen).
    """
    e = pts[eye_idx]
    cx, cy = e.mean(axis=0)
    side = max(np.ptp(e[:, 0]), np.ptp(e[:, 1])) * scale
    half = side / 2.0
    h, w = gray_or_bgr.shape[:2]
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, w))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return gray_or_bgr[y0:y1, x0:x1]


class Perclos:
    """Porcentaje de tiempo con los ojos cerrados dentro de una ventana."""

    def __init__(self, window_seconds: float = 60.0):
        self.window = window_seconds
        self.samples = deque()  # guardo (tiempo, cerrado)

    def update(self, t: float, closed: bool) -> float:
        self.samples.append((t, closed))
        while self.samples and t - self.samples[0][0] > self.window:
            self.samples.popleft()
        if len(self.samples) < 2:
            return 0.0
        total = closed_t = 0.0
        pts = list(self.samples)
        for (t0, c0), (t1, _) in zip(pts[:-1], pts[1:]):
            dt = t1 - t0
            total += dt
            closed_t += dt if c0 else 0.0
        return closed_t / total if total > 0 else 0.0
