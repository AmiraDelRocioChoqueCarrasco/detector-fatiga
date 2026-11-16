"""Paso 4: entreno una GRU con ventanas de características (no con imágenes).

Uso EAR normalizado, MAR y la variación del EAR. Es bastante liviano y entrena
en pocos minutos comparado con la CNN.

Uso: python src/train_temporal.py
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import *  # noqa
from splits import split_subjects


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["video", "frame"]).copy()
    # Normalizo el EAR de cada video con su mediana para que no afecte tanto
    # el tamaño de ojo de cada persona.
    df["ear_norm"] = df["ear"] / df.groupby("video")["ear"].transform("median")
    df["ear_delta"] = df.groupby("video")["ear_norm"].diff().fillna(0.0)
    return df


def make_windows(df: pd.DataFrame):
    win = int(FPS * WINDOW_SECONDS)
    stride = int(FPS * STRIDE_SECONDS)
    X, y, subj = [], [], []
    for (video, s), g in df.groupby(["video", "subject"]):
        arr = g[FEATURES].values.astype("float32")
        lab = g["label"].values
        for start in range(0, len(arr) - win + 1, stride):
            X.append(arr[start:start + win])
            y.append(int(round(lab[start:start + win].mean())))  # etiqueta mayoritaria
            subj.append(s)
    return np.array(X), np.array(y), np.array(subj)


def build_model(n_feat):
    return keras.Sequential([
        keras.Input((int(FPS * WINDOW_SECONDS), n_feat)),
        keras.layers.GRU(32),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="temporal_gru")


def main():
    keras.utils.set_random_seed(SEED)
    df = add_features(pd.read_csv(VIDEO_FEATURES))
    X, y, subj = make_windows(df)
    mapping = split_subjects(subj, SPLIT, SEED)
    sp = np.array([mapping[s] for s in subj])
    Xtr, ytr = X[sp == "train"], y[sp == "train"]
    Xva, yva = X[sp == "val"], y[sp == "val"]
    Xte, yte = X[sp == "test"], y[sp == "test"]
    print("ventanas train/val/test:", len(Xtr), len(Xva), len(Xte))

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    model = build_model(len(FEATURES))
    model.compile("adam", "binary_crossentropy",
                  metrics=[keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")])
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=30, batch_size=64,
              class_weight={0: float(cw[0]), 1: float(cw[1])},
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                       patience=5, restore_best_weights=True)])
    pred = (model.predict(Xte).ravel() >= 0.5).astype(int)
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred, target_names=["alerta", "fatiga"], digits=4))
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "temporal_gru.keras")


if __name__ == "__main__":
    main()
