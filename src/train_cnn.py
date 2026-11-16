"""Paso 2: entreno la CNN que clasifica ojos abiertos/cerrados.

Uso MobileNetV2 con pesos de ImageNet y lo hago en dos fases: primero entreno
solo la cabeza con el backbone congelado y después ajusto las últimas capas.
Al final evalúo con sujetos que el modelo nunca vio.

Uso: python src/train_cnn.py [--export-tflite]
"""
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import *  # noqa


def make_ds(df, training: bool):
    paths = df["path"].values
    labels = df["label_closed"].values.astype("float32")

    def load(path, y):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model():
    aug = keras.Sequential([                     # aumentos, solo en entrenamiento
        keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
        keras.layers.RandomContrast(0.25),
        keras.layers.RandomRotation(15 / 360),
        keras.layers.RandomZoom(0.1),
        keras.layers.GaussianNoise(5.0),
    ], name="augmentation")

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = aug(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    base = keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="p_closed")(x)
    return keras.Model(inputs, out, name="eye_state_mobilenetv2"), base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                           keras.metrics.Recall(name="recall"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.AUC(name="auc")])


def main(export_tflite: bool):
    keras.utils.set_random_seed(SEED)
    df = pd.read_csv(MRL_MANIFEST)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    ds_tr, ds_va, ds_te = make_ds(tr, True), make_ds(va, False), make_ds(te, False)

    # En MRL hay más ojos abiertos que cerrados, así que balanceo las clases
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=tr.label_closed.values)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    print("class_weight:", class_weight)

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt = str(MODELS_DIR / "eye_cnn.keras")
    cbs = [keras.callbacks.ModelCheckpoint(ckpt, monitor="val_auc", mode="max", save_best_only=True),
           keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)]

    model, base = build_model()

    # Fase 1: backbone congelado, entreno solo la cabeza
    compile_model(model, LR_HEAD)
    model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS_HEAD, class_weight=class_weight, callbacks=cbs)

    # Fase 2: ajuste fino de las últimas capas. Las de BatchNorm las dejo
    # congeladas porque con pocos datos suelen desestabilizar el entrenamiento.
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, LR_FINETUNE)
    model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS_FINETUNE,
              class_weight=class_weight, callbacks=cbs)

    # Evaluación final con sujetos que el modelo nunca vio
    y_true = te.label_closed.values
    y_prob = model.predict(ds_te).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["abierto", "cerrado"], digits=4))

    if export_tflite:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]      # cuantización dinámica
        (MODELS_DIR / "eye_cnn.tflite").write_bytes(conv.convert())
        print("Exportado:", MODELS_DIR / "eye_cnn.tflite")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-tflite", action="store_true")
    main(ap.parse_args().export_tflite)
