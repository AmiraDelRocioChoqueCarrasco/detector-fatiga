"""Paso 1: armo el manifiesto de MRL Eye con su partición por sujeto.

Los archivos de MRL se llaman así: s0001_00001_0_0_0_0_0_01.png
  [0] sujeto
  [1] id de imagen
  [2] género
  [3] gafas
  [4] estado del ojo (0 = cerrado, 1 = abierto)
  [5] reflejos
  [6] iluminación
  [7] sensor

En el proyecto uso label_closed = 1 cuando el ojo está CERRADO, porque esa es
la clase que me interesa para fatiga.
"""
import pandas as pd
from config import MRL_RAW, MRL_MANIFEST, SEED, SPLIT
from splits import split_subjects, assert_no_leakage


def build_manifest() -> pd.DataFrame:
    rows = []
    for p in MRL_RAW.rglob("*.png"):
        parts = p.stem.split("_")
        if len(parts) < 5:
            continue
        rows.append({
            "path": str(p),
            "subject": parts[0],
            "glasses": int(parts[3]),
            "label_closed": 1 - int(parts[4]),
        })
    if not rows:
        raise SystemExit(f"No encontré imágenes .png en {MRL_RAW}. "
                         "Descarga MRL Eye y descomprímelo ahí.")
    df = pd.DataFrame(rows)
    mapping = split_subjects(df["subject"], SPLIT, SEED)
    df["split"] = df["subject"].map(mapping)
    assert_no_leakage(df)
    return df


if __name__ == "__main__":
    df = build_manifest()
    MRL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MRL_MANIFEST, index=False)
    print(f"{len(df)} imágenes de {df.subject.nunique()} sujetos -> {MRL_MANIFEST}")
    print(df.groupby("split").agg(imagenes=("path", "count"), sujetos=("subject", "nunique"),
                                  pct_cerrado=("label_closed", "mean")).round(3))
