"""Partición por sujeto, no por imagen ni por fotograma.

Si una misma persona aparece en train y en test, el modelo puede "reconocerla"
y los resultados salen inflados. Por eso cada sujeto cae en un solo conjunto.
"""
import numpy as np


def split_subjects(subjects, ratios=(0.70, 0.15, 0.15), seed: int = 42) -> dict:
    """Devuelve {sujeto: 'train'|'val'|'test'} con una semilla fija."""
    uniq = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    mapping = {}
    for i, s in enumerate(uniq):
        mapping[s] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return mapping


def assert_no_leakage(df, subject_col="subject", split_col="split"):
    """Reviso que ningún sujeto haya quedado en más de un split."""
    g = df.groupby(subject_col)[split_col].nunique()
    bad = g[g > 1]
    assert bad.empty, f"Sujetos en más de un split: {list(bad.index)}"
