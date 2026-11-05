"""Estabilidad de la relación de cointegración en el tiempo (pregunta central del lab).

Dos evidencias de si la relación es estructural o un artefacto de haber probado muchos grupos:
1. Johansen móvil: el primer eigenvector y su estadístico de traza en ventanas que avanzan.
2. Estacionariedad fuera de muestra: ADF y vida media del spread estático (Johansen) y del
   dinámico (Kalman) en entrenamiento, prueba y validación.
"""
import numpy as np
import pandas as pd

from cointegration.integration import adf_pvalue
from cointegration.johansen import johansen_test
from cointegration.selection import half_life


def unit_normalize(vector: np.ndarray, anchor: int = 0) -> np.ndarray:
    """Escala el eigenvector a norma 1 con la pata `anchor` positiva.

    En ventanas cortas el componente de la pata normalizada puede acercarse a cero;
    dividir entre él haría explotar los demás pesos. La norma unitaria fija la escala
    y el signo sin ese problema, así que es la que se usa para graficar en el tiempo.
    """
    unit = vector / np.linalg.norm(vector)
    return unit * np.sign(unit[anchor])


def rolling_johansen(log_prices: pd.DataFrame, window: int, step: int, k_ar_diff: int) -> pd.DataFrame:
    """Corre Johansen en ventanas móviles de `window` días que avanzan `step` días.

    Cada fila usa solo datos hasta su fecha (causal). Columnas: primer eigenvector
    (norma 1, primera pata positiva), primer eigenvalor, traza con r0 = 0, su valor
    crítico al 95% y si la ventana rechaza "no cointegración".
    """
    rows = []
    for end in range(window, len(log_prices) + 1, step):
        joh = johansen_test(log_prices.iloc[end - window:end], k_ar_diff)
        vector = unit_normalize(joh["eigenvectors"][:, 0])
        rows.append({"date": log_prices.index[end - 1],
                     **{f"evec_{a}": float(v) for a, v in zip(log_prices.columns, vector)},
                     "eigenvalue": float(joh["eigenvalues"][0]), "trace": float(joh["trace"][0]),
                     "trace_cv95": float(joh["trace_cv95"][0]),
                     "cointegrated": bool(joh["trace"][0] > joh["trace_cv95"][0])})
    return pd.DataFrame(rows).set_index("date")


def stationarity_by_split(spreads: dict[str, pd.Series], labels: pd.Series) -> pd.DataFrame:
    """ADF, vida media y media de cada spread en cada partición.

    Args:
        spreads: {"estatico": spread de Johansen, "kalman": spread del filtro 1}.
        labels: partición de cada fecha ("train", "test", "val").
    """
    rows = []
    for name, spread in spreads.items():
        for split in ("train", "test", "val"):
            segment = spread[labels == split].dropna()
            rows.append({"spread": name, "split": split, "adf_p": adf_pvalue(segment),
                         "half_life": half_life(segment), "mean": float(segment.mean())})
    return pd.DataFrame(rows)
