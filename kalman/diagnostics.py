"""Diagnóstico de los filtros: las innovaciones normalizadas deben ser ruido blanco con varianza 1 (S13)."""
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


def innovation_diagnostics(innovations: pd.Series, lags: int = 10) -> dict:
    """Varianza, autocorrelación de orden 1 y prueba de Ljung-Box de las innovaciones normalizadas.

    Ljung-Box: H0 = no hay autocorrelación en los primeros `lags` rezagos.
    """
    x = innovations.dropna()
    centered = x - x.mean()
    acf1 = float((centered.iloc[1:].to_numpy() @ centered.iloc[:-1].to_numpy()) / (centered @ centered))
    lb_pvalue = float(acorr_ljungbox(x, lags=[lags])["lb_pvalue"].iloc[0])
    return {"n": int(len(x)), "variance": float(x.var()), "acf1": acf1,
            "acf1_band": float(2 / np.sqrt(len(x))), "ljung_box_p": lb_pvalue}
