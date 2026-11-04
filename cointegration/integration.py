"""Paso previo a la cointegración: verificar que cada log-precio sea I(1).

Cointegrar solo tiene sentido entre series integradas del mismo orden. Una serie
es I(1) si en niveles tiene raíz unitaria (ADF no rechaza H0) y su primera
diferencia es estacionaria (ADF sí rechaza H0).
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_pvalue(series: pd.Series) -> float:
    """p-value de la prueba ADF con constante y rezagos elegidos por AIC.

    H0: la serie tiene raíz unitaria (no es estacionaria).
    """
    return float(adfuller(series.dropna(), regression="c", autolag="AIC")[1])


def integration_order_table(log_prices: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Prueba ADF en niveles y en primeras diferencias para cada activo.

    Returns:
        Tabla con los p-values y la conclusión: "I(1)" si los niveles no son
        estacionarios y las diferencias sí.
    """
    rows = []
    for asset in log_prices.columns:
        p_level = adf_pvalue(log_prices[asset])
        p_diff = adf_pvalue(log_prices[asset].diff())
        is_i1 = p_level >= alpha and p_diff < alpha
        rows.append({"asset": asset, "adf_p_level": p_level, "adf_p_diff": p_diff,
                     "order": "I(1)" if is_i1 else "no I(1)"})
    return pd.DataFrame(rows).set_index("asset")


def all_integrated_of_order_one(table: pd.DataFrame) -> bool:
    """True si todos los activos resultaron I(1)."""
    return bool(np.all(table["order"] == "I(1)"))
