"""Pre-filtro de Engle-Granger (paso 3 de la sección 3.2).

Paso 1: OLS de un log-precio contra los demás. Paso 2: ADF sobre los residuales
con valores críticos de MacKinnon, que dependen del número de variables.
H0: no hay cointegración (los residuales tienen raíz unitaria).
"""
import pandas as pd
from statsmodels.tsa.stattools import coint


def engle_granger_group(log_prices: pd.DataFrame, group: tuple[str, ...]) -> dict:
    """Aplica Engle-Granger tomando a cada activo del grupo como variable dependiente.

    El resultado de Engle-Granger depende de qué activo se pone a la izquierda de la
    regresión, así que se prueban todas las orientaciones y se reportan todas.
    Como es solo un pre-filtro, el grupo avanza si alguna orientación rechaza H0.

    Returns:
        Diccionario con el p-value mínimo, la orientación que lo produjo y los
        p-values de todas las orientaciones.
    """
    pvalues = {}
    for dependent in group:
        others = [a for a in group if a != dependent]
        _, pvalue, _ = coint(log_prices[dependent], log_prices[others], trend="c")
        pvalues[dependent] = float(pvalue)
    best = min(pvalues, key=pvalues.get)
    return {"eg_pvalue": pvalues[best], "eg_dependent": best, "eg_all_pvalues": pvalues}
