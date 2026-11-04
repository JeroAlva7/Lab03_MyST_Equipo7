"""Filtro heurístico por correlación móvil (paso 2 de la sección 3.2).

Se usa la correlación de los log-retornos diarios, no de los precios: dos
precios con tendencia siempre parecen correlacionados (regresión espuria),
mientras que los retornos son estacionarios. La correlación es solo un
pre-filtro; la cointegración es el criterio definitivo.
"""
from itertools import combinations

import pandas as pd


def rolling_pair_correlation(log_prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Correlación móvil de log-retornos para cada par de activos.

    Returns:
        DataFrame con una columna "A-B" por par y una fila por fecha.
    """
    returns = log_prices.diff().dropna()
    series = {
        f"{a}-{b}": returns[a].rolling(window).corr(returns[b])
        for a, b in combinations(log_prices.columns, 2)
    }
    return pd.DataFrame(series).dropna()


def group_correlation(rolling_corr: pd.DataFrame, group: tuple[str, ...]) -> tuple[float, float]:
    """Correlación promedio y mínima del grupo, usando la mediana móvil de cada par.

    La mediana resume la correlación típica de cada par sin que la dominen los
    periodos de crisis, cuando todas las correlaciones se disparan.
    """
    medians = [rolling_corr[f"{a}-{b}"].median() for a, b in combinations(group, 2)]
    return float(sum(medians) / len(medians)), float(min(medians))
