"""Sensibilidad de la estrategia a la Q del filtro 1 (análisis ex post, no se usa para elegir).

La Q del filtro 1 ("drift velocity" de los pesos) se fijó por diseño en 1e-8 antes del
backtest (ver README y docs/powell_formulacion.md). Este módulo responde qué habría pasado
con otras Q: para cada valor se recalculan ambos filtros, se recalibran los umbrales con el
mismo grid search en entrenamiento y se evalúa la estrategia en las tres particiones.

Sirve para mostrar el riesgo de optimizar Q junto con los umbrales: una Q que se ve mejor en
entrenamiento puede ser la que más se deteriora fuera de muestra (sobreajuste).
"""
import numpy as np
import pandas as pd

from backtest.grid_search import grid_search
from backtest.walk_forward import run_strategy, summarize_by_split
from cointegration.stability import stationarity_by_split
from kalman.pipeline import run_filters
from strategy.sizing import kalman_weights


def q_sensitivity(prices: pd.DataFrame, beta0: np.ndarray, n_train: int, kalman_settings: dict,
                  q_values: list[float], tradable: pd.Series, splits: pd.Series, grid: dict,
                  execution: dict) -> pd.DataFrame:
    """Recalcula filtros, grid y backtest para cada Q y resume el resultado por partición.

    Args:
        prices: precios con la pata normalizada primero (orden de beta0).
        beta0: vector de Johansen normalizado.
        n_train: fechas de entrenamiento.
        kalman_settings: config.KALMAN; aquí solo se sustituye "q_weights".
        q_values: valores de Q a comparar.
        tradable, splits, grid, execution: como en backtest.grid_search.

    Returns:
        Una fila por (Q, partición): umbrales elegidos en entrenamiento, Sharpe, retorno,
        operaciones, drawdown máximo, y ADF y vida media del spread de esa Q.
    """
    assets = list(prices.columns)
    rows = []
    for q in q_values:
        f1, f2, params = run_filters(np.log(prices), beta0, n_train, {**kalman_settings, "q_weights": q})
        weights, half_life = kalman_weights(f1, assets), params["filter2"]["half_life"]
        _, chosen = grid_search(prices, weights, f2["z"], tradable, splits, half_life, grid, execution)
        _, bt, trades = run_strategy(prices, weights, f2["z"], tradable, splits, chosen, half_life, execution)
        summary = summarize_by_split(bt, trades, tradable)
        spread = stationarity_by_split({"kalman": f1["spread"].where(tradable)}, splits).set_index("split")
        for split in ("train", "test", "val"):
            rows.append({"q_weights": q, "split": split, **chosen,
                         "sharpe": summary.loc[split, "sharpe"], "return": summary.loc[split, "return"],
                         "trades": summary.loc[split, "trades"],
                         "max_drawdown": summary.loc[split, "max_drawdown"],
                         "spread_adf_p": spread.loc[split, "adf_p"],
                         "spread_half_life": spread.loc[split, "half_life"]})
    return pd.DataFrame(rows)
