"""Calibración de umbrales por grid search sobre la partición de entrenamiento (sección 3.6).

Para cada combinación de z_entrada, z_salida y múltiplo de vida media (tenencia máxima) se
corre el backtest completo con costos SOLO con datos de entrenamiento y se mide su Sharpe.
Prueba y validación no se usan para elegir nada: se evalúan después con los valores ganadores.

Para no premiar combinaciones que ganan con muy pocas operaciones (resultado poco confiable),
solo compiten las que hacen al menos `min_trades` operaciones en entrenamiento.
"""
from itertools import product

import pandas as pd

from backtest.engine import add_drawdown, run_backtest
from backtest.trades import build_trade_log
from performance.metrics import daily_returns, sharpe_ratio
from strategy.signals import generate_signals, max_holding_days


def evaluate(prices: pd.DataFrame, weights: pd.DataFrame, z: pd.Series, tradable: pd.Series,
             splits: pd.Series, params: dict, half_life: float, costs: dict) -> dict:
    """Corre señales + backtest con un juego de parámetros y resume el resultado."""
    max_holding = max_holding_days(half_life, params["max_holding_multiple"])
    signals = generate_signals(z, tradable, params["z_entry"], params["z_exit"],
                               params["z_stop"], max_holding)
    bt = run_backtest(prices, weights, signals["direction"], **costs)
    trades = build_trade_log(bt, signals["event"], splits)
    returns = daily_returns(bt["equity"][tradable])
    return {**params, "max_holding": max_holding, "trades": len(trades),
            "net_pnl": float(bt["equity"].iloc[-1] - bt["equity"].iloc[0]),
            "sharpe": sharpe_ratio(returns),
            "max_drawdown": float(add_drawdown(bt["equity"]).min()),
            "win_rate": float((trades["net_pnl"] > 0).mean()) if len(trades) else 0.0}


def grid_search(prices: pd.DataFrame, weights: pd.DataFrame, z: pd.Series, tradable: pd.Series,
                splits: pd.Series, half_life: float, grid: dict, costs: dict) -> tuple[pd.DataFrame, dict]:
    """Evalúa todas las combinaciones del grid con datos de entrenamiento.

    Args:
        prices, weights, z, tradable, splits: series completas; aquí se recortan a "train".
        half_life: vida media del spread operado (filtro 2).
        grid: listas de "z_entry", "z_exit", "max_holding_multiple", más "z_stop" y "min_trades".
        costs: argumentos de run_backtest (capital, fraction, commission, borrow_rate, trading_days).

    Returns:
        (tabla con una fila por combinación ordenada por Sharpe, parámetros ganadores).
    """
    train = (splits == "train").to_numpy()
    data = [prices[train], weights[train], z[train], tradable[train], splits[train]]
    rows = [evaluate(*data, {"z_entry": z_in, "z_exit": z_out, "z_stop": grid["z_stop"],
                             "max_holding_multiple": mult}, half_life, costs)
            for z_in, z_out, mult in product(grid["z_entry"], grid["z_exit"], grid["max_holding_multiple"])
            if z_out < z_in]

    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False, ignore_index=True)
    table["eligible"] = table["trades"] >= grid["min_trades"]
    best = table[table["eligible"]].iloc[0]
    chosen = {k: float(best[k]) for k in ("z_entry", "z_exit", "z_stop", "max_holding_multiple")}
    return table, chosen
