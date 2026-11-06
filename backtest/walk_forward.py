"""Evaluación walk-forward en prueba y validación (sección 3.6).

Protocolo (sin look-ahead):
    1. Los umbrales se eligen con el grid search SOLO en entrenamiento.
    2. Con esos parámetros congelados se corre una sola simulación continua de toda la muestra.
       Cada día solo se usa información hasta ese día: los filtros de Kalman se actualizan en
       línea y las señales de t se ejecutan en t+1. Nada se recalibra con prueba ni validación
       (declaración de alcance del charter).
    3. El resultado se reporta por partición. Prueba y validación son genuinamente fuera de muestra.
"""
import pandas as pd

from backtest.engine import add_drawdown, run_backtest
from backtest.trades import build_trade_log
from performance.metrics import daily_returns, sharpe_ratio
from strategy.signals import generate_signals, max_holding_days

SPLITS = ("train", "test", "val")


def execution_settings(cfg) -> dict:
    """Parámetros de ejecución de config.py: capital, 80%, comisión, préstamo y rebalanceo."""
    return {"capital": cfg.INITIAL_CAPITAL, "fraction": cfg.CAPITAL_FRACTION,
            "commission": cfg.COMMISSION, "borrow_rate": cfg.BORROW_RATE_ANNUAL,
            "trading_days": cfg.TRADING_DAYS, "rebalance_band": cfg.REBALANCE_BAND}


def run_strategy(prices: pd.DataFrame, weights: pd.DataFrame, z: pd.Series, tradable: pd.Series,
                 splits: pd.Series, params: dict, half_life: float,
                 execution: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Señales -> backtest -> bitácora con un juego de parámetros.

    Args:
        params: "z_entry", "z_exit", "z_stop" y "max_holding_multiple".
        half_life: vida media del spread operado (filtro 2).
        execution: parámetros de ejecución de run_backtest (capital, fraction, commission,
            borrow_rate, trading_days, rebalance_band); ver execution_settings().

    Returns:
        (señales, backtest diario con partición y drawdown, bitácora de operaciones).
    """
    max_holding = max_holding_days(half_life, params["max_holding_multiple"])
    signals = generate_signals(z, tradable, params["z_entry"], params["z_exit"],
                               params["z_stop"], max_holding)
    bt = run_backtest(prices, weights, signals["direction"], **execution)
    bt["split"] = splits.to_numpy()
    bt["drawdown"] = add_drawdown(bt["equity"])
    trades = build_trade_log(bt, signals["event"], splits)
    return signals, bt, trades


def summarize_by_split(bt: pd.DataFrame, trades: pd.DataFrame, tradable: pd.Series) -> pd.DataFrame:
    """Resultado de cada partición dentro de la corrida continua.

    El rendimiento se mide desde el valor al cierre del día previo a la partición; el drawdown,
    contra el máximo alcanzado dentro de la misma partición; las operaciones se asignan a la
    partición en que se abrieron.
    """
    rows = []
    for name in SPLITS:
        mask = (bt["split"] == name).to_numpy()
        start = mask.argmax()
        equity = bt["equity"].iloc[max(start - 1, 0):start + mask.sum()]
        own = trades[trades["split"] == name]
        rows.append({"split": name, "start": bt.index[start].date(), "end": bt.index[mask][-1].date(),
                     "return": float(equity.iloc[-1] / equity.iloc[0] - 1),
                     "net_pnl": float(equity.iloc[-1] - equity.iloc[0]),
                     "sharpe": sharpe_ratio(daily_returns(equity)[tradable[equity.index[1:]]]),
                     "max_drawdown": float(add_drawdown(equity).min()),
                     "trades": int(len(own)),
                     "win_rate": float((own["net_pnl"] > 0).mean()) if len(own) else 0.0})
    return pd.DataFrame(rows).set_index("split")
