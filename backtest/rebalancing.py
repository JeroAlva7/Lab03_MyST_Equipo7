"""Costo del rebalanceo diario contra políticas de banda de no-rebalanceo (sección 3.6).

Cada política corre la misma estrategia (mismos umbrales y señales) y solo cambia cuándo se
ajustan las patas mientras la posición está abierta. Se compara cuánto se paga en comisiones
de rebalanceo y qué tan lejos queda la exposición del 80% objetivo.
"""
import pandas as pd

from backtest.walk_forward import run_strategy
from performance.metrics import daily_returns, sharpe_ratio


def compare_rebalancing(prices: pd.DataFrame, weights: pd.DataFrame, z: pd.Series,
                        tradable: pd.Series, splits: pd.Series, params: dict, half_life: float,
                        costs: dict, policies: dict) -> pd.DataFrame:
    """Corre la estrategia con cada política de rebalanceo y resume costos y exposición.

    Args:
        policies: {nombre: banda} con banda = 0 (diario), > 0 (banda) o None (nunca).
        costs: argumentos de run_backtest; aquí se sustituye "rebalance_band".

    Returns:
        Una fila por política: días con rebalanceo, comisiones de entrada/salida y de
        rebalanceo, préstamo, P&L neto, Sharpe y exposición bruta (mínima, media, máxima)
        como fracción del valor del portafolio mientras hay posición.
    """
    rows = []
    for name, band in policies.items():
        _, bt, _ = run_strategy(prices, weights, z, tradable, splits, params, half_life,
                                {**costs, "rebalance_band": band})
        open_days = bt[bt["direction"] != 0]
        exposure = (open_days["long_notional"] - open_days["short_notional"]) / open_days["equity"]
        rows.append({"policy": name, "band": band, "rebalance_days": int(bt["rebalanced"].sum()),
                     "commission_trades": float(bt.loc[bt["traded"], "commission"].sum()),
                     "commission_rebalance": float(bt.loc[bt["rebalanced"], "commission"].sum()),
                     "borrow_cost": float(bt["borrow_cost"].sum()),
                     "net_pnl": float(bt["equity"].iloc[-1] - bt["equity"].iloc[0]),
                     "sharpe": sharpe_ratio(daily_returns(bt["equity"][tradable])),
                     "exposure_min": float(exposure.min()), "exposure_mean": float(exposure.mean()),
                     "exposure_max": float(exposure.max())})
    return pd.DataFrame(rows).set_index("policy")
