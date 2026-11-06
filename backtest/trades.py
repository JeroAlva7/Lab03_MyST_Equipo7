"""Bitácora de operaciones (sección 3.6 y vista ejecutiva del dashboard).

Una operación va del día en que se ejecuta la entrada al día en que se ejecuta la salida.
Mientras no hay posición el valor del portafolio no cambia (solo hay efectivo y no se paga
préstamo), así que toda la ganancia o pérdida de la estrategia ocurre dentro de alguna
operación y la suma de la bitácora cuadra con el resultado total.

Para cada operación:
    P&L neto  = valor al cierre del día de salida - valor al cierre del día previo a la entrada
    costos    = comisiones de entrada y salida + préstamo devengado mientras estuvo abierta
    P&L bruto = P&L neto + costos  (lo que se habría ganado sin costos con esas mismas posiciones)
    retorno   = P&L neto / valor del portafolio antes de entrar
"""
import numpy as np
import pandas as pd

DIRECTION_NAMES = {1: "larga", -1: "corta"}


def build_trade_log(backtest: pd.DataFrame, events: pd.Series, splits: pd.Series) -> pd.DataFrame:
    """Arma una fila por operación a partir de la salida de backtest.engine.run_backtest.

    Args:
        backtest: resultado diario del backtest (direction, equity, commission, borrow_cost).
        events: columna "event" de las señales; el motivo de salida es el evento del día
            anterior a la ejecución (la señal se decide un día antes de ejecutarse).
        splits: partición de cada fecha ("train", "test", "val").

    Returns:
        Tabla con fechas de entrada y salida, partición, dirección, días, motivo de salida,
        P&L bruto, comisiones, préstamo, P&L neto, retorno y resultado (ganadora o perdedora).
    """
    direction = backtest["direction"].to_numpy()
    entries = np.flatnonzero((direction != 0) & (np.r_[0, direction[:-1]] != direction))
    rows = []

    for start in entries:
        open_days = np.flatnonzero(direction[start:] != direction[start])
        end = start + open_days[0] if len(open_days) else len(direction) - 1
        is_closed = len(open_days) > 0
        window = backtest.iloc[start:end + 1]
        equity_before = backtest["equity"].iloc[start - 1]
        net = backtest["equity"].iloc[end] - equity_before
        commission, borrow = window["commission"].sum(), window["borrow_cost"].sum()
        rows.append({
            "entry_date": backtest.index[start],
            "exit_date": backtest.index[end] if is_closed else pd.NaT,
            "split": splits.iloc[start],
            "direction": DIRECTION_NAMES[int(direction[start])],
            "days": int(end - start),
            "exit_reason": events.iloc[end - 1] if is_closed else "abierta",
            "gross_pnl": net + commission + borrow,
            "commission": commission,
            "borrow_cost": borrow,
            "net_pnl": net,
            "return": net / equity_before,
            "result": "ganadora" if net > 0 else "perdedora",
        })
    return pd.DataFrame(rows)
