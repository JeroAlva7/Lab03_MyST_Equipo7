"""Motor de backtest multi-pata (sección 3.6).

Contabilidad: el portafolio tiene efectivo y una cantidad de acciones por activo (positiva =
larga, negativa = corta). Cada día:

    1. Se ejecuta la orden decidida al cierre de AYER (señal de t-1) al precio de cierre de hoy.
       Comprar acciones resta efectivo; vender en corto suma efectivo (se recibe el producto de
       la venta). Los pesos usados son los beta de ayer, conocidos al tomar la decisión.
       Comisión: 0.125% del nocional operado en CADA pata, a la entrada y a la salida.
    2. Préstamo: las patas cortas que quedan abiertas al cierre devengan la tasa anual / 252
       sobre su nocional de ese día (un día de préstamo por cada noche en corto).
    3. Se valúa el portafolio a precios de hoy:  valor = efectivo + sum(acciones_i * precio_i).
       Ambos costos ya salieron del efectivo, así que el valor es neto de costos.

Así ninguna decisión usa el precio con el que se ejecuta (sin look-ahead). Mientras la posición
está abierta las acciones no cambian; el rebalanceo diario contra una banda se agrega aparte.
"""
import numpy as np
import pandas as pd

from strategy.sizing import target_shares


def run_backtest(prices: pd.DataFrame, weights: pd.DataFrame, direction: pd.Series,
                 capital: float, fraction: float, commission: float, borrow_rate: float,
                 trading_days: int) -> pd.DataFrame:
    """Simula la estrategia día por día.

    Args:
        prices: precios (no logaritmos) por activo, mismo orden de columnas que `weights`.
        weights: vector beta_t del filtro 1 (strategy.sizing.kalman_weights).
        direction: dirección decidida al cierre de cada día (columna "direction" de las señales).
        capital: capital inicial.
        fraction: fracción del capital que se despliega al abrir (0.80).
        commission: comisión por pata como fracción del nocional operado (0.00125).
        borrow_rate: tasa anual de préstamo sobre el nocional corto (0.0025).
        trading_days: días hábiles por año para el devengo diario (252).

    Returns:
        Por fecha: dirección vigente, acciones de cada activo, efectivo, nocional largo y
        corto, valor del portafolio, si hubo operación, patas operadas, comisión y préstamo
        pagados ese día.
    """
    assets = list(prices.columns)
    p, w, d = prices.to_numpy(), weights.to_numpy(), direction.to_numpy()
    cash, shares, position = float(capital), np.zeros(len(assets)), 0
    daily_borrow = borrow_rate / trading_days
    rows = []

    for t in range(len(p)):
        order = int(d[t - 1]) if t > 0 else 0          # decisión del cierre de ayer
        traded = order != position
        legs_traded, fee = 0, 0.0
        if traded:
            equity_before = cash + shares @ p[t]
            new_shares = (target_shares(order, w[t - 1], p[t], equity_before, fraction)
                          if order != 0 else np.zeros(len(assets)))
            traded_notional = np.abs(new_shares - shares) * p[t]
            legs_traded = int((traded_notional > 0).sum())
            fee = commission * traded_notional.sum()  # 0.125% en cada pata operada
            cash -= (new_shares - shares) @ p[t] + fee  # comprar resta efectivo, vender suma
            shares, position = new_shares, order

        legs = shares * p[t]
        short_notional = legs[legs < 0].sum()
        borrow = daily_borrow * abs(short_notional)   # devengo de una noche en corto
        cash -= borrow
        rows.append([position, *shares, cash, legs[legs > 0].sum(), short_notional,
                     cash + legs.sum(), traded, legs_traded, fee, borrow])

    columns = (["direction"] + [f"shares_{a}" for a in assets]
               + ["cash", "long_notional", "short_notional", "equity", "traded",
                  "legs_traded", "commission", "borrow_cost"])
    return pd.DataFrame(rows, index=prices.index, columns=columns)


def add_drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown: caída porcentual del valor del portafolio desde su máximo previo."""
    return (equity / equity.cummax() - 1).rename("drawdown")
