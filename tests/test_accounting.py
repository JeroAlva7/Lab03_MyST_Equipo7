"""Prueba 3 — Contabilidad multi-pata.

1. El valor del portafolio es igual al efectivo más el valor de todas las patas, todos los días.
2. El costo total registrado coincide con el nocional operado en cada pata por la comisión,
   más el devengo diario del préstamo sobre las patas cortas (recalculados aquí desde las
   acciones y los precios, sin usar las columnas de costos del motor).
3. Un ejemplo pequeño calculado a mano.
"""
import numpy as np
import pandas as pd
import pytest

import config
from backtest.engine import run_backtest
from kalman.pipeline import order_assets, run_filters
from strategy.signals import generate_signals, max_holding_days
from strategy.sizing import kalman_weights


@pytest.fixture(scope="module")
def backtest_run(prices, log_prices, n_train, vector):
    """Backtest con los datos reales, los filtros y los parámetros de config.py."""
    _, beta0 = order_assets(vector)
    f1, f2, params = run_filters(log_prices, beta0, n_train, config.KALMAN)
    assets = list(log_prices.columns)
    tradable = pd.Series(np.arange(len(f2)) >= config.KALMAN["burn_in"], index=f2.index)
    s = config.SIGNALS
    max_holding = max_holding_days(params["filter2"]["half_life"], s["max_holding_multiple"])
    signals = generate_signals(f2["z"], tradable, s["z_entry"], s["z_exit"], s["z_stop"], max_holding)
    bt = run_backtest(prices[assets], kalman_weights(f1, assets), signals["direction"],
                      config.INITIAL_CAPITAL, config.CAPITAL_FRACTION, config.COMMISSION,
                      config.BORROW_RATE_ANNUAL, config.TRADING_DAYS, config.REBALANCE_BAND)
    return bt, prices[assets].to_numpy(), assets


def test_value_equals_cash_plus_legs(backtest_run):
    bt, p, assets = backtest_run
    shares = bt[[f"shares_{a}" for a in assets]].to_numpy()
    assert np.allclose(bt["equity"], bt["cash"] + (shares * p).sum(axis=1), rtol=0, atol=1e-6)


def test_costs_match_legs_and_borrow(backtest_run):
    bt, p, assets = backtest_run
    shares = bt[[f"shares_{a}" for a in assets]].to_numpy()
    traded = np.abs(np.diff(shares, axis=0, prepend=0.0)) * p      # nocional operado por pata
    short = np.where(shares < 0, -shares * p, 0.0).sum(axis=1)    # nocional corto al cierre

    expected_commission = config.COMMISSION * traded.sum()
    expected_borrow = config.BORROW_RATE_ANNUAL / config.TRADING_DAYS * short.sum()
    recorded = bt["commission"].sum() + bt["borrow_cost"].sum()

    assert bt["legs_traded"].sum() == (traded > 0).sum()
    moves = bt["traded"] | bt["rebalanced"]
    assert (bt.loc[bt["traded"], "legs_traded"] == len(assets)).all()   # entrada/salida: 4 patas
    assert (bt.loc[~moves, "commission"] == 0).all()                     # sin operación no hay comisión
    assert np.isclose(recorded, expected_commission + expected_borrow, rtol=0, atol=1e-6)

    # El efectivo final cuadra: capital - flujos de compra/venta - costos
    trade_flows = (np.diff(shares, axis=0, prepend=0.0) * p).sum()
    assert np.isclose(bt["cash"].iloc[-1], config.INITIAL_CAPITAL - trade_flows - recorded,
                      rtol=0, atol=1e-6)


def test_hand_computed_example():
    """Dos activos, beta = (1, -0.5), comisión 1%, préstamo 0.1% diario (25.2% anual / 252),
    sin rebalanceo para que las cuentas se puedan seguir a mano.

    Día 1: entra largo con valor 1000 -> A: +533.33 (53.33 acc a 10), B: -266.67 (-13.33 acc a 20)
           comisión 1% x 800 = 8; préstamo 0.1% x 266.67 = 0.2667
    Día 2: solo préstamo 0.2667
    Día 3: sale (A a 12, B a 22): comisión 1% x (640 + 293.33) = 9.3333
    P&L bruto = 53.33 x 2 - 13.33 x 2 = 80; costos = 8 + 9.3333 + 2 x 0.2667 = 17.8667
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"A": [10.0, 10.0, 11.0, 12.0], "B": [20.0, 20.0, 20.0, 22.0]}, index=dates)
    weights = pd.DataFrame({"A": 1.0, "B": -0.5}, index=dates)
    direction = pd.Series([1, 1, 0, 0], index=dates)     # decide entrar el día 0 y salir el día 2

    bt = run_backtest(prices, weights, direction, capital=1000.0, fraction=0.8,
                      commission=0.01, borrow_rate=0.252, trading_days=252, rebalance_band=None)

    assert np.allclose(bt["direction"], [0, 1, 1, 0])
    assert np.isclose(bt["commission"].sum(), 8 + 28 / 3)
    assert np.isclose(bt["borrow_cost"].sum(), 2 * 0.8 / 3)
    assert np.isclose(bt["equity"].iloc[-1], 1000 + 80 - (8 + 28 / 3 + 1.6 / 3))
