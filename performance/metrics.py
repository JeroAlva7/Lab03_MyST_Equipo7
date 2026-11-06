"""Métricas de desempeño (sección 3.8: tarjeta de la vista ejecutiva y Sharpe móvil).

Convenciones:
- Rendimientos diarios del valor total del portafolio (los días sin posición valen 0) y
  anualización con 252 días hábiles. Tasa libre de riesgo cero.
- Win rate y profit factor se calculan por operación con el P&L neto de la bitácora.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    """Rendimiento diario del valor del portafolio."""
    return equity.pct_change().dropna()


def drawdown(equity: pd.Series) -> pd.Series:
    """Caída porcentual del valor del portafolio desde su máximo previo (0 en cada máximo)."""
    return (equity / equity.cummax() - 1).rename("drawdown")


def max_drawdown(equity: pd.Series) -> float:
    """Peor caída desde un máximo (número negativo)."""
    return float(drawdown(equity).min())


def annualized_return(equity: pd.Series) -> float:
    """Rendimiento compuesto anual: (valor final / inicial)^(252 / días) - 1."""
    days = len(equity) - 1
    return float((equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / days) - 1) if days > 0 else 0.0


def sharpe_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Sharpe anualizado: media / desviación estándar x sqrt(252)."""
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(periods)) if std > 0 else 0.0


def sortino_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Sortino anualizado: como el Sharpe, pero solo penaliza la volatilidad de las pérdidas.

    Desviación a la baja = sqrt(promedio de min(r, 0)^2) sobre todos los días.
    """
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    return float(returns.mean() / downside * np.sqrt(periods)) if downside > 0 else 0.0


def calmar_ratio(equity: pd.Series) -> float:
    """Calmar: rendimiento anual / |drawdown máximo|."""
    mdd = max_drawdown(equity)
    return annualized_return(equity) / abs(mdd) if mdd < 0 else 0.0


def win_rate(trade_pnl: pd.Series) -> float:
    """Fracción de operaciones con P&L neto positivo."""
    return float((trade_pnl > 0).mean()) if len(trade_pnl) else 0.0


def profit_factor(trade_pnl: pd.Series) -> float:
    """Ganancias totales / pérdidas totales de las operaciones (inf si no hubo pérdidas)."""
    gains, losses = trade_pnl[trade_pnl > 0].sum(), -trade_pnl[trade_pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def rolling_sharpe(equity: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """Sharpe anualizado en ventanas móviles de `window` días (vista técnica)."""
    returns = daily_returns(equity)
    roll = returns.rolling(window)
    return (roll.mean() / roll.std() * np.sqrt(TRADING_DAYS)).rename("rolling_sharpe")


def performance_summary(equity: pd.Series, trade_pnl: pd.Series) -> dict:
    """Tarjeta de desempeño: todas las métricas de un tramo de la curva de capital.

    Args:
        equity: valor del portafolio, empezando el día previo al tramo evaluado.
        trade_pnl: P&L neto de las operaciones abiertas en ese tramo.
    """
    returns = daily_returns(equity)
    return {"return": float(equity.iloc[-1] / equity.iloc[0] - 1),
            "net_pnl": float(equity.iloc[-1] - equity.iloc[0]),
            "annual_return": annualized_return(equity),
            "annual_volatility": float(returns.std() * np.sqrt(TRADING_DAYS)),
            "sharpe": sharpe_ratio(returns), "sortino": sortino_ratio(returns),
            "calmar": calmar_ratio(equity), "max_drawdown": max_drawdown(equity),
            "trades": int(len(trade_pnl)), "win_rate": win_rate(trade_pnl),
            "profit_factor": profit_factor(trade_pnl)}
