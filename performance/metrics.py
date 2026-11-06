"""Métricas de desempeño (sección 3.8, tarjeta de la vista ejecutiva)."""
import numpy as np
import pandas as pd


def daily_returns(equity: pd.Series) -> pd.Series:
    """Rendimiento diario del valor del portafolio (los días sin posición valen 0)."""
    return equity.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Sharpe anualizado con tasa libre de riesgo cero: media / desviación x sqrt(252)."""
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(periods)) if std > 0 else 0.0
