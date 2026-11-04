"""Descarga de precios diarios ajustados y caché local en CSV."""
import time
from pathlib import Path

import pandas as pd


def download_prices(tickers: list[str], start: str, end: str,
                    retries: int = 3, wait_seconds: int = 10) -> pd.DataFrame:
    """Descarga precios de cierre ajustados por splits y dividendos desde Yahoo Finance.

    Args:
        tickers: Símbolos a descargar.
        start: Fecha inicial (inclusiva), formato AAAA-MM-DD.
        end: Fecha final (exclusiva), formato AAAA-MM-DD.
        retries: Intentos ante errores de red o de límite de solicitudes.
        wait_seconds: Espera entre intentos.

    Returns:
        DataFrame con una columna por activo, índice de fechas.
    """
    import yfinance as yf  # Solo se importa si realmente hay que descargar

    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              progress=False, threads=False)
            prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            prices = prices[tickers]
            if prices.dropna(how="all").empty:
                raise ValueError("Yahoo Finance devolvió una tabla vacía")
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            prices.index.name = "date"
            return prices
        except Exception as error:  # yfinance lanza varios tipos de error de red
            if attempt == retries:
                raise RuntimeError(f"No se pudieron descargar {tickers}: {error}") from error
            print(f"  Intento {attempt} fallido ({error}); reintentando en {wait_seconds} s...")
            time.sleep(wait_seconds)
    raise RuntimeError("Número de intentos inválido")


def load_prices(tickers: list[str], start: str, end: str, cache_file: Path,
                refresh: bool = False) -> pd.DataFrame:
    """Carga precios desde el CSV local; descarga y guarda solo si no existe.

    El CSV se versiona en el repositorio para que el pipeline sea reproducible
    aunque Yahoo Finance cambie o no esté disponible.
    """
    if cache_file.exists() and not refresh:
        prices = pd.read_csv(cache_file, index_col="date", parse_dates=True)
        missing = set(tickers) - set(prices.columns)
        if not missing:
            return prices[tickers]
        print(f"  Faltan {sorted(missing)} en el CSV local; se descargan de nuevo.")

    prices = download_prices(tickers, start, end)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_file, float_format="%.6f")
    print(f"  Datos descargados y guardados en {cache_file.name}")
    return prices
