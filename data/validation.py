"""Limpieza, validación y partición cronológica de los precios."""
import pandas as pd


def clean_prices(prices: pd.DataFrame, max_ffill_days: int) -> tuple[pd.DataFrame, dict]:
    """Limpia la tabla de precios y documenta cada corrección.

    Pasos: ordena por fecha, elimina fechas duplicadas, rellena hacia adelante
    huecos aislados de hasta `max_ffill_days` días (nunca hacia atrás, para no
    usar información futura) y elimina las fechas que aún tengan algún faltante.

    Returns:
        Tupla con los precios limpios y un diccionario con el reporte de limpieza.
    """
    prices = prices.sort_index()
    duplicates = int(prices.index.duplicated().sum())
    prices = prices[~prices.index.duplicated(keep="first")]

    missing_before = prices.isna().sum().to_dict()
    filled = prices.ffill(limit=max_ffill_days)
    clean = filled.dropna(how="any")

    report = {
        "rows_raw": int(len(prices)),
        "rows_clean": int(len(clean)),
        "duplicates_removed": duplicates,
        "missing_by_asset": {k: int(v) for k, v in missing_before.items()},
        "rows_dropped": int(len(prices) - len(clean)),
        "start": str(clean.index.min().date()),
        "end": str(clean.index.max().date()),
    }
    return clean, report


def validate_prices(prices: pd.DataFrame, min_years: float = 10.0) -> None:
    """Verifica que los precios cumplan los requisitos del laboratorio.

    Raises:
        ValueError: si hay precios no positivos, faltantes, fechas desordenadas
            o una ventana menor a `min_years` años.
    """
    if prices.isna().any().any():
        raise ValueError("Quedan precios faltantes después de la limpieza")
    if (prices <= 0).any().any():
        raise ValueError("Hay precios no positivos; no se puede tomar logaritmo")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Las fechas no están en orden cronológico")
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    if years < min_years:
        raise ValueError(f"La ventana tiene {years:.1f} años; se requieren al menos {min_years}")


def split_data(prices: pd.DataFrame, split: tuple[float, float, float]) -> dict[str, pd.DataFrame]:
    """Divide los datos cronológicamente en entrenamiento, prueba y validación.

    Los cortes se hacen por posición, sin mezclar fechas, de modo que cada
    partición solo contiene fechas posteriores a la anterior (sin look-ahead).
    """
    if abs(sum(split) - 1.0) > 1e-9:
        raise ValueError("Las proporciones de la partición deben sumar 1")
    n = len(prices)
    train_end = int(n * split[0])
    test_end = train_end + int(n * split[1])
    return {
        "train": prices.iloc[:train_end],
        "test": prices.iloc[train_end:test_end],
        "val": prices.iloc[test_end:],
    }
