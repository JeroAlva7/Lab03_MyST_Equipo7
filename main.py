"""Pipeline completo del Laboratorio 03. Se ejecuta con: python main.py

Etapas (cada una vive en su propio paquete):
    1. data/           Descarga, limpieza y validación de precios.
    2. cointegration/  Screening, Engle-Granger y Johansen sobre entrenamiento.
    3. kalman/         Filtro 1 (pesos beta_t) y filtro 2 (media/varianza del spread).
    4. strategy/       VECM, z-score causal, señales y sizing.
    5. backtest/       Simulación con costos y walk-forward.
    6. performance/    Métricas y escritura de artifacts/ para el dashboard.
"""
import json

import numpy as np

import config
from data.loader import load_prices
from data.validation import clean_prices, split_data, validate_prices


def run_data_stage() -> dict:
    """Etapa 1: carga, limpia, valida y parte los precios."""
    print("[1/6] Datos")
    raw = load_prices(config.TICKERS, config.START_DATE, config.END_DATE, config.PRICES_FILE)
    prices, report = clean_prices(raw, config.MAX_FFILL_DAYS)
    validate_prices(prices)
    splits = split_data(prices, config.SPLIT)

    report["splits"] = {
        name: {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "rows": len(df)}
        for name, df in splits.items()
    }
    prices.to_csv(config.ARTIFACTS_DIR / "prices.csv", float_format="%.6f")
    with open(config.ARTIFACTS_DIR / "data_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  {report['rows_clean']} días hábiles, {report['start']} a {report['end']}")
    for name, info in report["splits"].items():
        print(f"  {name:>5}: {info['start']} a {info['end']} ({info['rows']} días)")
    return splits


def main() -> None:
    """Ejecuta todas las etapas del pipeline en orden."""
    np.random.seed(config.SEED)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    print(f"Lab03 MyST — Equipo 7 | semilla = {config.SEED}")
    run_data_stage()


if __name__ == "__main__":
    main()
