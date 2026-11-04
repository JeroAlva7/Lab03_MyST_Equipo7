"""Parámetros globales del laboratorio (sección 3.1 de los lineamientos).

Todos los módulos leen sus parámetros de aquí para que el pipeline sea
reproducible desde un único lugar.
"""
from pathlib import Path

# --- Reproducibilidad -------------------------------------------------------
SEED = 42  # Semilla fija para numpy (documentada en el README)

# --- Rutas ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"

# --- Datos ------------------------------------------------------------------
# Universo: las 4 mayores posiciones del ETF XLE (sector energía del S&P 500).
# Integradas (XOM, CVX) y exploración y producción (COP, EOG), expuestas al mismo
# factor común: el precio del crudo.
TICKERS = ["XOM", "CVX", "COP", "EOG"]
START_DATE = "2011-11-01"        # 15 años de datos diarios
END_DATE = "2026-11-01"          # Fecha exclusiva: el último día es el 30-oct-2026
PRICES_FILE = DATA_DIR / "prices.csv"   # Copia local versionada; evita depender de Yahoo
SPLIT = (0.60, 0.20, 0.20)       # Entrenamiento / prueba / validación (cronológico)
MAX_FFILL_DAYS = 3               # Huecos aislados de hasta 3 días se rellenan hacia adelante

# --- Costos y capital -------------------------------------------------------
COMMISSION = 0.00125             # 0.125% por pata, a la entrada y a la salida
BORROW_RATE_ANNUAL = 0.0025      # 0.25% anual sobre el nocional corto
TRADING_DAYS = 252               # Devengo diario = tasa anual / 252
INITIAL_CAPITAL = 1_000_000
CAPITAL_FRACTION = 0.80          # 80% del capital desplegado en el spread
