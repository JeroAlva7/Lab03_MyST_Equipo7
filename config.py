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
TICKERS: list[str] = []          # Se definen en el paso de selección de activos
START_DATE = "2011-01-01"        # Ventana de ~15 años
END_DATE = "2025-12-31"
SPLIT = (0.60, 0.20, 0.20)       # Entrenamiento / prueba / validación (cronológico)

# --- Costos y capital -------------------------------------------------------
COMMISSION = 0.00125             # 0.125% por pata, a la entrada y a la salida
BORROW_RATE_ANNUAL = 0.0025      # 0.25% anual sobre el nocional corto
TRADING_DAYS = 252               # Devengo diario = tasa anual / 252
INITIAL_CAPITAL = 1_000_000
CAPITAL_FRACTION = 0.80          # 80% del capital desplegado en el spread
