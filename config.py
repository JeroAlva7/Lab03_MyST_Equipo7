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
# Universo: utilities eléctricas reguladas de EE. UU. (sector XLU). Sus utilidades
# las fija el regulador (ROE autorizado) y se comportan como bonos, así que comparten
# un factor común: las tasas de interés. Se eligió tras evaluar 6 universos de un
# solo sector con el mismo embudo (ver cointegration/universe_search.py).
TICKERS = ["DUK", "SO", "AEP", "XEL"]
START_DATE = "2011-11-01"        # 15 años de datos diarios
END_DATE = "2026-11-01"          # Fecha exclusiva: el último día es el 30-oct-2026
PRICES_FILE = DATA_DIR / "prices.csv"   # Copia local versionada; evita depender de Yahoo
SPLIT = (0.60, 0.20, 0.20)       # Entrenamiento / prueba / validación (cronológico)
MAX_FFILL_DAYS = 3               # Huecos aislados de hasta 3 días se rellenan hacia adelante

# --- Selección por cointegración (sección 3.2) -----------------------------
COINTEGRATION = {
    "alpha": 0.05,               # Nivel de significancia de ADF, Engle-Granger y Johansen
    "min_corr": 0.70,            # Filtro heurístico de correlación (promedio del grupo)
    "corr_window": 252,          # Ventana de la correlación móvil (1 año hábil)
    "max_lags": 10,              # Rezagos máximos al elegir el VECM por BIC
    "half_life_range": (5, 60),  # Vida media operable en días (S14)
}

# --- Búsqueda del universo (documentada para el control de pruebas múltiples) ---
# Cada universo es un solo sector con un factor económico común. Todos se evalúan
# con el mismo embudo y solo con la partición de entrenamiento; el reporte incluye
# todos los grupos probados en todos los universos.
CANDIDATE_UNIVERSES = {
    "petroleras": (["XOM", "CVX", "COP", "EOG"], "Precio del crudo"),
    "utilities": (["DUK", "SO", "AEP", "XEL"], "Tarifas reguladas y tasas de interés"),
    "consumo_basico": (["KO", "PEP", "PG", "CL"], "Demanda inelástica y márgenes estables"),
    "ferrocarriles": (["UNP", "CSX", "NSC", "CNI"], "Volumen de carga en Norteamérica"),
    "bancos": (["JPM", "BAC", "C", "WFC"], "Curva de tasas y ciclo de crédito"),
    "etf_tecnologia": (["XLK", "VGT", "IYW", "IGM"], "Mismas empresas tecnológicas subyacentes"),
}

# --- Costos y capital -------------------------------------------------------
COMMISSION = 0.00125             # 0.125% por pata, a la entrada y a la salida
BORROW_RATE_ANNUAL = 0.0025      # 0.25% anual sobre el nocional corto
TRADING_DAYS = 252               # Devengo diario = tasa anual / 252
INITIAL_CAPITAL = 1_000_000
CAPITAL_FRACTION = 0.80          # 80% del capital desplegado en el spread
