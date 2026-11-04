"""Pipeline completo del Laboratorio 03. Se ejecuta con: python main.py

Etapas (cada una vive en su propio paquete):
    1. data/           Descarga, limpieza y validación de precios.
    2. cointegration/  Screening, Engle-Granger y Johansen sobre entrenamiento.
    3. kalman/         Filtro 1 (pesos beta_t) y filtro 2 (media/varianza del spread).
    4. strategy/       VECM, z-score causal, señales y sizing.
    5. backtest/       Simulación con costos y walk-forward.
    6. performance/    Métricas y escritura de artifacts/ para el dashboard.
"""
import numpy as np

import config


def main() -> None:
    """Ejecuta todas las etapas del pipeline en orden."""
    np.random.seed(config.SEED)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    print(f"Lab03 MyST — Equipo 7 | semilla = {config.SEED}")
    print("Esqueleto del pipeline listo; las etapas se agregan en los siguientes pasos.")


if __name__ == "__main__":
    main()
