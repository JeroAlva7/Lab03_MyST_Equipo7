# Lab03_MyST_Equipo7 — Arbitraje estadístico con cointegración y filtros de Kalman

Microestructuras y Sistemas de Trading · ITESO · Módulo 3

## Integrantes

| Integrante | Usuario de GitHub | Responsabilidades principales |
|---|---|---|
| Jerónimo Rojas | [@JeroAlva7](https://github.com/JeroAlva7) | Datos, selección por cointegración, filtros de Kalman, pruebas |
| _[Nombre integrante 2]_ | _[usuario]_ | VECM y señales, backtest, métricas, dashboard |

Ambos integrantes participan en el reporte, la presentación y la revisión cruzada del código.

## Nivel de alcance

**Nivel A.** Un solo grupo cointegrado de 4 activos (2 × 2 integrantes). La sección 3.7 (Risk Parity entre
varios grupos) se omite por indicación del profesor; por lo mismo no aplican la figura 9 ni la parte de Nivel B
de la prueba 4.

## Descripción del proyecto

_Se completa al cerrar la selección de activos._ Estrategia de arbitraje estadístico sobre un portafolio
multi-activo: Johansen identifica el vector cointegrante, el filtro de Kalman 1 lo vuelve dinámico (β̂ₜ),
el filtro de Kalman 2 estima la media y la varianza del spread para un z-score causal, y un VECM con reglas
de entrada, salida, stop y tenencia máxima genera las operaciones. Ambos filtros y el sistema completo se
formulan como procesos de decisión secuencial (Powell).

## Estructura del repositorio

```
Lab03_MyST_Equipo7/
├── README.md
├── requirements.txt        # dependencias con versiones fijadas
├── .gitignore
├── config.py               # parámetros globales y semilla
├── main.py                 # pipeline completo: python main.py
├── data/                   # ingesta, limpieza y validación
├── cointegration/          # screening, Engle-Granger, Johansen
├── kalman/                 # ambos filtros
├── strategy/               # VECM, z-score, señales, sizing
├── backtest/               # motor con costos y walk-forward
├── performance/            # métricas
├── dashboard/app.py        # Streamlit
├── project_management/     # charter, sprints, post-mortem
├── artifacts/              # salidas del backtest que lee el dashboard
├── tests/                  # pruebas con pytest
└── docs/reporte.pdf
```

## Instalación

Requiere Python 3.12.

```bash
git clone https://github.com/JeroAlva7/Lab03_MyST_Equipo7.git
cd Lab03_MyST_Equipo7
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducción

```bash
python main.py                 # pipeline completo; escribe resultados en artifacts/
pytest                         # pruebas obligatorias
streamlit run dashboard/app.py # dashboard (lee artifacts/, no reejecuta el backtest)
```

**Semilla:** `SEED = 42`, definida en `config.py` y aplicada al inicio de `main.py`.

## Guía de uso del dashboard

_Se completa cuando el dashboard esté terminado._

## Supuestos del modelo

_Se documentan conforme se implementa cada etapa._

## Uso de herramientas de IA

_Se documenta al final indicando en qué partes del proyecto se usó asistencia de IA._
