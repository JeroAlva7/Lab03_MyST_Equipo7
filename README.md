# Lab03_MyST_Equipo7 — Arbitraje estadístico con cointegración y filtros de Kalman

Microestructuras y Sistemas de Trading · ITESO · Módulo 3

## Integrantes

| Integrante | Usuario de GitHub | Responsabilidades principales |
|---|---|---|
| Jerónimo Rojas | [@JeroAlva7](https://github.com/JeroAlva7) | Datos, selección por cointegración, filtros de Kalman, pruebas |
| Raúl Zanatta | [@rzanatta0] | VECM y señales, backtest, métricas, dashboard |

Ambos integrantes participan en el reporte, la presentación y la revisión cruzada del código.

## Nivel de alcance

**Nivel A.** Un solo grupo cointegrado de 4 activos (2 × 2 integrantes). La sección 3.7 (Risk Parity entre
varios grupos) se omite por indicación del profesor; por lo mismo no aplican la figura 9 ni la parte de Nivel B
de la prueba 4.

## Descripción del proyecto

Estrategia de arbitraje estadístico sobre un portafolio de 4 activos cointegrados. Johansen identifica el
vector cointegrante, el filtro de Kalman 1 lo vuelve dinámico (β̂ₜ), el filtro de Kalman 2 estima la media y
la varianza del spread para un z-score causal, y un VECM con reglas de entrada, salida, stop y tenencia máxima
genera las operaciones. Ambos filtros y el sistema completo se formulan como procesos de decisión secuencial
(Powell).

### Universo de activos

**Utilities eléctricas reguladas de EE. UU.: Duke Energy (DUK), Southern Company (SO), American Electric Power
(AEP) y Xcel Energy (XEL).** Sus utilidades las fija el regulador mediante un ROE autorizado, por lo que se
comportan de forma parecida a los bonos y comparten un mismo factor común: las tasas de interés. Esa es la
tendencia estocástica común que la cointegración requiere.

| Parámetro | Valor |
|---|---|
| Datos | Precios diarios ajustados de Yahoo Finance, 1-nov-2011 a 30-oct-2026 (15 años, 3,771 días) |
| Entrenamiento (60%) | 1-nov-2011 a 27-oct-2020 |
| Prueba (20%) | 28-oct-2020 a 26-oct-2023 |
| Validación (20%) | 27-oct-2023 a 30-oct-2026 |

### Selección del grupo

Se evaluaron **6 universos de un solo sector** (petroleras, utilities, consumo básico, ferrocarriles, bancos y
ETFs de tecnología), **66 grupos en total**, con el mismo embudo y solo con datos de entrenamiento:
orden de integración → correlación móvil > 0.7 → Engle-Granger → Johansen → ADF y vida media del spread.
Solo las utilities aprobaron (5 de 11 grupos). El resultado completo está en `artifacts/universe_search.csv`.

| | Grupo final | Mejor par (comparación) |
|---|---|---|
| Activos | DUK, SO, AEP, XEL | DUK, AEP |
| Vector β normalizado | (1, 0.181, −0.614, −0.099) | (1, −0.612) |
| Traza de Johansen (r₀ = 0) | 55.7 > 47.9 (95%) | 26.2 > 15.5 (95%) |
| ADF del spread (p-value) | 0.0002 | 0.0005 |
| Vida media | 24 días | 26 días |

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
├── reporting/              # figuras estáticas del reporte
├── project_management/     # charter, sprints, post-mortem
├── artifacts/              # salidas del backtest que lee el dashboard
├── tests/                  # pruebas con pytest
└── docs/                   # reporte.pdf y figures/ (figuras del reporte)
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
