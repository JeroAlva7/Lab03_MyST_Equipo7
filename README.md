# Lab03_MyST_Equipo7 — Arbitraje estadístico con cointegración y filtros de Kalman

Microestructuras y Sistemas de Trading · ITESO · Módulo 3

## Integrantes

| Integrante | Usuario de GitHub | Responsabilidades principales |
|---|---|---|
| Jerónimo Rojas | [@JeroAlva7](https://github.com/JeroAlva7) | Datos, selección por cointegración, filtros de Kalman, pruebas |
| Raúl Zanatta | [@rzanatta0](https://github.com/rzanatta0) | VECM y señales, backtest, métricas, dashboard |

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

Decisiones de modelado que no fija el enunciado o que lo interpretan. Cada una está implementada en el
módulo indicado y sus valores viven en `config.py`.

### Datos (`data/`)
- Se usan **precios de cierre ajustados por dividendos y splits** (Yahoo Finance). En corto se pagan los
  dividendos, y el precio ajustado ya incorpora ese efecto.
- Todos los modelos trabajan con **log-precios**: así los pesos se interpretan en términos porcentuales y la
  varianza es más estable que en niveles.
- Huecos de hasta 3 días se rellenan **solo hacia adelante** (nunca hacia atrás, para no usar información
  futura); en la muestra no hubo ninguno.
- La partición 60/20/20 es **cronológica y por posición**. Todo parámetro (vector de Johansen, Q, R, φ,
  umbrales) se estima **solo con entrenamiento**.

### Selección del grupo (`cointegration/`)
- **Correlación:** mediana de la correlación móvil de 252 días de los **log-retornos** de cada par; el grupo
  pasa si el promedio de sus pares supera 0.7. Se usan retornos y no precios para evitar correlaciones
  espurias por tendencia.
- **Engle-Granger:** se prueba cada activo como variable dependiente y el grupo avanza si alguna orientación
  rechaza H₀ (p < 0.05). Al ser un pre-filtro, es permisivo; Johansen es la prueba definitiva.
- **Johansen:** con constante (`det_order = 0`); rezagos del VECM = p − 1, con p elegido por BIC en un VAR en
  niveles (mínimo 1); rango determinado por el estadístico de traza al 95%.
- **Normalización:** el vector se divide entre la pata de mayor valor absoluto en entrenamiento (DUK = +1),
  lo que fija la escala y el signo.
- **Validación del spread:** ADF con p < 0.05 y vida media entre 5 y 60 días.
- **Grupo final:** entre los aprobados de 3 o 4 activos, el de menor vida media; el mejor par se reporta
  aparte para comparar.
- **Universo:** se eligió después de evaluar 6 universos de un solo sector con el mismo embudo
  (66 grupos). Todos se reportan en `artifacts/universe_search.csv`.

### Filtro de Kalman 1 (`kalman/weights_filter.py`)
- El estado incluye un **intercepto cₜ** además de los pesos libres (SO, AEP, XEL); el peso de DUK queda fijo
  en +1. Así el spread es Sₜ = cₜ + vₜ.
- Transición de caminata aleatoria (A = I) con **Q = 10⁻⁸·I, fijada por diseño**: elegirla por máxima
  verosimilitud hacía que los pesos absorbieran la reversión del spread (ver `docs/powell_formulacion.md`).
- **R** = varianza del spread de Johansen en entrenamiento. Inicialización en el vector de Johansen con
  P₀ = 10⁻⁶·I.
- El spread se calcula con los **pesos prior** (información hasta t − 1).
- Se suponen ruidos gaussianos e independientes, supuesto estándar del filtro de Kalman.

### Filtro de Kalman 2 (`kalman/spread_filter.py`)
- El spread revierte como un **AR(1) con φ constante**, estimado por OLS en entrenamiento (φ = 0.956,
  vida media de 15.5 días). Por eso la observación del filtro es yₜ = Sₜ − φSₜ₋₁ = (1 − φ)μₜ + eₜ.
- **Q** se elige por máxima verosimilitud en entrenamiento; **R** es la varianza de los residuales del AR(1).
- La varianza σₜ² se actualiza con un **EWMA (λ = 0.97)** de las innovaciones, en paralelo a la media.
- El z-score usa **media y desviación prior**: zₜ = (Sₜ − μ̂ₜ|ₜ₋₁) / σ̂ₜ|ₜ₋₁.
- **Burn-in de 60 días** (unas cuatro vidas medias): no se opera antes del 30 de enero de 2012.

### Señales y VECM (`strategy/`)
- Por indicación del profesor, los "residuales" de la sección 3.6 corresponden a la **salida normalizada del
  filtro de Kalman 2**: las señales se generan sobre su z-score. El VECM se estima con el vector de Johansen
  fijo como término de corrección de error y se usa para reportar las velocidades de ajuste α.
- La señal del día t usa solo zₜ (conocido al cierre de t) y se ejecuta a partir de t + 1.

### Pruebas y diagnóstico (`tests/`, `cointegration/stability.py`)
- **Prueba 4:** tolerancia declarada de varianza de las innovaciones normalizadas entre 0.8 y 1.2 y Ljung-Box
  (10 rezagos) con p > 0.05, evaluada **fuera de muestra** (prueba y validación). En entrenamiento Ljung-Box
  rechaza por la volatilidad de marzo de 2020.
- **Johansen móvil:** ventanas de 756 días (3 años) que avanzan 21 días; el eigenvector se escala a norma 1
  con DUK positivo para graficarlo sin que se dispare cuando el componente de DUK es pequeño.

## Uso de herramientas de IA

Se usó **Claude (Anthropic)** como asistente durante el proyecto:

- **Análisis:** lectura de los lineamientos, la rúbrica y el material de clase (S11 a S14) para organizar el
  plan de trabajo y la estructura del repositorio.
- **Código:** redacción de primeras versiones de los módulos de datos, cointegración, filtros de Kalman,
  pruebas, figuras y estrategia, a partir de las ecuaciones vistas en clase.
- **Verificación:** comprobación independiente de resultados (por ejemplo, Johansen replicado en numpy y
  reconstrucción paso a paso de las ecuaciones de Kalman) y revisión del cumplimiento de la rúbrica.
- **Documentación:** redacción del README, del charter, del sprint y de la formulación de Powell.

Cada resultado fue ejecutado, revisado y validado por el equipo antes de subirse al repositorio. Las decisiones
de modelado (universo, partición, criterios del embudo, calibración de Q y R) se discutieron y se tomaron como
equipo, y ambos integrantes pueden explicar cualquier parte del código.
