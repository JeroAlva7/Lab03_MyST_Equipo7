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

### Resultados principales

Backtest continuo con costos, umbrales calibrados solo en entrenamiento (z de entrada 2.0, salida al cruzar
0, stop en |z| = 4 y tenencia máxima de 63 días). Detalle y análisis en [`docs/reporte.pdf`](docs/reporte.pdf).

| | Entrenamiento | Prueba | Validación | Muestra completa |
|---|---|---|---|---|
| Periodo | 30-ene-2012 a 27-oct-2020 | 28-oct-2020 a 26-oct-2023 | 27-oct-2023 a 30-oct-2026 | 2012-2026 |
| Rendimiento anual | 1.59% | 0.90% | 0.65% | 1.26% |
| Sharpe | 0.41 | 0.32 | 0.30 | 0.37 |
| Drawdown máximo | −6.7% | −5.2% | −2.9% | −6.7% |
| Operaciones (win rate) | 19 (74%) | 5 (80%) | 5 (60%) | 29 (72%) |
| ADF p del spread de Kalman | < 0.0001 | 0.011 | 0.002 | |
| ADF p del spread con pesos fijos | 0.0002 | 0.003 | 0.054 | |

La relación es real pero débil: con pesos fijos el spread deja de ser estacionario en validación y con los
pesos del filtro 1 no; el Johansen móvil solo detecta cointegración en 19% de las ventanas de tres años. Las
comisiones consumen 26% de la ganancia bruta, y las dos operaciones del COVID aportan 61% de la ganancia de
entrenamiento. Con 10 operaciones fuera de muestra el Sharpe no es estadísticamente distinto de cero.

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

```bash
python main.py                  # una vez: genera artifacts/
streamlit run dashboard/app.py  # abre http://localhost:8501
```

El dashboard (`dashboard/app.py`) **solo lee `artifacts/`**: no vuelve a correr el backtest. Si falta algún
archivo, muestra el aviso "Corra primero: python main.py". Todas las gráficas son interactivas (Plotly):
arrastre para hacer zoom, doble clic para regresar, pase el cursor para ver valores y haga clic en la leyenda
para ocultar o mostrar series. En todas, las líneas punteadas verticales marcan el inicio de prueba y de
validación.

**Barra lateral.** Grupo de activos, capital y los parámetros que eligió el grid search en entrenamiento
(z de entrada, z de salida, stop, tenencia máxima y política de rebalanceo).

**Pestaña "Vista ejecutiva"** (portfolio manager o comité de inversión):

1. **Periodo:** el selector de arriba cambia la tarjeta de desempeño y la atribución de costos entre la muestra
   completa, entrenamiento, prueba y validación.
2. **Tarjeta de desempeño:** Sharpe, Sortino, Calmar, drawdown máximo, win rate y profit factor; abajo,
   rendimiento total y anual, volatilidad, P&L neto, número de operaciones y periodo.
3. **Curva de capital:** valor del portafolio neto de costos, un color por partición, con el drawdown debajo.
4. **Atribución de costos:** cascada de P&L bruto → comisiones → préstamo → P&L neto del periodo elegido.
5. **Bitácora de operaciones:** filtros por periodo, dirección del spread (larga o corta), resultado y motivo de
   salida (reversión, stop o tiempo). Debajo de la tabla, el P&L neto de las operaciones filtradas.
6. **Distribución de retornos:** histograma de los retornos por operación con la normal superpuesta;
   **responde a los filtros de la bitácora** (por ejemplo, solo prueba y validación).

**Pestaña "Vista técnica"** (investigador cuantitativo o gestor de riesgos):

1. **Spread con señales:** spread Sₜ del filtro 1 con entradas largas (▲), cortas (▼) y salidas por motivo
   (✕), en la fecha en que se decide; se ejecutan al cierre del día siguiente.
2. **z-score con umbrales:** líneas de entrada (±2), salida (0) y stop (±4).
3. **Vector de pesos β̂ₜ:** una línea por activo (DUK fijo en +1).
4. **Eigenvector de Johansen móvil:** componentes del primer eigenvector en ventanas de 3 años; los cuadros
   verdes de abajo marcan las ventanas en que la traza rechaza "no cointegración".
5. **Ganancia de Kalman:** magnitud de la ganancia de cada peso (filtro 1) y de la media μₜ (filtro 2, escala log).
6. **Varianza posterior:** incertidumbre de cada estado en el tiempo (convergencia de los filtros).
7. **Sharpe móvil de 252 días.**
8. **Costo del rebalanceo** (desplegable): rebalanceo diario contra bandas de 2%, 5% y 10% y sin rebalanceo.

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
- **Grupo final:** el tamaño lo fija la sección 3.1, 2 × n = 4 activos (`config.GROUP_SIZE`); entre los
  aprobados de ese tamaño se elige el de menor vida media (el más reversivo). El mejor par se reporta aparte
  para comparar.
- **Universo:** se eligió después de evaluar 6 universos de un solo sector con el mismo embudo
  (66 grupos). Todos se reportan en `artifacts/universe_search.csv`.
- **Número de relaciones:** en el grupo final Johansen identifica **una sola relación** (r = 1 por traza y por
  máximo eigenvalor), así que solo hay un vector candidato y no aplica la tabla comparativa entre columnas.
- **Pruebas múltiples:** con 66 grupos al 5% se esperan ~3.3 falsos positivos por azar. Los 5 aprobados están
  todos en utilities (0 de 55 en los otros sectores), lo que apunta a una relación económica común y no a
  falsos positivos dispersos. Con corrección de Bonferroni (0.05/66) solo DUK-AEP pasa Engle-Granger; la traza
  de Johansen del grupo final (55.7) supera el valor crítico del 99% (54.7) por poco. Se declara como
  limitación: la evidencia en entrenamiento es significativa pero no abrumadora.

### Filtro de Kalman 1 (`kalman/weights_filter.py`)
- El estado incluye un **intercepto cₜ** además de los pesos libres (SO, AEP, XEL); el peso de DUK queda fijo
  en +1. Así el spread es Sₜ = cₜ + vₜ.
- Transición de caminata aleatoria (A = I) con **Q = 10⁻⁸·I, fijada por diseño antes del backtest**:
  elegirla por máxima verosimilitud hacía que los pesos absorbieran la reversión del spread (ver
  `docs/powell_formulacion.md`). Con 10⁻⁸ los pesos se mueven lento y el spread de Kalman es estacionario
  en entrenamiento. Q **no entra al grid search**; ver "Calibración" abajo.
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
  filtro de Kalman 2**: las señales se generan sobre su z-score. El VECM se estima en entrenamiento con el
  vector de Johansen fijo como término de corrección de error (OLS ecuación por ecuación) y se usa para
  reportar las velocidades de ajuste α y verificar −2 < β'α < 0 (β'α = −0.026, vida media de 26 días).
- **Entrada:** spread largo (+1) si −z_stop ≤ zₜ < −z_entrada; corto (−1) si z_entrada < zₜ ≤ z_stop. No se
  entra si |zₜ| ya pasó el stop.
- **Salida:** se cierra cuando dirección·zₜ > −z_salida. Para z_salida > 0 es la regla del enunciado
  (|zₜ| < z_salida) **más** el caso en que el z-score cruza la banda de un día a otro y aparece del otro
  lado; sin esa extensión la posición quedaría abierta justo cuando ya se cumplió la reversión. Con
  z_salida = 0 equivale a "salir al cruzar la media".
- **Stop:** |zₜ| > z_stop = 4, fijo como límite de riesgo (no se optimiza).
- **Tenencia máxima:** múltiplo de la vida media del filtro 2 (15.5 días), redondeado hacia arriba.
- Después de un stop o de una salida por tiempo, no se vuelve a entrar hasta que |zₜ| regrese por debajo de
  z_entrada (**re-armado**): así no se reabre la misma divergencia que acaba de fallar.
- La señal del día t usa solo zₜ (conocido al cierre de t) y se ejecuta al cierre de t + 1.
- **Sizing:** Nᵢ = 0.8 · V · β̂ᵢ / Σ|β̂ⱼ|, con V el valor del portafolio y β̂ los pesos del filtro 1 del
  día de la decisión. Se permiten **acciones fraccionarias** (simplificación estándar con $1,000,000 de
  capital: el error de redondeo es de centavos por pata).

### Backtest y costos (`backtest/engine.py`)
- **Ejecución:** la orden decidida al cierre de t − 1 se ejecuta al precio de cierre de t, con los pesos de
  t − 1. Ninguna decisión usa el precio con el que se ejecuta.
- **Comisión:** 0.125% del nocional operado en **cada pata**, a la entrada, a la salida y en cada
  rebalanceo.
- **Préstamo:** 0.25% anual / 252 sobre el nocional corto al cierre de cada día con posición abierta.
- **Rebalanceo:** mientras la posición está abierta se rebalancea **diario** al objetivo con los β̂ más
  recientes (política principal). Como comparación, `artifacts/rebalancing.csv` repite la corrida con
  bandas de no-rebalanceo de 2%, 5% y 10% y sin rebalanceo.
- **Walk-forward:** una sola corrida continua sobre las tres particiones con los parámetros congelados de
  entrenamiento; los filtros siguen actualizándose día a día con información pasada. Las operaciones se
  asignan a la partición en que se abrieron.
- **Métricas:** rendimientos diarios del valor total (los días sin posición cuentan como 0), anualización
  con 252 días y tasa libre de riesgo 0. Sharpe móvil con ventana de 252 días.

### Calibración (`backtest/grid_search.py`)
- **Grid search solo en entrenamiento** sobre z_entrada ∈ {1, 1.25, 1.5, 1.75, 2, 2.5},
  z_salida ∈ {0, 0.25, 0.5, 0.75} y tenencia máxima ∈ {2, 3, 4} vidas medias (72 combinaciones válidas con
  z_salida < z_entrada). Criterio: **Sharpe neto de costos**.
- Para competir, una combinación debe hacer **al menos 15 operaciones** en entrenamiento: con menos, el
  Sharpe depende de un puñado de operaciones y no es confiable (por eso z_entrada = 2.5, con 8–9
  operaciones, queda fuera aunque tenga el Sharpe más alto).
- Elegida: z_entrada = 2.0, z_salida = 0, tenencia de 4 vidas medias (63 días). La figura A1
  (`docs/figures/figA1_grid_robustez.png`) muestra que la elección está en una zona estable: z_salida = 0
  es la mejor columna para cualquier z_entrada y tenencia.
- **La Q del filtro 1 no entra al grid.** Optimizarla junto con los umbrales elige Q = 10⁻⁹ (Sharpe de
  entrenamiento 0.70 contra 0.41), pero con esa Q los pesos casi no se adaptan y el spread de Kalman deja de
  ser estacionario fuera de muestra; su Sharpe en prueba y validación es menor. Es sobreajuste: el grid
  premia la Q que mejor explica el pasado. Esta comparación se reporta **ex post** en
  `artifacts/q_sensitivity.csv` y no se usó para elegir.

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
- **Documentación:** redacción del README, del charter, del sprint, de la formulación de Powell, del reporte
  ejecutivo (`docs/reporte.pdf`) y de la parte del post-mortem de Jerónimo.

Cada resultado fue ejecutado, revisado y validado por el equipo antes de subirse al repositorio. Las decisiones
de modelado (universo, partición, criterios del embudo, calibración de Q y R) se discutieron y se tomaron como
equipo, y ambos integrantes pueden explicar cualquier parte del código.
